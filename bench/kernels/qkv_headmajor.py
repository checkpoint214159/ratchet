"""Fused QKV projection that writes Q, K and V already in the layout attention wants.

THE SELF-INFLICTED TAX
----------------------
v2 onward computes one fused QKV GEMM producing [B, S, 3D], splits it, and then does
`.view(B, S, H, hd).transpose(1, 2)` to get [B, H, S, hd]. That transpose is a VIEW, so
FlashAttention reads q/k/v through a non-contiguous stride pattern. Measured at config 6's
shape:

    flash on STRIDED views      4055.8 us
    flash on CONTIGUOUS         2277.4 us      1.78x
    cost of a .contiguous() repack             3477.5 us   <- never pays

    cfg 13   1.07x     cfg 1   0.99x           <- the tax is config-6 specific

So the win exists but cannot be collected by repacking: the copy costs more than the tax.
It can only be collected by a GEMM that OWNS ITS EPILOGUE and scatters each output tile
straight into head-major buffers, paying no extra pass over memory at all.

THE INDEXING, WHICH IS THE WHOLE KERNEL
---------------------------------------
Row m of the token axis is (b, s) with b = m // S, s = m % S.
Column j within one projection is (h, e) with h = j // hd, e = j % hd.
Destination in a [B, H, S, hd] contiguous tensor:  ((b*H + h)*S + s)*hd + e.

The GEMM tile is [BM, BN] over (m, j); every destination offset is computable from the
tile indices, so the scatter is free -- it is just a different address computation on a
store the kernel was already performing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_headmajor(
    X, W, B,            # X fp16 [M, D]; W fp16 [D, 3D] (pre-transposed); B fp16 [3D]
    Q, K, V,            # fp16 [Bsz, H, S, hd], contiguous
    M,
    S: tl.constexpr, H: tl.constexpr, HD: tl.constexpr,
    D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    keep = rm < M

    x = tl.load(X + rm[:, None] * D + rd[None, :], mask=keep[:, None], other=0.0)
    w = tl.load(W + rd[:, None] * (3 * D) + rn[None, :])
    acc = tl.dot(x, w, out_dtype=tl.float32) + tl.load(B + rn)[None, :].to(tl.float32)
    out = acc.to(tl.float16)

    # Which projection each column belongs to, and where it lands head-major.
    which = rn // D                 # 0 = Q, 1 = K, 2 = V
    j = rn % D
    h = j // HD
    e = j % HD
    b = rm // S
    s = rm % S
    dst = ((b[:, None] * H + h[None, :]) * S + s[:, None]) * HD + e[None, :]

    tl.store(Q + dst, out, mask=keep[:, None] & (which == 0)[None, :])
    tl.store(K + dst, out, mask=keep[:, None] & (which == 1)[None, :])
    tl.store(V + dst, out, mask=keep[:, None] & (which == 2)[None, :])


def qkv_headmajor(xn: torch.Tensor, w_t: torch.Tensor, bias: torch.Tensor,
                  batch: int, seq: int, heads: int,
                  block_m: int = 64, block_n: int = 128, num_warps: int = 4):
    """(q, k, v), each [batch, heads, seq, head_dim] and CONTIGUOUS.

    `w_t` is the fused QKV weight already transposed to [D, 3D].
    """
    m, d = xn.shape
    hd = d // heads
    q = torch.empty((batch, heads, seq, hd), device=xn.device, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    _qkv_headmajor[(triton.cdiv(m, block_m), triton.cdiv(3 * d, block_n))](
        xn, w_t, bias, q, k, v, m,
        S=seq, H=heads, HD=hd, D=d, BM=block_m, BN=block_n, num_warps=num_warps,
    )
    return q, k, v


def worth_it(batch: int, seq: int, d_model: int, heads: int) -> bool:
    """Is the strided-read tax large enough here to be worth a hand-written GEMM?

    TILING. BN=128, not 64. The first guess used BN=64 and the kernel LOST to cuBLAS at
    0.88x; autotuning over (BM, BN, num_warps) found BM64/BN128/w4 at 1.163x. The output
    axis is 3*D = 384 wide, so a 64-wide tile fragments it. The mechanism was right and
    the tiling was wrong, which is not something a mechanism argument can tell you.

    The tax comes from flash re-reading a [B, H, S, hd] view whose innermost stride jumps
    by 3*D. It matters when the score-matrix work per head is large enough that the read
    pattern dominates the kernel -- measured 1.78x at B=10000, 1.07x at B=64/S=1024, and
    nothing at B=64/S=128. The discriminator is total token count, not any config id.

    Threshold set at the measured crossover with margin: cfg 6 (1.28M tokens) wins,
    cfg 13 (65k) does not.
    """
    if d_model % heads or (d_model // heads) < 16:
        return False
    return batch * seq >= 500_000
