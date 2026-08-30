"""Fused FFN megakernel: out = (gelu(x @ W1^T + b1)) @ W2^T + b2, in ONE Triton kernel.

This is the place a hand-written kernel can beat separate cuBLAS calls: the baseline runs
the FFN as two GEMMs and writes the [M, ffn_dim] hidden activation to HBM between them
(~16 MB of write+read per layer at ffn_dim=2048, M=1024). Here each program owns a tile of
BM rows, streams the hidden dimension in blocks, and keeps the hidden slice on-chip -- the
intermediate never touches HBM. x stays resident; only the [M, d_model] output is written.

Precision defaults to tf32x3: the probe showed single-pass tf32 (9e-2 raw error) is 2.5x
worse than cuBLAS tf32 and fails the gate through 6 layers, while tf32x3 (3e-5) is better
than fp32 and passes comfortably. The bet is that the saved HBM traffic + single launch
offsets the 3-pass tensor-core cost.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    # N2=d_model=512 is the full width of the GEMM2 output tile, so the w2 operand staged
    # per step is N2*BH; BH is kept at 32 and the loop single-buffered (num_stages=1) to
    # stay under GB10's 99 KB shared-memory budget.
    cfgs = []
    for bm in (32, 64, 128):
        for bk in (128, 256, 512):
            for w in (4, 8):
                cfgs.append(triton.Config(
                    {"BM": bm, "BH": 32, "BK": bk}, num_warps=w, num_stages=1))
    return cfgs


@triton.autotune(configs=_configs(), key=["M", "H", "K1", "N2"])
@triton.jit
def _fused_ffn(
    X, W1, B1, W2, B2, Out,
    M, H, K1: tl.constexpr, N2: tl.constexpr,
    sxm, sxk, sw1h, sw1k, sw2n, sw2h, som, son,
    PREC: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    om = pid * BM + tl.arange(0, BM)
    mmask = om < M
    on2 = tl.arange(0, N2)

    out = tl.zeros([BM, N2], tl.float32)          # [BM, d_model] accumulator, stays resident

    for h0 in range(0, H, BH):
        oh = h0 + tl.arange(0, BH)
        # GEMM1: g1[BM,BH] = x[BM,K1] @ W1[oh,:K1]^T, contracted over K1 in BK blocks so no
        # tile is ever the full 512 wide.
        g1 = tl.zeros([BM, BH], tl.float32)
        for k0 in range(0, K1, BK):
            okb = k0 + tl.arange(0, BK)
            xk = tl.load(X + om[:, None] * sxm + okb[None, :] * sxk,
                         mask=mmask[:, None], other=0.0)              # [BM, BK]
            w1 = tl.load(W1 + oh[:, None] * sw1h + okb[None, :] * sw1k)  # [BH, BK]
            g1 += tl.dot(xk, tl.trans(w1), input_precision=PREC)
        g1 += tl.load(B1 + oh)[None, :]
        h = g1 * 0.5 * (1.0 + tl.erf(g1 * 0.7071067811865476))       # exact GELU, [BM, BH]
        # GEMM2 partial: out[BM,N2] += h[BM,BH] @ W2[:N2, oh]^T ; the hidden slice never
        # leaves the kernel.
        w2 = tl.load(W2 + on2[:, None] * sw2n + oh[None, :] * sw2h)   # [N2, BH]
        out += tl.dot(h.to(w2.dtype), tl.trans(w2), input_precision=PREC).to(tl.float32)

    out += tl.load(B2 + on2)[None, :]
    tl.store(Out + om[:, None] * som + on2[None, :] * son, out.to(Out.dtype.element_ty),
             mask=mmask[:, None])


def fused_ffn(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor,
              w2: torch.Tensor, b2: torch.Tensor, prec: str = "tf32x3") -> torch.Tensor:
    """x:[...,K1] -> [...,N2]. w1:[H,K1], b1:[H], w2:[N2,H], b2:[N2] (nn.Linear layout)."""
    *lead, K1 = x.shape
    M = 1
    for d in lead:
        M *= d
    H, _ = w1.shape
    N2 = w2.shape[0]
    x2 = x.reshape(M, K1).contiguous()
    w1 = w1.contiguous()
    w2 = w2.contiguous()
    out = torch.empty((M, N2), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]),)

    _fused_ffn[grid](
        x2, w1, b1, w2, b2, out,
        M, H, K1, N2,
        x2.stride(0), x2.stride(1), w1.stride(0), w1.stride(1),
        w2.stride(0), w2.stride(1), out.stride(0), out.stride(1),
        PREC=prec,
    )
    return out.reshape(*lead, N2)
