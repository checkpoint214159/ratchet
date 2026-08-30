"""The attention out-projection, its fp32 cast and the residual add, in one Triton kernel.

WHAT THIS REPLACES
------------------
Every candidate from v2 onward ends the attention half of a layer with

    x = x + F.linear(ctx.transpose(1, 2).reshape(B, S, D), out_w, out_b).float()

which is a `[M, D] x [D, D]` fp16 GEMM writing an fp16 `[M, D]` temporary, followed by a
pointwise kernel that reads that temporary, reads the fp32 residual, and writes the fp32
result. This kernel does the GEMM with the residual add and the fp32 widening in its
EPILOGUE, so the fp16 temporary never exists.

THE PREMISE THAT DIED, AND IS RECORDED HERE RATHER THAN QUIETLY DROPPED
----------------------------------------------------------------------
Proposal D-02 was motivated by a transposed read: `ctx` is `[B, H, S, hd]` and the
projection wants `[B*S, D]`, so the `.transpose(1, 2).reshape(...)` looked like a
head-major gather -- the mirror image of `qkv_headmajor`'s epilogue scatter, which is
worth 1.163x on its own segment. **It is not.** Measured on this card
(`bench/probe_outproj.py --layout-only`), `F.scaled_dot_product_attention` does not return a
`[B, H, S, hd]`-contiguous tensor; it returns one whose strides are

    shape (B, H, S, hd)   stride (H*S*hd, hd, H*hd, 1)

i.e. a `[B, S, H, hd]`-contiguous buffer viewed as head-major. That is true at every
shape in the matrix, on every head_dim from 8 to 256. So `ctx.transpose(1, 2)` is ALREADY
contiguous, `.reshape(B, S, D)` is a free view, and there is no gather and no copy to
absorb. The `Memcpy DtoD` in the profile is not this.

The gather is implemented anyway, as a strided load driven by `ctx`'s runtime strides
rather than an assumed layout. It costs nothing when the strides are the friendly ones
(the address arithmetic is the same either way) and it means the kernel stays correct if
a different SDPA backend, a different head_dim, or a future dispatch produces a genuinely
head-major `ctx`. Correctness that does not depend on a layout we merely observed once
is the point (L24).

SO WHAT IS LEFT IS THE EPILOGUE, AND IT IS SMALLER
--------------------------------------------------
Per token, bytes moved by the two-kernel path against this one:

    two kernels   read ctx 2D + write o 2D | read o 2D + read res 4D + write y 4D = 14D
    fused         read ctx 2D              | read res 4D + write y 4D             = 10D

29% of the segment's traffic and one launch of the two. The claim is bounded by that,
not by the 18.9% GEMM bucket -- most of that bucket is arithmetic this kernel still has
to do, against cuBLAS's own tuned `ampere_fp16_s16816gemm`.

TWO CORRECTNESS POINTS
----------------------
1. **The residual stays fp32.** Finding 08: an fp16 residual fails 12 of 14 configs. The
   accumulator is fp32, the residual is loaded fp32, and the sum is stored fp32.
2. The fused path is strictly MORE accurate than what it replaces, because the two-kernel
   path rounds the projection output to fp16 before `.float()` widens it again. The
   rounding step is deleted, not added -- the same direction as the FFN megakernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _outproj_resid(
    CTX,                        # fp16 [B, H, S, HD], strides passed explicitly
    W, BIAS,                    # fp16 [D, D] pre-transposed (contract over axis 0), fp16 [D]
    RES,                        # fp32 [M, D]
    Y,                          # fp32 [M, D]
    M,
    sc_b, sc_h, sc_s, sc_e,     # element strides of CTX
    sc_row,                     # row stride of the [M, D] view, valid only when CONTIG
    S: tl.constexpr, H: tl.constexpr, HD: tl.constexpr, D: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, CONTIG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    keep = rm < M

    # Row m of the token axis is (b, s); column k of the contraction is (h, e). The
    # destination-side arithmetic of qkv_headmajor, run backwards as a gather.
    b = rm // S
    s = rm % S
    row_base = b * sc_b + s * sc_s

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in tl.range(0, D, BK):
        rk = k0 + tl.arange(0, BK)
        if CONTIG:
            # The two address expressions are numerically identical when the head-major
            # view is contiguous, but only this one lets Triton prove a contiguous run of
            # BK and emit a vector load. With the general form it can prove a run of only
            # HD, which at head_dim=8 is 16 bytes -- half a sector -- and measured 1.185x
            # against 1.486x on the same work. See the candidate docstring.
            x = tl.load(CTX + rm[:, None] * sc_row + rk[None, :],
                        mask=keep[:, None], other=0.0)
        else:
            h = rk // HD
            e = rk % HD
            x = tl.load(CTX + row_base[:, None] + (h * sc_h + e * sc_e)[None, :],
                        mask=keep[:, None], other=0.0)
        w = tl.load(W + rk[:, None] * D + rn[None, :])
        acc += tl.dot(x, w, out_dtype=tl.float32)

    acc += tl.load(BIAS + rn)[None, :].to(tl.float32)

    # EPILOGUE: fp32 residual read and add, in registers. No fp16 temporary anywhere.
    off = rm[:, None] * D + rn[None, :]
    res = tl.load(RES + off, mask=keep[:, None], other=0.0)
    tl.store(Y + off, res + acc, mask=keep[:, None])


def smem_bytes(block_m: int, block_n: int, block_k: int, elem_size: int,
               num_stages: int = 3) -> int:
    """One program's staged tiles: the activation tile and the weight tile, held
    `num_stages` deep by Triton's software pipeline."""
    return num_stages * (block_m * block_k + block_k * block_n) * elem_size


def fits(d_model: int, heads: int, elem_size: int,
         block_m: int, block_n: int, block_k: int, smem_optin: int,
         num_stages: int = 3) -> bool:
    """Dispatch predicate: tensor shapes and one MEASURED device property, never a config
    id and never an announced constant (CLAUDE.md rule 2).

    `tl.dot` on sm_89 is m16n8k16, so every tiled dimension must be >= 16 and a power of
    two; `tl.arange` needs powers of two for the head decomposition. head_dim itself may
    be smaller than 16 (config 7 and 11 are 8) -- it is not a dot dimension here, only an
    address component.
    """
    if heads <= 0 or d_model % heads:
        return False
    for n in (d_model, block_m, block_n, block_k, heads, d_model // heads):
        if n <= 0 or (n & (n - 1)):
            return False
    if block_m < 16 or block_n < 16 or block_k < 16:
        return False
    if block_n > d_model or block_k > d_model:
        return False
    return smem_bytes(block_m, block_n, block_k, elem_size, num_stages) <= smem_optin


def bytes_saved_per_token(d_model: int) -> int:
    """Traffic the fusion removes: the fp16 temporary's write and its read back."""
    return 2 * d_model * 2


def two_kernel_bytes_per_token(d_model: int) -> int:
    """read ctx (fp16) + write o (fp16) + read o (fp16) + read res (fp32) + write y (fp32)."""
    return 2 * d_model + 2 * d_model + 2 * d_model + 4 * d_model + 4 * d_model


# --------------------------------------------------------------------------------------
# TILING. Two tiles, and a predicate over the MEASURED SM count that chooses between them.
#
# The v20 lesson is that a guessed tile loses (0.88x) where a tuned one wins (1.163x), so
# these come from a sweep of 108 (BM, BN, BK, warps) points at seven shapes plus a rule
# comparison over eleven -- `bench/probe_outproj.py`. What the sweep found:
#
#   tokens   WIDE(64, 128, 64, w8)   SMALL(32, 32, 128, w8)
#      128           0.992x                 1.451x
#      512           1.147x                 1.491x
#    2,048           1.048x                 1.379x
#    8,192           1.382x                 1.043x
#   65,536           1.406x                 0.939x
# 1,280,000          1.399x                 0.872x
#
# (`tune2.py`; speedups against the torch.compile'd two-kernel path on the same data.)
#

#
# The sign flips between 2,048 and 8,192 tokens, and the reason is not the token count:
# WIDE emits ceil(M/64) * ceil(D/128) programs, which is 32 at M=2,048 and 128 at
# M=8,192, against this card's 66 SMs. Below saturation the wide tile leaves SMs idle;
# above it, the small tile's four-fold-larger program count costs more in launch and
# scheduling than it wins. So the predicate is "does the wide tile fill the machine",
# read from `props.multi_processor_count` -- a measured device property, not a token
# threshold fitted to this matrix (CLAUDE.md rule 2).
# --------------------------------------------------------------------------------------
WIDE_TILE = (64, 128, 64, 8)        # BM, BN, BK, num_warps
SMALL_TILE = (32, 32, 128, 8)


def _clamp(tile: tuple[int, int, int, int], d_model: int) -> tuple[int, int, int, int]:
    bm, bn, bk, nw = tile
    return bm, min(bn, d_model), min(bk, d_model), nw


def programs(tokens: int, d_model: int, block_m: int, block_n: int) -> int:
    """CTAs the launch produces. This is what has to cover the SMs."""
    return -(-tokens // block_m) * -(-d_model // block_n)


def tiling_for(tokens: int, d_model: int, sm_count: int) -> tuple[int, int, int, int]:
    """Pick a tile from the shape and the measured SM count. See the block above."""
    wide = _clamp(WIDE_TILE, d_model)
    if programs(tokens, d_model, wide[0], wide[1]) >= sm_count:
        return wide
    return _clamp(SMALL_TILE, d_model)


def outproj_resid(ctx: torch.Tensor, res: torch.Tensor,
                  w_t: torch.Tensor, bias: torch.Tensor,
                  block_m: int = 64, block_n: int = 128, block_k: int = 64,
                  num_warps: int = 4, num_stages: int = 3) -> torch.Tensor:
    """`res + (ctx.transpose(1,2).reshape(M, D) @ w_t + bias).float()`, in one launch.

    `ctx` is [B, H, S, hd] (any strides); `res` is fp32 [M, D] with M = B*S; `w_t` is the
    out-projection weight already transposed to [D, D] because the kernel contracts over
    the leading axis.
    """
    bsz, heads, seq, hd = ctx.shape
    d = heads * hd
    m = bsz * seq
    y = torch.empty((m, d), device=ctx.device, dtype=torch.float32)
    sc_b, sc_h, sc_s, sc_e = ctx.stride()

    # If the token-major view is already contiguous -- which is what every SDPA backend
    # measured on this card returns -- take the vectorizable address form. Checked, never
    # assumed: an SDPA backend that returned a genuinely head-major `ctx` would fall to
    # the general gather and still be correct.
    tokenmajor = ctx.transpose(1, 2)
    contig = bool(tokenmajor.is_contiguous())
    src = tokenmajor.reshape(m, d) if contig else ctx
    sc_row = src.stride(0) if contig else 0

    _outproj_resid[(triton.cdiv(m, block_m), triton.cdiv(d, block_n))](
        src, w_t, bias, res, y, m,
        sc_b, sc_h, sc_s, sc_e, sc_row,
        S=seq, H=heads, HD=hd, D=d,
        BM=block_m, BN=block_n, BK=block_k, CONTIG=contig,
        num_warps=num_warps, num_stages=num_stages,
    )
    return y
