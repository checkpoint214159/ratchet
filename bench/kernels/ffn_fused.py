"""One Triton kernel for the whole FFN block: GEMM -> GELU -> GEMM -> fp32 residual.

WHY THIS IS POSSIBLE HERE AND NOWHERE ELSE
------------------------------------------
Inductor fuses elementwise into a GEMM epilogue. It does NOT fuse GEMM into GEMM, because
that needs the intermediate tile held in registers across two `mma` chains instead of
round-tripping through HBM. That is a structural gap, and this matrix exposes it because
**ffn_dim == d_model on all 14 announced rows** (matrix.py notes the equality, but only as
a reason the FFN is LESS dominant than the reference's 4x expansion).

At d_model == ffn_dim == 128 in fp16, W1 and W2 together are 2*128*128*2 = 64 KB, inside
this device's measured 99 KB opt-in shared memory. On a conventional 4x-expansion
transformer they would be 512 KB and this would be impossible.

Per token the activation traffic drops from four tensor passes (read x, write h, read h,
write y) to two, and four launches (norm, GEMM, GELU+GEMM, add) collapse toward one.

TWO CORRECTNESS POINTS THAT ARE EASY TO GET WRONG
-------------------------------------------------
1. **The reference GELU is `approximate="none"`, i.e. the exact erf form**, not tanh. The
   tanh approximation differs by up to ~1e-3 relative, which is half our entire 2e-3
   budget spent on an approximation nobody asked for. We use erf.
2. **The residual add is fp32.** Finding 08 proved the fp32 residual load-bearing: an
   fp16 residual failed 12 of 14 configs. The kernel accumulates in fp32 and adds to the
   fp32 residual in registers, so it introduces no new rounding on that path.

`h` is rounded to fp16 before the second dot, matching what the fp16 candidate path
already does, so the second GEMM still reaches tensor cores.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ffn_block(
    XN,          # normalized input, fp16   [M, D]
    RES,         # residual stream, fp32    [M, D]
    W1, B1,      # [D, F] , [F]             fp16
    W2, B2,      # [F, D] , [D]             fp16
    Y,           # output, fp32             [M, D]
    M,
    D: tl.constexpr, F: tl.constexpr, BM: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    rf = tl.arange(0, F)
    keep = rm < M

    xn = tl.load(XN + rm[:, None] * D + rd[None, :], mask=keep[:, None], other=0.0)
    w1 = tl.load(W1 + rd[:, None] * F + rf[None, :])
    w2 = tl.load(W2 + rf[:, None] * D + rd[None, :])
    b1 = tl.load(B1 + rf)
    b2 = tl.load(B2 + rd)

    # First GEMM, accumulated in fp32 and KEPT in fp32 registers -- never sees HBM.
    h = tl.dot(xn, w1, out_dtype=tl.float32) + b1[None, :].to(tl.float32)

    # Exact erf GELU, matching the reference's approximate="none".
    h = h * 0.5 * (1.0 + tl.erf(h * 0.70710678118654752440))

    # Second GEMM. h -> fp16 so this reaches the tensor cores, as the fp16 path already
    # does; the fp32 residual below is what finding 08 says must not be rounded.
    y = tl.dot(h.to(w2.dtype), w2, out_dtype=tl.float32) + b2[None, :].to(tl.float32)

    res = tl.load(RES + rm[:, None] * D + rd[None, :], mask=keep[:, None], other=0.0)
    tl.store(Y + rm[:, None] * D + rd[None, :], res + y, mask=keep[:, None])


def smem_bytes(d_model: int, ffn_dim: int, elem_size: int, block_m: int) -> int:
    """Shared memory one program needs: both weight matrices plus one activation tile."""
    return (d_model * ffn_dim + ffn_dim * d_model) * elem_size + block_m * d_model * 4


def fits(d_model: int, ffn_dim: int, elem_size: int, block_m: int, smem_optin: int) -> bool:
    """Dispatch predicate. A pure function of tensor shapes and a MEASURED device
    property -- never a config id, never a literal (CLAUDE.md rule 2). On a 48 KB-smem
    card this declines and the caller falls back, which is the 'another GPU can evaluate
    it' test v14_dispatch was built to satisfy.

    tl.dot also requires each dimension >= 16 on sm_89, whose MMA is m16n8k16.
    """
    if d_model < 16 or ffn_dim < 16 or block_m < 16:
        return False
    if d_model & (d_model - 1) or ffn_dim & (ffn_dim - 1):
        return False          # tl.arange needs powers of two
    return smem_bytes(d_model, ffn_dim, elem_size, block_m) <= smem_optin


# Derived from the measured crossover in finding 25, not fitted to config ids. Speedup is
# monotone in weight-bytes-per-token, and the sign flips between 1.0 and 8.0:
#     0.051 -> -7.6%    0.5 -> -5.7%    1.0 -> -2.5%    8.0 -> ~+1.3%    128 -> +49%
# Expressed as a FRACTION of activation traffic so it carries to other widths and cards.
AMORTIZE_FRACTION = 0.002


def activation_bytes_per_token(d_model: int) -> int:
    """One token's traffic through the fused block: fp16 normalized input read, fp32
    residual read, fp32 output write."""
    return d_model * 2 + d_model * 4 + d_model * 4


def amortizes(tokens: int, d_model: int, ffn_dim: int, elem_size: int) -> bool:
    """Is there enough work to pay for hoisting the weights into shared memory?

    The kernel's entire advantage is loading both weight matrices ONCE and streaming
    activations past them, so the advantage scales with how many tokens reuse them.
    Below the crossover the program moves more bytes of weights than of data (L37).
    """
    if tokens <= 0:
        return False
    weight_bytes = 2 * d_model * ffn_dim * elem_size
    return weight_bytes / tokens <= AMORTIZE_FRACTION * activation_bytes_per_token(d_model)


def fused_ffn(xn: torch.Tensor, res: torch.Tensor,
              w1: torch.Tensor, b1: torch.Tensor,
              w2: torch.Tensor, b2: torch.Tensor,
              block_m: int = 64, num_warps: int = 4) -> torch.Tensor:
    """res + (gelu(xn @ w1 + b1) @ w2 + b2), computed in one launch.

    `w1`/`w2` are already TRANSPOSED relative to nn.Linear's [out, in] layout, because
    the kernel contracts over the leading axis.
    """
    m, d = xn.shape
    f = w1.shape[1]
    y = torch.empty((m, d), device=xn.device, dtype=torch.float32)
    _ffn_block[(triton.cdiv(m, block_m),)](
        xn, res, w1, b1, w2, b2, y, m,
        D=d, F=f, BM=block_m, num_warps=num_warps,
    )
    return y


# ======================================================================================
# The norm-fused variant (generation 19, proposal D-01).
#
# v17's kernel takes an already-normalized `xn` and an already-added residual `res`, both
# produced by a SEPARATE Inductor kernel that reads the residual and the attention output,
# adds them, normalizes, and writes two full-size tensors to HBM -- which the megakernel
# then reads straight back. A third kernel reads the megakernel's output and normalizes it
# for the next layer.
#
# All three are bandwidth-saturated at ~661 GB/s against the 613.7 GB/s device figure, i.e.
# AT the achievable roofline. They cannot be made faster. They can only be made not to
# exist. At d_model = 128 a token's entire row is one tile, so the kernel already holds
# the whole output row in registers and can do both reductions itself.
#
#     res = x + attn ; xn = LN2(res) ; h = gelu(xn @ W1 + b1)
#     y   = res + h @ W2 + b2 ; store y ; store LN_next(y)
#
# Traffic per token: 28*D bytes -> 12*D bytes. Weight traffic is unchanged, so the
# amortization crossover from finding 25 still governs.
#
# PRECISION. The residual stays fp32 from the add, through the normalization, to the
# store, never round-tripping HBM -- strictly FEWER rounding steps than the Inductor
# kernel it replaces, which materializes `res` in fp32 and `xn` in fp16. Finding 08's
# fp32 residual is preserved verbatim.
# ======================================================================================


@triton.jit
def _ffn_block_normed(
    X, ATTN,            # fp32 [M, D]: residual in, attention output
    N2W, N2B,           # fp16 [D]   : norm2 weight/bias
    W1, B1, W2, B2,     # fp16       : the two FFN matrices and biases
    NNW, NNB,           # fp16 [D]   : the NEXT consumer's norm weight/bias
    Y,                  # fp32 [M, D]: this layer's output (the residual stream)
    YN,                 # fp16 [M, D]: LN_next(Y), the next layer's input
    M, EPS,
    D: tl.constexpr, F: tl.constexpr, BM: tl.constexpr,
    STORE_NEXT: tl.constexpr,
):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    rf = tl.arange(0, F)
    keep = rm < M
    off = rm[:, None] * D + rd[None, :]

    # --- attention residual add, in fp32 and in registers -----------------------------
    res = (tl.load(X + off, mask=keep[:, None], other=0.0)
           + tl.load(ATTN + off, mask=keep[:, None], other=0.0))

    # --- norm2, reduced in fp32 over the row we already hold --------------------------
    mu = tl.sum(res, axis=1) / D
    d0 = res - mu[:, None]
    var = tl.sum(d0 * d0, axis=1) / D
    xn = d0 * tl.rsqrt(var[:, None] + EPS)
    xn = xn * tl.load(N2W + rd)[None, :].to(tl.float32) + tl.load(N2B + rd)[None, :].to(tl.float32)

    # --- the two GEMMs, intermediate never leaving registers --------------------------
    w1 = tl.load(W1 + rd[:, None] * F + rf[None, :])
    w2 = tl.load(W2 + rf[:, None] * D + rd[None, :])
    h = tl.dot(xn.to(w1.dtype), w1, out_dtype=tl.float32) + tl.load(B1 + rf)[None, :].to(tl.float32)
    h = h * 0.5 * (1.0 + tl.erf(h * 0.70710678118654752440))
    y = (res + tl.dot(h.to(w2.dtype), w2, out_dtype=tl.float32)
         + tl.load(B2 + rd)[None, :].to(tl.float32))
    tl.store(Y + off, y, mask=keep[:, None])

    # --- the NEXT consumer's LayerNorm, from the same registers ------------------------
    # Skipped on the LAST layer: its consumer is `final_norm`, whose result is the model's
    # fp32 output. Emitting that in fp16 here would round the answer on the way out, which
    # is the one place in the stack where a downcast is not already being paid.
    if STORE_NEXT:
        mu2 = tl.sum(y, axis=1) / D
        d2 = y - mu2[:, None]
        var2 = tl.sum(d2 * d2, axis=1) / D
        yn = d2 * tl.rsqrt(var2[:, None] + EPS)
        yn = (yn * tl.load(NNW + rd)[None, :].to(tl.float32)
              + tl.load(NNB + rd)[None, :].to(tl.float32))
        tl.store(YN + off, yn.to(YN.dtype.element_ty), mask=keep[:, None])


def fused_ffn_normed(x, attn, n2w, n2b, w1, b1, w2, b2, nnw, nnb, eps,
                     block_m: int = 64, num_warps: int = 8, store_next: bool = True):
    """(y, yn) = the whole post-attention block plus the next layer's normalization.

    `w1`/`w2` are pre-transposed relative to nn.Linear's [out, in], as in `fused_ffn`.
    """
    m, d = x.shape
    f = w1.shape[1]
    y = torch.empty((m, d), device=x.device, dtype=torch.float32)
    yn = torch.empty((m, d), device=x.device, dtype=torch.float16)
    _ffn_block_normed[(triton.cdiv(m, block_m),)](
        x, attn, n2w, n2b, w1, b1, w2, b2, nnw, nnb, y, yn, m, eps,
        D=d, F=f, BM=block_m, num_warps=num_warps, STORE_NEXT=store_next,
    )
    return y, (yn if store_next else None)
