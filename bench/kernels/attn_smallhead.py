"""Causal flash attention for head dimensions BELOW the tensor core's K width.

WHAT THIS IS FOR, AND WHAT IT IS *NOT* FOR
------------------------------------------
Configs 7 and 11 run head_dim = 8. `docs/findings/23` closed the reason this project
originally cared about that region: **no vendor backend refuses head_dim=8.** flash,
mem-efficient, cuDNN and math all accept it. The one refusal in the whole matrix is cuDNN
at head_dim=256 (config 8), the opposite end of the range.

The region is still the most interesting one, for a better-founded reason. sm_89's only
tensor-core instruction is `mma.sync.m16n8k16`, so every 16-bit `tl.dot` has a **K >= 16**
floor -- Triton's own NVIDIA backend states it, and this module *queries* it rather than
writing 16 down (`min_dot_k` below). PyTorch's bundled FlashAttention-2 has no head_dim=8
kernel either; `HEADDIM_SWITCH` in

    torch/include/ATen/native/transformers/cuda/flash_attn/static_switch.h

rounds anything <= 32 up to kHeadDim = 32. So the vendor kernel is not refused at
head_dim 8 -- it is *wrong-shaped*: it contracts over 32 lanes where 8 carry data.

Padding D to the instruction's 16 INSIDE the kernel costs nothing (the padded lanes are
loaded as literal zeros and never touch HBM). Padding it in HBM is numerically exact but
was measured 1.2-2.7x SLOWER, and that variant is closed -- see finding 23.

WHAT THIS KERNEL SAVES, AND ONE THING I EXPECTED IT TO SAVE AND IT DOES NOT
---------------------------------------------------------------------------
1. **Half the score-matrix MMA work.** 8 -> 16 instead of 8 -> 32.
2. **The causal triangle**, which the loop bound skips outright. Exact, not approximate:
   masked entries carry exactly zero softmax weight (matrix.py, finding 02).
3. **NOT the output repack, which turned out to be free.** The lineage writes
   `ctx.transpose(1, 2).reshape(B, S, D)` after every SDPA call, and I expected that
   `reshape` on a transposed view to materialise a copy worth a full read and write of
   the context tensor. Measured (indicative, isolated `do_bench`, median of 5):

       cfg 7   SDPA alone 24.32 us   SDPA + repack 22.97 us
       cfg 11  SDPA alone 58.54 us   SDPA + repack 57.40 us

   The repack costs nothing, because PyTorch's flash backend works internally in
   `[B, S, H, hd]` and hands back a *transposed view* of that buffer -- so
   `.transpose(1, 2)` restores contiguity and `reshape` is free. The token-major epilogue
   below is kept because it is genuinely free for us too (a different address computation
   on a store we were already doing), but **it must not be credited with any win.**
   Writing that down is the point: the mechanism I was most confident about was the one
   that was worth zero, which is what a measurement is for.

EXACTNESS
---------
Same online-softmax recurrence as FlashAttention, fp16 operands, fp32 accumulators and
fp32 softmax statistics. Scores are scaled in fp32 after the dot rather than by pre-
scaling q in fp16. The padded lanes d in [head_dim, 16) hold literal 0.0 in both K and V,
so they add exactly 0.0 to every score and exactly 0.0 to every context element; the
output store drops them. Nothing here is an approximation of anything.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl


# ======================================================================================
# The hardware fact this kernel is built around, QUERIED rather than asserted.
# ======================================================================================

def min_dot_k(bitwidth: int = 16) -> int:
    """The K (contraction) floor of `tl.dot` on this backend, for `bitwidth` operands.

    Read out of Triton's own NVIDIA backend so the number tracks the compiler rather than
    a comment. On sm_89 this is 16 for 16-bit operands (`mma.sync.m16n8k16`) and 32 for
    8-bit. Falls back to 16 -- the value for every NVIDIA tensor core that exists at
    fp16 -- if the private helper moves, which is the safe direction: a too-large floor
    only declines to dispatch.
    """
    try:
        from triton.backends.compiler import GPUTarget
        from triton.backends.nvidia.compiler import min_dot_size

        class _T:                       # duck-types what check_dot_compatibility reads
            class scalar:
                primitive_bitwidth = bitwidth

        cc = torch.cuda.get_device_capability()
        target = GPUTarget("cuda", cc[0] * 10 + cc[1], 32)
        return int(min_dot_size(target)(_T, _T)[2])
    except Exception:
        return 16


# ======================================================================================
# The kernel
# ======================================================================================

@triton.jit
def _attn_fwd_smallhead(
    Q, K, V, Out,
    sqz, sqh, sqm,          # Q strides: batch, head, seq   (last dim assumed unit stride)
    skz, skh, skn,
    svz, svh, svn,
    S,                      # sequence length, runtime
    qk_scale,               # head_dim**-0.5 * log2(e), folded so the softmax uses exp2
    H: tl.constexpr,
    D: tl.constexpr,        # true head_dim  (8 here)
    DP: tl.constexpr,       # head_dim padded to the mma K floor (16)
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """One program computes a [BM, D] slice of the output for one (batch, head).

    Out is TOKEN-MAJOR, [Z, S, H*D] contiguous -- see mechanism 2 in the module docstring.
    """
    pid_m = tl.program_id(0)
    pid_zh = tl.program_id(1)
    z = pid_zh // H
    h = pid_zh % H

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, DP)
    dk = offs_d < D                      # the pad lanes, held at literal zero

    q_ptr = Q + z * sqz + h * sqh
    k_ptr = K + z * skz + h * skh
    v_ptr = V + z * svz + h * svh

    q = tl.load(q_ptr + offs_m[:, None] * sqm + offs_d[None, :],
                mask=(offs_m < S)[:, None] & dk[None, :], other=0.0)

    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, DP], dtype=tl.float32)

    # CAUSAL: keys beyond the last query row of this tile contribute exactly zero softmax
    # weight, so the loop simply never visits them. Exact, and it halves the work.
    hi = tl.minimum((pid_m + 1) * BM, S)

    for start_n in tl.range(0, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        nk = offs_n < S
        # K loaded transposed to [DP, BN] so the dot contracts over DP.
        k = tl.load(k_ptr + offs_n[None, :] * skn + offs_d[:, None],
                    mask=nk[None, :] & dk[:, None], other=0.0)
        qk = tl.dot(q, k, out_dtype=tl.float32) * qk_scale
        qk = tl.where((offs_m[:, None] >= offs_n[None, :]) & nk[None, :],
                      qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)

        v = tl.load(v_ptr + offs_n[:, None] * svn + offs_d[None, :],
                    mask=nk[:, None] & dk[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    acc = acc / l_i[:, None]

    # Token-major epilogue: [z, s, h*D + d] in a [Z, S, H*D] contiguous tensor.
    out = Out + (z * S + offs_m[:, None]) * (H * D) + h * D + offs_d[None, :]
    tl.store(out, acc.to(Out.dtype.element_ty),
             mask=(offs_m < S)[:, None] & dk[None, :])


# ======================================================================================
# Tiling. Swept, not guessed -- the sweep table is below.
# ======================================================================================

# MEASURED, not reasoned. v20 lost at 0.88x with a guessed tile and won at 1.163x after a
# sweep; a mechanism argument cannot pick a tile size.
#
# 18 tiles swept on both target shapes, then the top 8 re-run at median-of-5. Isolated
# `do_bench`, fp16, causal, strided q/k/v views exactly as the lineage produces them.
# INDICATIVE ONLY (L41) -- a probe may propose a tile, it may not conclude a speedup.
#
#     tile (BM,BN,warps,stages)   cfg 7           cfg 11          vs SDPA + repack
#     (64,  64, 4, 2)             16.77 us        42.23 us        1.397x / 1.398x  <- kept
#     (64,  64, 4, 3)             17.24 us        41.72 us        1.359x / 1.415x
#     (64,  64, 4, 1)             16.78 us        41.73 us        1.397x / 1.414x
#     (128, 32, 4, 2)             17.65 us        45.91 us        1.327x / 1.286x
#     (32,  64, 2, 2)             18.24 us        44.86 us        1.284x / 1.316x
#     (16, 128, 2, 2)             30.84 us        80.37 us        0.745x / 0.714x  <- worst
#
# The spread across tiles is 1.9x on the same kernel and the same mechanism. The three
# leaders are within 2% of each other, i.e. tied; (64,64,4,2) is kept because it leads on
# cfg 7 and because num_stages=2 is Triton's own default.
TILING: tuple[tuple[int, int, int, int], ...] = (
    # (BM, BN, num_warps, num_stages)
    (16, 32, 2, 2), (16, 64, 2, 2), (16, 128, 2, 2),
    (32, 32, 2, 2), (32, 64, 2, 2), (32, 128, 2, 2), (32, 64, 4, 2),
    (64, 32, 4, 2), (64, 64, 4, 2), (64, 128, 4, 2), (64, 64, 2, 2),
    (64, 64, 4, 3), (64, 128, 8, 2), (128, 32, 4, 2), (128, 64, 4, 2),
    (128, 128, 8, 2), (128, 64, 8, 2), (128, 128, 4, 2),
)

DEFAULT_TILE: tuple[int, int, int, int] = (64, 64, 4, 2)

# THE HEADROOM THIS KERNEL DOES NOT COLLECT, measured so the next generation does not have
# to rediscover it. q/k/v here are views of one [Z, S, 3D] GEMM output, so the head-dim
# axis is 16 contiguous bytes and then jumps 3*D*2 bytes -- at head_dim=8 every load is a
# 16-byte request inside its own 32-byte sector, so half of all attention DRAM traffic is
# structurally wasted. Same kernel, same tile, on CONTIGUOUS [Z, H, S, hd] inputs:
#
#                     strided (what we get)      contiguous (what we would like)
#     cfg 7           16.77 us  = 1.397x         13.52-14.00 us = 1.67-1.73x
#     cfg 11          42.23 us  = 1.398x         32.10-32.47 us = 1.82-1.84x
#
# ~30% more, and it is NOT collectable by calling `.contiguous()`: repacking three
# 2.1 MB tensors costs ~20 us against the ~10 us it would save. It is collectable only by
# a QKV projection that OWNS ITS EPILOGUE and scatters head-major -- which is exactly
# `bench/kernels/qkv_headmajor.py` (v20), whose own `worth_it()` currently returns False
# for `head_dim < 16`. The recombination v20 x this kernel is a real, measured lead.


def _pow2_le(n: int) -> int:
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def clamp_tile(tile: tuple[int, int, int, int], seq_len: int
               ) -> tuple[int, int, int, int]:
    """A tile wider than the sequence wastes lanes and can fall below the mma floor.

    Clamped to the largest power of two <= seq_len, floored at the mma M/N minimum, which
    is the same 16 the K floor comes from.
    """
    bm, bn, w, st = tile
    cap = max(16, _pow2_le(max(seq_len, 16)))
    return (min(bm, cap), min(bn, cap), w, st)


# ======================================================================================
# Dispatch predicate and entry point
# ======================================================================================

def applicable(head_dim: int, seq_len: int, d_model: int, heads: int,
               bitwidth: int = 16) -> bool:
    """Is this kernel the right shape of tool for these tensors, on THIS device?

    A function of tensor shapes and a MEASURED/queried device property -- the `tl.dot`
    contraction floor reported by the compiler backend for this GPU -- never a config id
    and never an announced shape constant (CLAUDE.md rule 2).

    The claim is only made where the vendor kernel is mis-tiled: `head_dim` strictly below
    the tensor core's contraction width. At or above it the vendor kernel's tiles fit the
    hardware and there is nothing structural to win, so we decline and the caller keeps
    SDPA. On a backend whose floor is 8 (none exists today at fp16) this declines
    everywhere, which is the correct behaviour: the mechanism would not be there.
    """
    if head_dim <= 0 or heads <= 0 or seq_len <= 0:
        return False
    if d_model != head_dim * heads:
        return False
    floor = min_dot_k(bitwidth)
    if head_dim >= floor:
        return False
    if head_dim & (head_dim - 1):
        return False                     # tl.arange and the pad mask need a power of two
    return True


def padded_head_dim(head_dim: int, bitwidth: int = 16) -> int:
    return max(head_dim, min_dot_k(bitwidth))


class Plan(NamedTuple):
    """Everything about a launch that is a function of the SHAPE, resolved once.

    THIS SPLIT IS NOT TIDINESS, IT IS THE DIFFERENCE BETWEEN A 1.6x WIN AND A 2.2x LOSS.
    The candidate's `_core` runs inside `torch.compile`. The first version of this file
    resolved the plan inside the launch wrapper, which meant `min_dot_k()` -- a try/except
    around an import, a locally defined class, and `torch.cuda.get_device_capability()` --
    executed in Dynamo's traced region. Dynamo cannot trace that, dropped the whole frame
    to eager, and **Inductor's fused LayerNorm kernels disappeared**: profiled on config 7,
    `triton_per_fused_*` was replaced by 9 calls to ATen's `vectorized_layer_norm_kernel`
    at 151 us, turning a 1.63x attention win into a 2.18x end-to-end LOSS.

    So: everything that queries the device or the compiler happens at prime time, and the
    traced region sees plain ints and a tensor allocation. `_ffn_block` was safe by
    accident (its wrapper does nothing but arithmetic); this one had to be made safe on
    purpose. Generalises: a hand-written kernel dropped into a compiled region must be
    audited for what its LAUNCH WRAPPER does, not only for what the kernel does.
    """
    dp: int                 # head_dim padded to the mma contraction floor
    bm: int
    bn: int
    num_warps: int
    num_stages: int
    qk_scale: float         # head_dim**-0.5, folded with log2(e) for the exp2 softmax


def plan_for(head_dim: int, seq_len: int,
             tile: tuple[int, int, int, int] = DEFAULT_TILE) -> Plan:
    """Resolve a launch plan. Call this OUTSIDE any traced or captured region."""
    bm, bn, warps, stages = clamp_tile(tile, seq_len)
    return Plan(padded_head_dim(head_dim), bm, bn, warps, stages,
                float(head_dim) ** -0.5 * math.log2(math.e))


def smallhead_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        plan: Plan) -> torch.Tensor:
    """Causal attention over [Z, H, S, hd] fp16 inputs -> [Z, S, H*hd] fp16 TOKEN-MAJOR.

    The returned layout is deliberately not SDPA's: the caller wants `[Z, S, d_model]` for
    the output projection, and producing it here costs one different address computation
    instead of a whole extra pass over memory. (It saves nothing -- see the module
    docstring -- but it costs nothing either.)

    q/k/v may be arbitrary strided views (they are, in this lineage: a `[Z,S,3D]` GEMM
    output split and transposed), so the strides are passed rather than assumed. Only the
    head-dim axis is required to be unit-stride, which the split/view/transpose preserves.

    Nothing in this function may query the device or the compiler: it is traced. See Plan.
    """
    z, h, s, d = q.shape
    out = torch.empty((z, s, h * d), device=q.device, dtype=q.dtype)
    _attn_fwd_smallhead[(triton.cdiv(s, plan.bm), z * h)](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        s, plan.qk_scale,
        H=h, D=d, DP=plan.dp, BM=plan.bm, BN=plan.bn,
        num_warps=plan.num_warps, num_stages=plan.num_stages,
    )
    return out


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           tile: tuple[int, int, int, int] = DEFAULT_TILE) -> torch.Tensor:
    """Plan-and-launch, for tests and probes. NOT for use inside a compiled region."""
    assert k.shape == q.shape and v.shape == q.shape, "q/k/v must share a shape"
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    return smallhead_attention(q, k, v, plan_for(q.shape[-1], q.shape[-2], tile))
