"""Causal attention with the K/V axis in a LOOP -- the second tile shape, for the grids
`attn_single_tile` is legal on but marginal.

RELATIONSHIP TO `attn_single_tile`
----------------------------------
`attn_single_tile` deletes flash's K/V loop because at S<=128 the whole score matrix fits
in the register file. Its own docstring records the price: **a loop-free kernel cannot
software-pipeline**, so every program is one dependent chain (load Q/K/V -> dot -> softmax
-> dot -> store) and the only latency hiding it has is other resident blocks. Its
`MIN_RESIDENT_BLOCKS = 4` predicate is that trade-off written down, and the measured table
in that file shows head_dim 64 sitting *exactly* on the crossover at 4.9 blocks/SM with a
1.19x that is the weakest of the shapes it accepts.

This kernel takes the opposite side of the same trade: it puts the K/V axis back in a
loop, so K and V tiles stage through shared memory via Triton's pipeliner
(`num_stages`) and the operand term leaves the register file. It gives up the deleted
bookkeeping (a running max, a running sum, an accumulator rescale) to buy pipelining.

WHERE THAT TRADE IS SUPPOSED TO WIN, AND WHY THE PREDICATE IS THE GRID
----------------------------------------------------------------------
Finding 48 measured both forms, each swept over its full legal grid, on four shapes:

    shape                     looped      single_tile(autotuned)   sdpa+repack
    H=2 hd=64  S=128 (cfg 10) 20.623 us   24.757                   27.469
    H=1 hd=128 S=128 (cfg  9) 20.848      (declines)               23.424
    H=4 hd=256 S=128 (cfg  8) 133.814     (declines)               134.606
    H=4 hd=32  S=1024 (cfg 13) 268.138    (declines)               309.108

and read the pattern off the grid rather than off head_dim:

    cfg  9   B*H =  64  ->  64 CTAs at BM=128, on 66 SMs   one wave, nothing to hide behind
    cfg 10   B*H = 128  -> 128 CTAs at BM=128              two waves; latency has somewhere to go
    cfg  8   B*H = 256  -> 1024 CTAs at BM=32              grid ample; hd=256 is register-bound

Pipelining pays when there is more than one wave of blocks, because that is what gives the
scheduler something to overlap the loop's memory latency with. So the predicate here is
`B * heads * cdiv(S, BM)` against the device's measured `multi_processor_count` -- shapes
and measured device properties only, no config id (CLAUDE.md rule 2).

**THE PREDICATE IS A HYPOTHESIS, NOT A RESULT.** It was read off four points. It gates
which shapes are *considered*; it never decides the winner. `bench/kernels/attn_choice.py`
sweeps both forms over their complete legal grids and picks by timing, because a mechanism
argument cannot pick a tile size and this project has measured the cost of letting it try
(0.88x on a guessed tile, 1.163x on a swept one).

EXACTNESS
---------
Online softmax with a multi-tile reduction is what `attn_single_tile` removed as
*bookkeeping*, not as *approximation*: with one K tile the two are the same arithmetic
with the rescalings removed. Restoring the loop restores the rescalings, so this kernel is
numerically flash-equivalent -- the accumulator is fp32 throughout and only `P` is rounded
to fp16 for the tensor cores, exactly as `attn_single_tile` and FlashAttention both do.

Causality is exact and is *cheaper* here than in the loop-free form: `kv_end` truncates the
loop at the query block's own last row, so whole K/V tiles are never loaded, rather than
being loaded and masked away by `tl.where`.

The output layout is head-major `[B, S, heads*head_dim]` -- byte-identical to
`sdpa(...).transpose(1, 2).reshape(B, S, d_model)` and to `single_tile_attention`'s, so
this is a drop-in for the same call site and the same repack elision (finding 31).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .attn_single_tile import MMA_WIDTH, next_pow2, padded_head_dim


@triton.jit
def _attn_looped(
    QKV,                 # [B, S, 3*DM] fp16, contiguous in the last axis
    OUT,                 # [B, S, DM]   fp16
    stride_qkv_b, stride_qkv_s,
    stride_o_b, stride_o_s,
    scale,
    S: tl.constexpr,     # sequence length
    DH: tl.constexpr,    # true head_dim
    DP: tl.constexpr,    # head_dim padded up to the MMA width, power of two
    DM: tl.constexpr,    # heads * head_dim, the stride from Q to K to V
    BM: tl.constexpr,    # query rows per program
    BN: tl.constexpr,    # key rows per loop trip
):
    """Flash-style causal attention over the fused QKV buffer.

    Registers hold Q [BM, DP] fp16, acc [BM, DP] fp32 and the running m/l vectors. The
    K and V tiles are [BN, DP] and are staged through shared memory by Triton's
    pipeliner -- which is the entire point: the operand term leaves the register file,
    and the loop gives the pipeliner something to overlap.
    """
    m_block = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    rm = m_block * BM + tl.arange(0, BM)
    rd = tl.arange(0, DP)
    keep_m = rm < S
    keep_d = rd < DH                      # False lanes load 0.0 -> exact zero padding

    head = QKV + b * stride_qkv_b + h * DH
    q = tl.load(head + rm[:, None] * stride_qkv_s + rd[None, :],
                mask=keep_m[:, None] & keep_d[None, :], other=0.0)

    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, DP], dtype=tl.float32)

    # Causal, exact, and FREE rather than masked: this query block never needs a key
    # beyond its own last row, so those K/V tiles leave the loop entirely instead of
    # being loaded and discarded by a `tl.where`.
    kv_end = tl.minimum((m_block + 1) * BM, S)

    for start_n in range(0, kv_end, BN):
        rn = start_n + tl.arange(0, BN)
        keep_n = rn < S
        k = tl.load(head + DM + rn[:, None] * stride_qkv_s + rd[None, :],
                    mask=keep_n[:, None] & keep_d[None, :], other=0.0)
        v = tl.load(head + 2 * DM + rn[:, None] * stride_qkv_s + rd[None, :],
                    mask=keep_n[:, None] & keep_d[None, :], other=0.0)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        valid = (rn[None, :] <= rm[:, None]) & keep_n[None, :]
        s = tl.where(valid, s, float("-inf"))

        # Online softmax. Rows with rm >= S (only reachable when S is not a multiple of
        # BM) keep every column valid, so no row is ever entirely -inf and no NaN is
        # produced in the lanes the masked store is about to discard.
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(OUT + b * stride_o_b + rm[:, None] * stride_o_s + h * DH + rd[None, :],
             acc.to(OUT.dtype.element_ty),
             mask=keep_m[:, None] & keep_d[None, :])


# ------------------------------------------------------------------ launcher

def looped_attention(qkv: torch.Tensor, heads: int, head_dim: int, scale: float,
                     block_m: int, block_n: int, num_warps: int,
                     num_stages: int = 2, _return_handle: bool = False):
    """Causal SDPA over the fused `[B, S, 3*d_model]` buffer, K/V axis looped.

    Returns `[B, S, d_model]` fp16 head-major -- the same bytes as
    `single_tile_attention` and as `sdpa(...).transpose(1, 2).reshape(...)`.
    """
    b, s, three_dm = qkv.shape
    dm = heads * head_dim
    assert three_dm == 3 * dm, f"expected [B, S, 3*{dm}], got {tuple(qkv.shape)}"
    out = torch.empty((b, s, dm), device=qkv.device, dtype=qkv.dtype)
    handle = _attn_looped[(triton.cdiv(s, block_m), heads, b)](
        qkv, out,
        qkv.stride(0), qkv.stride(1),
        out.stride(0), out.stride(1),
        scale,
        S=s, DH=head_dim, DP=padded_head_dim(head_dim), DM=dm,
        BM=block_m, BN=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return (out, handle) if _return_handle else out


# ------------------------------------------------------------------ working set

def register_bytes(block_m: int, block_n: int, head_dim: int) -> int:
    """Per-program register working set.

    Unlike the loop-free form, the K/V operands are NOT counted: Triton stages them
    through shared memory for the `tl.dot`, which is the whole reason this shape exists.
    What stays in registers is the score tile, the fp32 accumulator and Q.
    """
    dp = padded_head_dim(head_dim)
    scores = block_m * block_n * 4          # fp32
    acc = block_m * dp * 4                  # fp32
    q = block_m * dp * 2                    # fp16
    return scores + acc + q


def smem_bytes(block_n: int, head_dim: int, num_stages: int) -> int:
    """Shared memory the pipeliner needs for the staged K and V tiles."""
    return num_stages * 2 * block_n * padded_head_dim(head_dim) * 2


def grid_ctas(batch: int, heads: int, seq_len: int, block_m: int) -> int:
    return batch * heads * triton.cdiv(seq_len, block_m)


# ------------------------------------------------------------------ predicate
#
# MIN_WAVES is the hypothesis, and it is stated as a hypothesis. Finding 48 measured the
# looped form winning at 128 CTAs on 66 SMs (~1.9 waves) and losing its margin at 64 CTAs
# (~1.0 wave). A loop is only worth its bookkeeping if the scheduler has other blocks to
# overlap the memory latency with, and one wave means it does not. Expressed against the
# device's measured `multi_processor_count` so another card evaluates it without being
# retuned.
#
# It gates CONSIDERATION only. `attn_choice.autotune` sweeps and times.

MIN_WAVES = 1.5
MAX_SMEM_FRACTION = 1.0     # the opt-in ceiling is a measured device property


def fits(seq_len: int, head_dim: int, block_m: int, block_n: int, num_warps: int,
         num_stages: int, regs_per_sm: int, smem_optin: int,
         warp_size: int = 32) -> bool:
    """Legality: the MMA's minimum dimensions, the register file, and opt-in smem."""
    if block_m < MMA_WIDTH or block_n < MMA_WIDTH:
        return False
    if seq_len < MMA_WIDTH or head_dim < 1:
        return False
    if block_m > next_pow2(seq_len) or block_n > next_pow2(seq_len):
        return False
    from .attn_single_tile import register_budget
    if register_bytes(block_m, block_n, head_dim) > register_budget(
            num_warps, regs_per_sm, warp_size):
        return False
    return smem_bytes(block_n, head_dim, num_stages) <= smem_optin * MAX_SMEM_FRACTION


def pays(batch: int, heads: int, seq_len: int, block_m: int,
         multi_processor_count: int) -> bool:
    """Is there more than one wave of blocks for the pipeliner to hide behind?"""
    return (grid_ctas(batch, heads, seq_len, block_m) /
            max(1, multi_processor_count)) >= MIN_WAVES


# The grid this form is swept over, and the two MECHANISM constraints that shape it.
# Both forms are then swept over their full remaining grid, or neither is: finding 47
# measured a 4.5% best-of-N-against-best-of-1 handicap and finding 48 committed it --
# 180 arms for the challenger against 4 at one warp count for the incumbent.
#
#   block_n < block_m   THE LOOP MUST HAVE MORE THAN ONE TRIP. `kv_end` truncates the
#                       loop at the query block's last row, so a program covering the
#                       first block runs `cdiv(block_m, block_n)` trips; at
#                       block_n >= block_m that is ONE, and a one-trip loop is the
#                       loop-free kernel plus an online-softmax rescale it does not
#                       need -- strictly worse than `attn_single_tile` by construction.
#                       This is finding 47's defect stated as a constraint: F-03 was
#                       declined because its winning arms ran their grid-stride loop
#                       exactly once, and nobody checked before pricing it.
#
#   num_stages >= 2     PIPELINING IS THE ENTIRE MECHANISM. `num_stages=1` tells Triton
#                       not to pipeline, which removes the only thing this form buys.
#
# PRE-REGISTERED COST OF THE PRUNE, so it is a constraint and not a fit: it excludes
# finding 48's winners on config 8 (BM=32 BN=32, measured 1.006x -- nothing) and config
# 13 (BM=64 BN=64, 1.153x on a config past the 3.0 cap, worth zero). It keeps its winners
# on configs 9 (BM=128 BN=16 st=3) and 10 (BM=128 BN=16 st=4), which are the two rows
# that can still score. If a swept winner is ever found at block_n >= block_m, this
# constraint is wrong and should be deleted rather than argued with.
SWEEP_TILES: tuple[tuple[int, int, int, int], ...] = tuple(
    (bm, bn, w, st)
    for bm in (16, 32, 64, 128)
    for bn in (16, 32, 64, 128)
    for w in (2, 4, 8)
    for st in (2, 3, 4)
    if bn < bm
)


def viable_tiles(batch: int, heads: int, seq_len: int, head_dim: int, props
                 ) -> list[tuple[int, int, int, int]]:
    """Every (block_m, block_n, num_warps, num_stages) that both fits and pays here."""
    out = []
    for bm, bn, w, st in SWEEP_TILES:
        if not fits(seq_len, head_dim, bm, bn, w, st,
                    props.regs_per_multiprocessor,
                    props.shared_memory_per_block_optin, props.warp_size):
            continue
        if not pays(batch, heads, seq_len, bm, props.multi_processor_count):
            continue
        out.append((bm, bn, w, st))
    return out


def applies(batch: int, heads: int, seq_len: int, head_dim: int, props
            ) -> tuple[bool, str]:
    """(consider_it, reason). Measured device properties only -- no config ids."""
    tiles = viable_tiles(batch, heads, seq_len, head_dim, props)
    if not tiles:
        g = grid_ctas(batch, heads, seq_len, min(128, next_pow2(seq_len)))
        return False, (f"looped declined: no tile both fits and reaches {MIN_WAVES} "
                       f"waves ({g} CTAs on {props.multi_processor_count} SMs)")
    return True, f"looped: {len(tiles)} legal tiles on this device"
