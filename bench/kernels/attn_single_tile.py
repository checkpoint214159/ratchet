"""Causal self-attention in ONE tile: no K/V loop, no online softmax, no transposes.

WHY THIS IS POSSIBLE HERE
-------------------------
FlashAttention exists to avoid materializing an S x S score matrix that does not fit on
chip. Ten of the fourteen announced rows have `seq_len == 128` and one has 32, so per
(batch, head) the whole score matrix is 128x128 -- 64 KB of fp32, which fits in the
register file of one thread block on this card (measured `regs_per_multiprocessor`
= 65536 32-bit registers = 256 KB per SM). In that regime flash's machinery -- a running
max, a running sum, and a rescale of the accumulator on every K block -- is bookkeeping
for a loop that executes exactly once.

This kernel loads Q, K and V for one (batch, head, query-block), computes `S = QK^T`,
masks the causal triangle with one `tl.where`, takes ONE ordinary max-subtract-exp-sum
softmax, multiplies by V and stores. Single pass.

EXACTNESS, NOT A TOLERANCE HOPE
-------------------------------
With one K block, the online softmax and the textbook softmax are the same arithmetic
with the rescalings removed: flash's running rescale exists only to make a multi-tile
reduction equal the single-tile one. Removing tiles removes the rescale AND its rounding,
so this is numerically no worse than the path it replaces. Causality is exact (the
reference masks with `triu(diagonal=1)` and a masked entry carries exactly zero softmax
weight), and zero-padding `head_dim` from 8 to 16 inside the kernel contributes exactly
zero to the QK^T contraction -- a zero row of K times anything is zero.

The one place precision is spent is the `P @ V` dot, where P is rounded to fp16 so the
product reaches the tensor cores. That is exactly what FlashAttention does too, and the
accumulator is fp32 throughout (finding 08: the fp32 residual is load-bearing; an fp16
accumulator in the softmax would eat the tolerance budget).

THE SECOND WIN, WHICH IS NOT ABOUT ATTENTION AT ALL
---------------------------------------------------
The candidate path around SDPA is

    qkv.split -> 3x view -> 3x transpose(1,2) -> SDPA -> transpose(1,2).reshape

and that final `reshape` after a transpose is a **real copy** of a whole activation
tensor, one extra kernel per layer. This kernel reads Q, K and V straight out of the
fused `[B, S, 3*d_model]` buffer by stride arithmetic and writes `[B, S, d_model]` in
head-major order -- which is bit-for-bit the layout `transpose(1,2).reshape` produces. So
the split, the views, the transposes and the repack all disappear with it.

WHY `head_dim` IS PADDED TO 16 INSIDE THE KERNEL
------------------------------------------------
sm_89's tensor-core instruction is `mma.sync.m16n8k16`, so Triton's `tl.dot` requires
every dimension >= 16. head_dim=8 (configs 7 and 11) is below that. Padding D to 16 with
a masked `tl.load(..., other=0.0)` costs zero extra HBM traffic. Padding in HBM instead is
exact but measured 1.2-2.7x SLOWER and is closed (finding 23) -- do not re-propose it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ------------------------------------------------------------------ the kernel

@triton.jit
def _attn_single_tile(
    QKV,                 # [B, S, 3*DM] fp16, contiguous in the last axis
    OUT,                 # [B, S, DM]   fp16
    stride_qkv_b, stride_qkv_s,
    stride_o_b, stride_o_s,
    scale,
    S: tl.constexpr,     # sequence length (constexpr: shapes are static here)
    DH: tl.constexpr,    # true head_dim
    DP: tl.constexpr,    # head_dim padded up to the MMA width, power of two
    DM: tl.constexpr,    # heads * head_dim, the stride from Q to K to V
    BM: tl.constexpr,    # query rows per program
    BN: tl.constexpr,    # key columns per program == next_pow2(S); ONE tile, no loop
):
    m_block = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    rm = m_block * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, DP)

    keep_m = rm < S
    keep_n = rn < S
    keep_d = rd < DH                      # False lanes load 0.0 -> exact zero padding

    head = QKV + b * stride_qkv_b + h * DH
    q = tl.load(head + rm[:, None] * stride_qkv_s + rd[None, :],
                mask=keep_m[:, None] & keep_d[None, :], other=0.0)
    k = tl.load(head + DM + rn[:, None] * stride_qkv_s + rd[None, :],
                mask=keep_n[:, None] & keep_d[None, :], other=0.0)
    v = tl.load(head + 2 * DM + rn[:, None] * stride_qkv_s + rd[None, :],
                mask=keep_n[:, None] & keep_d[None, :], other=0.0)

    # The whole score matrix for this query block, in fp32 registers. It never sees HBM.
    s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

    # Causal, exact. `rn < S` also kills the columns that only exist because BN was
    # rounded up to a power of two. Rows with rm >= S keep every column valid, so no row
    # is entirely -inf and no NaN is produced in the lanes we are about to discard.
    valid = (rn[None, :] <= rm[:, None]) & keep_n[None, :]
    s = tl.where(valid, s, float("-inf"))

    # ONE softmax. No running max, no running sum, no accumulator rescale.
    row_max = tl.max(s, 1)
    p = tl.exp(s - row_max[:, None])
    row_sum = tl.sum(p, 1)

    # P -> fp16 for the tensor cores (flash does the same); accumulate in fp32 and
    # normalize once at the end rather than S*S times.
    acc = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32) / row_sum[:, None]

    tl.store(OUT + b * stride_o_b + rm[:, None] * stride_o_s + h * DH + rd[None, :],
             acc.to(OUT.dtype.element_ty),
             mask=keep_m[:, None] & keep_d[None, :])


# ------------------------------------------------------------------ shape helpers

def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


MMA_WIDTH = 16          # sm_89's mma.sync is m16n8k16; tl.dot needs every dim >= this
MAX_REGS_PER_THREAD = 255       # architectural cap, sm_89


def padded_head_dim(head_dim: int) -> int:
    """head_dim rounded up to a power of two that the MMA can address."""
    return max(MMA_WIDTH, next_pow2(head_dim))


def register_bytes(seq_len: int, head_dim: int, block_m: int) -> int:
    """Per-program on-chip working set: the fp32 score tile plus the fp16 Q/K/V tiles.

    The score tile dominates and is what makes this kernel refuse long sequences: at
    S=1024 it alone is 4 MB, which is sixteen times the whole register file of an SM.
    """
    bn = next_pow2(seq_len)
    dp = padded_head_dim(head_dim)
    scores = block_m * bn * 4
    operands = (block_m + 2 * bn) * dp * 2
    return scores + operands


def register_budget(num_warps: int, regs_per_sm: int, warp_size: int = 32) -> int:
    """Bytes one thread block may hold in registers on the measured device.

    Two independent ceilings: the SM's whole register file, and the per-thread cap times
    the threads in the block. Both are properties of the hardware, not of this matrix.
    """
    return min(regs_per_sm, num_warps * warp_size * MAX_REGS_PER_THREAD) * 4


def fits(seq_len: int, head_dim: int, block_m: int, num_warps: int,
         regs_per_sm: int, warp_size: int = 32) -> bool:
    """Dispatch predicate. Shapes and MEASURED device properties only -- no config ids,
    no announced shape literals (CLAUDE.md rule 2).

    On this card it accepts S<=128 at head_dim 8/16/32/64 and refuses:
      * head_dim 256 (config 8)  -- the Q/K/V tiles alone overflow the register file
      * S = 1024 (config 13)     -- a 128x1024 fp32 score tile is 512 KB
      * S = 100000 (config 14)   -- likewise, by four orders of magnitude
    and on a card with a smaller register file it refuses more, without being retuned.
    """
    if seq_len < MMA_WIDTH or head_dim < 1 or block_m < MMA_WIDTH:
        return False
    if next_pow2(seq_len) < MMA_WIDTH:
        return False
    if block_m > next_pow2(seq_len):
        return False
    return register_bytes(seq_len, head_dim, block_m) <= register_budget(
        num_warps, regs_per_sm, warp_size)


# ----------------------------------------------------------- does it PAY?
#
# `fits` says the tile is legal. It does not say the kernel is worth using, and the
# sweep found two shapes where it is legal and LOSES: head_dim 128 (0.94x) and head_dim
# 256 (0.84x). The mechanism is specific and it is not about arithmetic.
#
# This kernel has NO LOOP. There is nothing to software-pipeline: every program is one
# long dependent chain (load Q/K/V -> dot -> softmax -> dot -> store). The only latency
# hiding available to it is OTHER RESIDENT BLOCKS on the same SM -- and the fp32 score
# tile plus the Q/K/V operands sit in registers, so the register working set caps
# residency directly. That is the price of deleting the tile loop: flash hides its
# memory latency inside the loop, we have to hide ours across blocks.
#
# Measured on this card, resident blocks per SM against op-level speedup vs SDPA:
#
#     head_dim   best tile   regs/block   blocks/SM   speedup
#            8      64 x 4       10752        6.1       1.58x   (cfg 7)
#           32      64 x 4       13312        4.9       1.55x   (cfg 1)
#           64      32 x 8       13312        4.9       1.19x   (cfg 10)
#          128      64 x 4       28672        2.3       0.94x   (cfg 9)
#          256      64 x 8       49152        1.3       0.84x   (cfg 8)
#
# The sign flips between 2.3 and 4.9 blocks per SM. MIN_RESIDENT_BLOCKS is that measured
# crossover, expressed against the device's own `regs_per_multiprocessor` and
# `max_threads_per_multi_processor` so a card with a different register file evaluates it
# differently without being retuned. Same discipline as ffn_fused.AMORTIZE_FRACTION.

# OPEN QUESTION, pre-registered in docs/findings/31 so it is a prediction and not a fit.
# The screen measured config 10 (head_dim 64) at -7.1% end to end -- the marginal case,
# sitting at exactly MIN_RESIDENT_BLOCKS, one pass, inside the +/-7% floor. Raising the
# threshold CANNOT fix it: head_dim 32 is a 1.55x win at the same 4.9 blocks/SM. The
# discriminator that would is `scores >= operands`, i.e.
#     block_m * BN * 4  >=  (block_m + 2 * BN) * pad16(head_dim) * 2
# which holds at head_dim 8 and 32 and fails at 64, 128 and 256. It is deliberately NOT
# implemented until a full sweep confirms the regression is real.
#
# ANSWERED AT GENERATION 41 (finding 51), AND THE PREDICATE STAYS AS IT IS.
# `bench/probes/g41_attn_audit/probe_three_arms.py` swept all three paths -- this kernel,
# `attn_looped`, and `sdpa+repack` -- over their complete legal grids on all thirteen
# runnable configs, L2-hot inside a captured graph, twice. The regression IS real and it
# is confined to ONE shape:
#
#     hd=32/8 shapes (cfg 1,2,3,4,5,6,7,11,12)  sdpa is 0.20x-0.90x of this kernel
#     hd=64   (cfg 10)                          sdpa is 1.119x of this kernel   <- real
#     hd=128/256, S=1024/100000                 declined already
#
# So the discriminator above (`scores >= operands`) would have separated the measured
# wins from the measured loss exactly as predicted -- and implementing it is still the
# wrong move, for a reason the prediction could not have known: `attn_looped` reaches
# 9.090 us on that same shape against sdpa's 9.988, so the answer at head_dim 64 is not
# "use the vendor" but "use the other Triton form", which `attn_choice.autotune_looped`
# already does by timing. Narrowing `pays` would remove the loop-free arm from a sweep
# that is not choosing it anyway, and would silently narrow it on every other card.
#
# What was added instead is a floor, not a predicate: `attn_choice.autotune_vendor` times
# the chosen tile against the vendor and hands the shape back if the vendor decisively
# wins. A measurement can decline; a formula fitted to one row cannot un-decline.
MIN_RESIDENT_BLOCKS = 4

# A margin below which the timer cannot separate two tiles: these kernels run in 1-13 us,
# the CUDA event timer resolves ~1 us, and the project's own noise floor is +/-7% (L29).
# The autotuner only overrides the derived tile by more than this.
DECISIVE = 0.10


def resident_blocks(seq_len: int, head_dim: int, block_m: int, num_warps: int,
                    regs_per_sm: int, max_threads_per_sm: int,
                    warp_size: int = 32) -> int:
    """How many of these blocks an SM can hold at once, by the two binding limits."""
    need = register_bytes(seq_len, head_dim, block_m)
    if need <= 0:
        return 0
    by_registers = (regs_per_sm * 4) // need
    by_threads = max_threads_per_sm // (num_warps * warp_size)
    return min(by_registers, by_threads)


def pays(seq_len: int, head_dim: int, block_m: int, num_warps: int,
         regs_per_sm: int, max_threads_per_sm: int, warp_size: int = 32) -> bool:
    """Is there enough block-level parallelism left to hide the memory latency?

    A loop-free kernel cannot pipeline, so this is the whole of its latency hiding.
    Below the crossover the kernel is legal, correct, and slower than the vendor's --
    which is exactly the case the dispatcher exists to decline.
    """
    return resident_blocks(seq_len, head_dim, block_m, num_warps, regs_per_sm,
                           max_threads_per_sm, warp_size) >= MIN_RESIDENT_BLOCKS


# --------------------------------------------------------------------- tiling
#
# The tile was CHOSEN BY SWEEPING, not by argument -- a mechanism argument cannot pick a
# tile size, and a sibling candidate lost at 0.88x on a guessed tile and won at 1.163x on
# a swept one. Every legal (block_m, num_warps) was timed against SDPA on every runnable
# shape in the matrix (min of 5 x do_bench, GPU lock held; INDICATIVE ONLY, L41).
#
#   S=128, head_dim 8 :  64x4 12.3us   128x4 13.3   128x8 13.3   64x2 13.6   32x2 14.3
#   S=128, head_dim 32:  64x4 20.5us   128x4 21.5   128x8 21.5   64x2 22.7   32x2 24.6
#   S=128, H=16 hd 8  :  64x4 30.7us   128x8 30.7   128x4 32.8   64x2 35.8   32x2 37.9
#   S=32,  head_dim 32:  16x2  7.2us    16x4  7.2    32x8  8.2    32x2  8.2    32x4  8.2
#
# THE FIRST ROW IS THE RECONCILIATION between the two proposals that asked for this.
# C-01 specifies one program per (batch, head) -- block_m == seq_len, the literal
# "single block". D-04 specifies a single *tile* of K/V without fixing the query block.
# **block_m = 64 beats block_m = 128 on every shape big enough to resolve, by ~5%.**
# So the single-K/V-tile claim survives and the single-block claim does not: 128 query
# rows put the fp32 score tile at 64 KB, which halves resident blocks per SM for no gain,
# and the K/V tiles are cheap enough to re-read once. We implement D-04's shape.
#
# num_warps is then not swept but DERIVED, and the sweep agrees with the derivation on
# every row above: enough warps that the score tile stays near half the 255-register
# architectural cap, so the operands and the softmax temporaries have room.

TILE_ROWS = 64                  # swept; see the table above
TARGET_REGS_PER_THREAD = 128    # half the sm_89 architectural cap of 255
MAX_WARPS = 8


def _warps_for(seq_len: int, head_dim: int, block_m: int, warp_size: int = 32) -> int:
    """Fewest warps (power of two) that keep the working set near TARGET_REGS_PER_THREAD."""
    need_regs = register_bytes(seq_len, head_dim, block_m) // 4
    w = 2
    while w < MAX_WARPS and need_regs > w * warp_size * TARGET_REGS_PER_THREAD:
        w *= 2
    return w


def choose_tile(seq_len: int, head_dim: int, regs_per_sm: int, max_threads_per_sm: int,
                warp_size: int = 32) -> tuple[int, int, int] | None:
    """(block_m, num_warps, num_stages), or None if no tile both fits and pays.

    Widest swept block first, narrowing only when the register file says it must.
    """
    bn = next_pow2(seq_len)
    bm = min(TILE_ROWS, bn)
    while bm >= MMA_WIDTH:
        w = _warps_for(seq_len, head_dim, bm, warp_size)
        if (fits(seq_len, head_dim, bm, w, regs_per_sm, warp_size)
                and pays(seq_len, head_dim, bm, w, regs_per_sm, max_threads_per_sm,
                         warp_size)):
            return (bm, w, 1)
        bm //= 2
    return None


def viable_tiles(seq_len: int, head_dim: int, regs_per_sm: int,
                 max_threads_per_sm: int, warp_size: int = 32) -> list[tuple[int, int, int]]:
    """Every (block_m, num_warps, num_stages) that both fits and pays on this device."""
    bn = next_pow2(seq_len)
    out = []
    for bm, w, st in SWEEP_TILES:
        if bm > bn:
            continue
        if not fits(seq_len, head_dim, bm, w, regs_per_sm, warp_size):
            continue
        if not pays(seq_len, head_dim, bm, w, regs_per_sm, max_threads_per_sm, warp_size):
            continue
        out.append((bm, w, st))
    return out


RTOL, ATOL = 0.02, 0.002          # the locked tolerance. Never widened.


def sdpa_reference(qkv: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """What this kernel must reproduce: SDPA plus the head-major repack.

    Lives here, next to the kernel whose contract it states, so that every tuner in this
    package gates on ONE definition. `attn_choice._reference` is an alias for it.
    """
    b, s, _ = qkv.shape
    dm = heads * head_dim
    q, k, v = qkv.split(dm, dim=2)
    q = q.view(b, s, heads, head_dim).transpose(1, 2)
    k = k.view(b, s, heads, head_dim).transpose(1, 2)
    v = v.view(b, s, heads, head_dim).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).reshape(b, s, dm)


def flushed_time(fn, reps: int = 2) -> float:
    """What `autotune_tile` ranked with from generation 23 to 41. Milliseconds.

    KEPT, NAMED, AND STILL THE DEFAULT -- because a candidate is only measurable against
    a parent that is byte-identical to the parent (finding 49's addendum, finding 50's
    structural fix). Generation 42's whole claim is "this timer picks the wrong tile on
    config 2", and the only way to measure that claim is to run both timers as two arms
    resident in ONE process. So the default here preserves v23's behaviour exactly and
    the candidate selects `hot_time` explicitly; the default flips when the measurement
    says it should, not before.

    Do not read the default as an endorsement. See `hot_time` for what it cannot do.
    """
    import triton.testing as tt
    return min(tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
               for _ in range(reps))


def hot_time(fn, reps: int = 2) -> float:
    """THE TIMER `attn_choice` HAS RANKED WITH SINCE GENERATION 40. Milliseconds.

    [L53]: use a timer whose regime matches the call site's -- or write in the code why
    not. This kernel is replayed inside a captured CUDA graph, on operands a 48 MB L2 has
    already seen (the whole QKV buffer is 6.29 MB at the announced shapes). `do_bench`
    models neither: it flushes L2 between reps and pays a launch per call.

    IT IS ALSO THE ONLY ONE OF THE TWO THAT CAN RESOLVE THESE KERNELS AT ALL, which is
    the sharper reason and the one generation 42 measured. `do_bench` times each call
    with a pair of CUDA events, and the event timer's quantum on this card is 1.024 us.
    These kernels run in 1.9-11 us. `bench/probes/g42_tile_timer/probe_timer_regimes.py`
    swept the full eight-tile grid under both timers, twice, on every shape the kernel
    accepts:

    The eight-tile grid at B=1, S=128, head_dim=32 -- the smallest announced shape, and
    the one where the quantum is the largest fraction of the quantity:

        flushed   5.120  5.120  5.120  5.120  5.120  6.144  6.144  6.144
        hot       1.905  2.260  2.409  2.438  2.486  3.344  3.423  3.718

    Five of the eight arms report the IDENTICAL number under `do_bench`, and the entire
    spread of the grid is one quantum. The ranking it produces is not noisy, it is
    absent -- and the tie then hands the decision to whatever tiebreak follows. Under the
    hot timer the same grid spreads 1.9x and the best arm is 1.28x clear of the tile that
    tie shipped. At B=4 the same quantization went the other way: a one-quantum artefact
    cleared the 10% `DECISIVE` bar for a tile the hot timer ranks 1.9% SLOWER, and the
    flushed sweep picked a DIFFERENT tile in each of two runs of itself. That instability
    is what finding 50 recorded as "deterministic against a comparable process state, not
    absolutely"; it is quantization, and the fix is resolution.

    `do_bench_cudagraph` times a graph of many replays and divides, so the event quantum
    is amortized rather than paid per call. That is why it resolves 1.9 us kernels and
    `do_bench` cannot.

    A device or context that refuses capture falls back to `do_bench` and the caller is
    none the wiser -- which is worse than failing closed on a tuner (v23's rule), and is
    why the fallback exists rather than a raise.
    """
    import triton.testing as tt
    try:
        return min(tt.do_bench_cudagraph(fn, rep=25, return_mode="min")
                   for _ in range(reps))
    except Exception:
        return min(tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
                   for _ in range(reps))


def autotune_tile(seq_len: int, head_dim: int, heads: int, batch: int,
                  device="cuda", timer=None,
                  replicates: int = 1) -> tuple[tuple[int, int, int], str]:
    """Pick the tile by TIMING it on this device, not by arguing about it.

    A mechanism argument cannot pick a tile size -- a sibling candidate lost at 0.88x on
    a guessed tile and won at 1.163x on a swept one -- and the sweep on this card shows no
    single formula fits: 64x4 wins at head_dim 32 while 32x8 wins at head_dim 64, at the
    identical register cost. So the candidate sweeps its own shape once, at prime time,
    before compilation and graph capture, and `choose_tile`'s derived answer is only the
    fallback if that fails.

    The probe batch is CAPPED. Per-program work does not depend on batch size once the
    grid is large enough to fill the machine, so timing config 6's 10000-row batch would
    allocate 983 MB to learn what 66 rows already say. The cap is derived from the
    measured SM count.

    `timer` IS THE WHOLE OF GENERATION 42, AND IT IS NOT A TILE
    ------------------------------------------------------------
    From generation 23 to 41 this routine ranked with `do_bench(warmup=10, rep=25)` --
    L2 flushed between reps, one launch per call, one pair of CUDA events per call. The
    event quantum on this card is 1.024 us and these kernels run in 1.9-11 us, so on the
    smallest shapes the sweep it produced was a table of TIES. `hot_time`'s docstring
    carries the measured grids: on config 2 five of the eight tiles reported the
    identical 5.120 us, the tie fell through to the derived-tile tiebreak below, and the
    tile that tiebreak kept is one the hot timer ranks 1.28x behind the best arm. That is
    an uncapped scoring row, and finding 51 called it the largest unclaimed op-level
    margin on its table.

    So the defect is not "the wrong tile was hardcoded for config 2" and the fix is not a
    tile. It is that this routine and `attn_choice` -- two tuners answering the same
    question about the same kernel, one calling the other -- ranked on different
    instruments, and only one of them can resolve the kernels being ranked. Pass
    `hot_time` and the routine selects `(16, 4, 1)` on config 2 BY ITSELF, from shapes
    and measured device properties, with no config id anywhere; and it is then right on
    shapes nobody has swept.

    `timer=None` means `flushed_time`, i.e. exactly what v23 through v41 shipped. That
    default is deliberate and temporary: see `flushed_time`. Nothing else about the
    decision changes under either timer -- same grid, same trial budget, same `DECISIVE`
    bar, same tiebreak, same probe batch.

    CORRECTNESS BEFORE TIMING, PER ARM -- ALSO NEW
    -----------------------------------------------
    `attn_choice` gates every arm against the reference at the locked tolerance before
    admitting it to the timing set, on the rule that a fast wrong kernel must never win a
    sweep. This routine did not, and was the one place in the package where a tile could
    be selected on speed alone. Measured, the gap is latent and not live: the g42 probe
    checked all eight tiles on all ten accepted shapes, twice, and every arm matched. It
    is closed anyway, because [L38] a check nobody has seen fail is indistinguishable
    from a check that cannot.

    `replicates` IS GENERATION 43, AND IT IS THE OTHER HALF OF GENERATION 42
    ------------------------------------------------------------------------
    The hot timer bought resolution and paid for it in variance. Finding 53 measured
    both and pre-registered this fix rather than netting them out.

    A selection rule has an ACCURACY and it has a STABILITY, and they are different
    properties. `flushed_time` could not separate these kernels at all, so its sweep was
    a table of ties that fell through to the derived-tile tiebreak -- every time, on
    every shape, in every process state. That is a systematic error, and its stability
    was doing unnoticed work: a plan that never varies adds no variance to the
    measurements taken of it (L29). `hot_time` can separate the arms, and on the shapes
    whose TRUE margin is small it converted that bias into a variance. Measured over six
    independent runs at B=4, whose two leading tiles are ~2% apart: the flushed sweep
    selected one tile in 6 of 6 and the hot sweep selected THREE DIFFERENT TILES in 6,
    because prime time with a resident model occasionally supplies a spurious margin past
    the 10% `DECISIVE` bar. At B=1, where the true margin is 28%, the hot sweep selected
    the same tile in 7 of 7.

    `DECISIVE = 0.10` was calibrated against the OLD instrument's noise. Re-fitting it
    against the new one would be a constant fitted to this matrix -- rule 2 forbids it,
    and it would be wrong on the next card anyway. So the SWEEP is replicated instead:
    `replicates` passes over the whole grid, reduced PER ARM BY THEIR MINIMUM, and then
    v23's decision rule run once on the reduced table with nothing else changed.

    A FLOOR, NOT A VOTE -- AND THE DIFFERENCE WAS MEASURED, NOT REASONED
    ---------------------------------------------------------------------
    Finding 53 pre-registered this as "displace only if both sweeps clear `DECISIVE` and
    agree on the winner". Built and measured, that rule is WRONG, and how it fails says
    what the defect actually is. `bench/probes/g43_stable_tiles/sweep_grids.py` dumps
    every arm of every sweep in the model's own regime (a second model resident and
    primed, inside `inference_mode`). Six back-to-back sweeps at B=4:

        tile        sw1     sw2     sw3     sw4     sw5     sw6     floor
        (16,4,1)  2.743   4.002   2.547   4.029   2.741   2.574    2.547
        (64,4,1)  2.711   2.517   3.752   2.708   2.529   4.245    2.517   <- derived
        winner    (64)    (64)    (16)    (64)    (64)    (16)

    The two spurious displacements -- sweeps 3 and 6, reported at 1.473x and 1.649x --
    are **not** a challenger reading fast. They are the INCUMBENT reading slow: 3.752 and
    4.245 against its own 2.517 floor, while the challenger barely moved. Contamination
    on this harness is one-sided (a descheduled host thread, an unsettled graph and a
    co-resident allocation can only make a reading slower), so the noise lands on
    whichever arm it lands on, and a per-sweep `min()` then hands the sweep to the arm
    that happened to be missed. **A vote between two contaminated rankings is decided by
    where the contamination fell.** At B=1 that cost the vote rule the shape v42 won:
    measured over ten fresh processes, the same shape whose true margin is 28% held the
    derived tile in 5 of 10 -- always the 5 where the tuner primed second.

    The floor is the estimator the mechanism demands and the one CLAUDE.md already
    prescribes for a card whose clocks will not lock: the mean is a statistic about how
    often the machine misbehaved, the minimum is a statistic about the code. Reduced by
    the floor over adjacent PAIRS of the sweeps above, every window gives the same
    answer -- B=4 holds the derived tile (0.988x, inside `DECISIVE`), B=1 displaces it
    (1.277x, 1.284x, 1.284x), which is the value four independent measurements across
    three generations agree on.

    Every arm is timed in every sweep and reduced over the same number of readings, so
    no arm gets a budget its rivals did not; an arm that failed to time in any sweep is
    dropped rather than reduced over fewer (finding 47's best-of-N handicap, inverted).
    The count of distinct per-sweep winners is reported in the reason string, because
    [L62]'s cheap check -- look at how many distinct answers the grid contains -- should
    be visible in the ledger even when the reduction absorbs the instability.

    The cost is one extra pass of the grid at prime time, ~1 s against the frontier's
    existing 14-67 s of tuning, and no extra compilation: the second sweep hits Triton's
    JIT cache, and the probe tensor and its reference are allocated once for all sweeps
    so that the replicates differ only in WHEN they ran.

    `replicates=1` is EXACTLY the parent's routine, including its reason strings, for the
    same byte-identical-control reason `timer=None` is still `flushed_time`. The default
    moves when a measurement says it should, not before.
    """
    timer = flushed_time if timer is None else timer
    n_sweeps = max(1, int(replicates))
    props = torch.cuda.get_device_properties(device)
    tiles = viable_tiles(seq_len, head_dim, props.regs_per_multiprocessor,
                         props.max_threads_per_multi_processor, props.warp_size)
    if not tiles:
        raise ValueError("no viable tile")
    fallback = choose_tile(seq_len, head_dim, props.regs_per_multiprocessor,
                           props.max_threads_per_multi_processor, props.warp_size)
    try:
        probe_b = max(1, min(batch, 4 * props.multi_processor_count // max(1, heads)))
        dm = heads * head_dim
        qkv = torch.randn(probe_b, seq_len, 3 * dm, device=device, dtype=torch.float16)
        scale = head_dim ** -0.5
        ref = sdpa_reference(qkv, heads, head_dim)
        # ONE probe tensor and ONE reference for EVERY sweep. The replicates must differ
        # only in when they ran -- a fresh tensor per sweep would resample the allocator
        # as well as the timer, and a disagreement would then not be attributable to the
        # thing being tested.
        sweeps: list[dict] = []
        dropped: list[tuple] = []
        for _ in range(n_sweeps):
            timed = {}
            for bm, w, st in tiles:
                try:
                    fn = (lambda bm=bm, w=w, st=st:
                          single_tile_attention(qkv, heads, head_dim, scale, bm, w, st))
                    out = fn()
                    torch.cuda.synchronize()
                    if not torch.allclose(out.float(), ref.float(),
                                          atol=ATOL, rtol=RTOL):
                        if (bm, w, st) not in dropped:
                            dropped.append((bm, w, st))
                        continue
                    timed[(bm, w, st)] = timer(fn, 2)
                except Exception:
                    continue
            sweeps.append(timed)
        del qkv, ref
        # The timer is NAMED in every reason string. [L36]: a candidate whose mechanism
        # never engages is its parent with extra build time, and the only externally
        # visible trace of which instrument ranked this sweep is what it says it was.
        # From g43 the same holds for the replicate count and for the DISAGREEMENT that
        # made a sweep hold -- otherwise a tuner that reverted because its answer did not
        # replicate is indistinguishable from one that confirmed the derived tile.
        tn = getattr(timer, "__name__", str(timer))
        note = f" ({len(dropped)} dropped on tolerance)" if dropped else ""
        scored = [t for t in sweeps if t]
        if scored:
            # THE REPLICATES ARE REDUCED PER ARM BY THEIR MINIMUM, and then v23's rule
            # runs on the reduced table, UNCHANGED. See the docstring for why this is a
            # floor and not a vote; the short form is that the contamination this is
            # correcting lands on the INCUMBENT's reading, so a per-sweep vote is
            # decided by which arm happened to be hit.
            #
            # An arm is eligible only if it was timed in EVERY sweep, so every arm's
            # floor is a minimum over the same number of readings. An arm reduced over
            # fewer trials than its rivals is finding 47's best-of-N handicap inverted.
            floor: dict[tuple, float] = {}
            for k in scored[0]:
                if all(k in t for t in scored):
                    floor[k] = min(t[k] for t in scored)
            if floor:
                best, best_ms = min(floor.items(), key=lambda kv: kv[1])
                base_ms = floor.get(fallback)
                margin = (base_ms / best_ms) if base_ms else float("nan")
                # [L62]'s cheap check, reported rather than merely run: how many distinct
                # answers did the individual sweeps give? A grid whose winner moves every
                # sweep has not ranked anything, and the ledger should be able to see
                # that even when the reduction absorbs it.
                winners = [min(t, key=t.get) for t in scored]
                spread = ("" if n_sweeps == 1 else
                          f", {len(set(winners))} distinct per-sweep winners over "
                          f"{len(scored)} sweeps")
                head = (f"autotuned over {len(tiles)} tiles at batch {probe_b} "
                        f"by {tn}{note}")
                # THE DERIVED TILE HOLDS THE GROUND unless something beats it DECISIVELY.
                # A candidate whose own tile varies run to run adds that noise to every
                # measurement taken of it (L29), so the bar stays where v23 put it.
                if base_ms is None or best_ms < base_ms * (1.0 - DECISIVE):
                    return best, (f"{head}: {best} beat the derived tile {fallback} "
                                  f"decisively ({margin:.3f}x{spread})")
                return fallback, (f"derived tile {fallback}, confirmed by {tn} against "
                                  f"{len(tiles)} tiles at batch {probe_b}{note} "
                                  f"(best challenger {best} at {margin:.3f}x{spread}, "
                                  f"inside {DECISIVE:.0%})")
    except Exception:
        pass
    return fallback, "derived (autotune unavailable)"


def applies(seq_len: int, head_dim: int, props) -> tuple[bool, str]:
    """The dispatch predicate as the candidate asks it. (use_it, human-readable reason).

    `props` is a `torch.cuda.get_device_properties` result -- every number below is
    measured off the device at run time, none is a config id or an announced shape.
    """
    tile = choose_tile(seq_len, head_dim, props.regs_per_multiprocessor,
                       props.max_threads_per_multi_processor, props.warp_size)
    if tile is None:
        need = register_bytes(seq_len, head_dim, min(TILE_ROWS, next_pow2(seq_len)))
        return False, (f"declined: {need/1024:.0f} KB of register working set per program "
                       f"leaves fewer than {MIN_RESIDENT_BLOCKS} blocks resident on a "
                       f"{props.regs_per_multiprocessor*4/1024:.0f} KB register file")
    bm, w, _ = tile
    n = resident_blocks(seq_len, head_dim, bm, w, props.regs_per_multiprocessor,
                        props.max_threads_per_multi_processor, props.warp_size)
    return True, f"single-tile attention: {bm}x{w} warps, {n} blocks resident per SM"


# --------------------------------------------------------------------- launcher

def single_tile_attention(qkv: torch.Tensor, heads: int, head_dim: int,
                          scale: float, block_m: int, num_warps: int,
                          num_stages: int = 1) -> torch.Tensor:
    """Causal SDPA over the fused `[B, S, 3*d_model]` projection buffer.

    Returns `[B, S, d_model]` fp16 in head-major order -- the same bytes as
    `sdpa(q, k, v, is_causal=True).transpose(1, 2).reshape(B, S, d_model)`, with the
    transpose, the reshape copy and the three input views never happening.
    """
    b, s, three_dm = qkv.shape
    dm = heads * head_dim
    assert three_dm == 3 * dm, f"expected [B, S, 3*{dm}], got {tuple(qkv.shape)}"
    out = torch.empty((b, s, dm), device=qkv.device, dtype=qkv.dtype)
    grid = (triton.cdiv(s, block_m), heads, b)
    _attn_single_tile[grid](
        qkv, out,
        qkv.stride(0), qkv.stride(1),
        out.stride(0), out.stride(1),
        scale,
        S=s, DH=head_dim, DP=padded_head_dim(head_dim), DM=dm,
        BM=block_m, BN=next_pow2(s),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# --------------------------------------------------------------- offline sweep

# Every tile the sweep considers. Not used at run time -- `choose_tile` is; this list
# exists so the choice can be re-derived on another card instead of being trusted.
SWEEP_TILES: tuple[tuple[int, int, int], ...] = (
    (16, 2, 1), (16, 4, 1),
    (32, 2, 1), (32, 4, 1), (32, 8, 1),
    (64, 2, 1), (64, 4, 1), (64, 8, 1),
    (128, 4, 1), (128, 8, 1),
)


def sweep_tile(seq_len: int, head_dim: int, heads: int, batch: int,
               device: str = "cuda", reps: int = 5) -> list[tuple]:
    """Time every legal tile on a real tensor. INDICATIVE ONLY (L41).

    A probe: it proposes a tile, it concludes nothing about the candidate, and nothing it
    prints belongs in the ledger. It exists so the tile is swept rather than guessed --
    the difference between 0.88x and 1.163x on the sibling candidate that learned it the
    hard way. Take the GPU lock before calling it.
    """
    import triton.testing as tt

    props = torch.cuda.get_device_properties(device)
    dm = heads * head_dim
    qkv = torch.randn(batch, seq_len, 3 * dm, device=device, dtype=torch.float16)
    scale = head_dim ** -0.5
    rows = []
    for bm, w, st in SWEEP_TILES:
        if bm > next_pow2(seq_len):
            continue
        if not fits(seq_len, head_dim, bm, w, props.regs_per_multiprocessor,
                    props.warp_size):
            continue
        try:
            fn = (lambda bm=bm, w=w, st=st:
                  single_tile_attention(qkv, heads, head_dim, scale, bm, w, st))
            fn()
            ms = min(tt.do_bench(fn, warmup=25, rep=50, return_mode="min")
                     for _ in range(reps))
            rows.append((bm, w, st, ms))
        except Exception:                              # a tile the compiler refuses
            rows.append((bm, w, st, float("inf")))
    rows.sort(key=lambda r: r[3])
    return rows
