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


def autotune_tile(seq_len: int, head_dim: int, heads: int, batch: int,
                  device="cuda") -> tuple[tuple[int, int, int], str]:
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
    """
    props = torch.cuda.get_device_properties(device)
    tiles = viable_tiles(seq_len, head_dim, props.regs_per_multiprocessor,
                         props.max_threads_per_multi_processor, props.warp_size)
    if not tiles:
        raise ValueError("no viable tile")
    fallback = choose_tile(seq_len, head_dim, props.regs_per_multiprocessor,
                           props.max_threads_per_multi_processor, props.warp_size)
    try:
        import triton.testing as tt
        probe_b = max(1, min(batch, 4 * props.multi_processor_count // max(1, heads)))
        dm = heads * head_dim
        qkv = torch.randn(probe_b, seq_len, 3 * dm, device=device, dtype=torch.float16)
        scale = head_dim ** -0.5
        timed = {}
        for bm, w, st in tiles:
            try:
                fn = (lambda bm=bm, w=w, st=st:
                      single_tile_attention(qkv, heads, head_dim, scale, bm, w, st))
                fn()
                timed[(bm, w, st)] = min(
                    tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
                    for _ in range(2))
            except Exception:
                continue
        del qkv
        if timed:
            best, best_ms = min(timed.items(), key=lambda kv: kv[1])
            base_ms = timed.get(fallback)
            # THE DERIVED TILE HOLDS THE GROUND unless something beats it DECISIVELY.
            # These kernels run in 1-13 us and the CUDA event timer resolves ~1 us, so
            # inside DECISIVE the ranking is noise -- and a candidate whose own tile
            # varies run to run adds that noise to every measurement taken of it (L29).
            if base_ms is None or best_ms < base_ms * (1.0 - DECISIVE):
                return best, (f"autotuned over {len(tiles)} tiles at batch {probe_b}: "
                              f"{best} beat the derived tile decisively")
            return fallback, (f"derived tile {fallback}, confirmed against "
                              f"{len(tiles)} tiles at batch {probe_b}")
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
