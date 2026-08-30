"""Q/K/V projection AND causal attention in one kernel: the projection never sees HBM.

WHAT THIS REPLACES
------------------
v23's path is two kernels:

    qkv = F.linear(norm1(x).to(fp16), Wqkv, bqkv)      # cuBLAS, writes [B, S, 3*D]
    ctx = single_tile_attention(qkv, ...)              # ours, reads it straight back

On config 6 (B=10000, S=128, D=128, H=4) the fresh profile puts those at 15.4% and 15.5%
of wall time. The first writes 983 MB per layer and the second reads it back immediately.
This kernel deletes the buffer: each program loads the NORMALIZED INPUT tile, projects
its own head's Q, K and V from it in registers, and attends -- one launch, no [B,S,3D]
tensor anywhere.

WHY THE 96 KB WEIGHT DOES NOT HAVE TO FIT
-----------------------------------------
The brief for this candidate framed the question as "the fused QKV weight is 128 x 384
fp16 = 96 KB against 99 KB of opt-in shared memory -- does it fit?". **It does, with 3 KB
to spare, and that is the wrong question**, because a program that owns one head never
needs the whole weight. Head h of Q uses columns [h*DH, h*DH+DH) of Wq, and likewise for
K and V, so the resident slice is

    3 * d_model * pad16(head_dim) * 2 bytes

which on the announced rows is 24 KB (head_dim 32, configs 1-6, 12), 12 KB (head_dim 8,
config 11), 3 KB (config 7), 48 KB (head_dim 64, config 10) and 96 KB (head_dim 128,
config 9). Only config 9 approaches the 96 KB figure, and it is declined for other
reasons. `smem_resident_bytes` below is that arithmetic, evaluated against the device's
own `shared_memory_per_block_optin` -- so a 48 KB-smem card declines head_dim 64 without
being retuned.

The binding constraint is not the weight. It is the X TILE the three projections all read:
[BN, d_model] fp16, which is 32 KB at d_model 128 and 256 KB at d_model 1024 (config 8).
That is what refuses the wide model, and it is a shape fact, not a config id.

WHY block_m MUST EQUAL THE SEQUENCE LENGTH HERE, WHICH REVERSES v23
-------------------------------------------------------------------
v23 swept and found block_m = 64 beats block_m = 128 at S = 128, by ~5%: a 128-row fp32
score tile halves resident blocks per SM for no reduction in work. That reasoning does not
survive the fusion, because now there IS a reduction in work.

A program covering query rows [m*BM, (m+1)*BM) still needs K and V for ALL S key rows, so
it must project S rows of K and S rows of V. With cdiv(S, BM) query blocks, K and V get
projected cdiv(S, BM) times over. Rows of projection per (batch, head):

    fused  = cdiv(S, BM) * (BM + 2*BN)          BM=64, S=128 -> 2 * 320 = 640
    actual = 3 * S                                                  -> 384

so block_m = 64 does **1.67x the projection arithmetic** the cuBLAS GEMM did, while
block_m = BN = next_pow2(S) does exactly 1.0x. `pays` below charges that redundancy
against the HBM bytes the fusion saves, at the device's MEASURED ridge point -- the one
number that says how many FLOPs a saved byte is worth on this card.

THE HONEST ROOFLINE, WRITTEN BEFORE ANY MEASUREMENT (L33, L41)
--------------------------------------------------------------
Config 6, per layer, 10000 batch elements of 128 tokens at d_model 128:

  * separate QKV GEMM: 504 GFLOP over the 4 layers and 5.24 GB of traffic -- an
    arithmetic intensity of 96 FLOP/B against this card's 144 FLOP/B ridge, so it is
    BANDWIDTH-bound, and 5.24 GB / 8.8 ms measured = 595 GB/s is 97% of the 613.7 GB/s
    roofline. It is not a badly written kernel with headroom in it. It is at the wall.
  * v23's attention: 336 GFLOP in 8.9 ms = 37.7 TFLOP/s, 43% of the 88.2 TFLOP/s peak.
  * combined: 840 GFLOP in 17.7 ms = 47.5 TFLOP/s.

Fusing does not delete the projection's FLOPs, it MOVES them into our kernel. Traffic
falls ~5x (3.28 GB -> 0.66 GB per layer) and intensity rises to ~328 FLOP/B, well past the
ridge: the stage stops being bandwidth-bound and becomes compute-bound at a 9.5 ms floor.
**So the fused kernel wins if and only if it exceeds 47.5% of peak** -- i.e. if it runs
the projection dots better than v23's attention currently runs its own. There is no free
lunch in the 15.4%; the GEMM was already at its roofline. This is a genuine coin flip and
the prediction is recorded here so the measurement can falsify it.

PRECISION
---------
The projection accumulates in fp32 and rounds to fp16 once, with the bias added in fp32
before that rounding -- structurally identical to what `F.linear` on fp16 operands with an
fp32-accumulating cuBLAS epilogue already does, so no rounding step is added or removed.
The softmax stays fp32 throughout (finding 08). Padding head_dim to the MMA width is exact:
the padded weight columns load as zero, the padded bias lanes load as zero, so the padded
Q/K/V lanes are exactly zero and contribute exactly zero to the QK^T contraction.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .attn_single_tile import (MMA_WIDTH, next_pow2, padded_head_dim,  # noqa: F401
                               register_budget)


# ------------------------------------------------------------------ the kernel

@triton.jit
def _attn_qkv_fused(
    X,                   # [B, S, DB] fp16 -- the NORMALIZED input, contiguous
    W,                   # [DB, 3*DM] fp16 -- the fused Q|K|V weight, TRANSPOSED
    BIAS,                # [3*DM]     fp16
    OUT,                 # [B, S, DM] fp16 -- head-major, out_proj's layout
    stride_x_b, stride_x_s,
    stride_w_d,
    stride_o_b, stride_o_s,
    scale,
    S: tl.constexpr,     # sequence length
    DB: tl.constexpr,    # d_model; a power of two, so tl.arange can address it
    DH: tl.constexpr,    # true head_dim
    DP: tl.constexpr,    # head_dim padded up to the MMA width
    DM: tl.constexpr,    # heads * head_dim, the stride from Wq to Wk to Wv
    BM: tl.constexpr,    # query rows per program
    BN: tl.constexpr,    # key rows per program == next_pow2(S); ONE tile, no loop
    SAME: tl.constexpr,  # BM == BN, so the query tile IS the key tile -- load it once
):
    m_block = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    rm = m_block * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, DB)
    rp = tl.arange(0, DP)

    keep_m = rm < S
    keep_n = rn < S
    keep_p = rp < DH                      # False lanes load 0.0 -> exact zero padding

    # --- this head's slice of the projection. 3 * DB * DP * 2 bytes, not 3 * DB * DM. --
    wcol = h * DH + rp
    woff = rd[:, None] * stride_w_d + wcol[None, :]
    wq = tl.load(W + woff, mask=keep_p[None, :], other=0.0)
    wk = tl.load(W + woff + DM, mask=keep_p[None, :], other=0.0)
    wv = tl.load(W + woff + 2 * DM, mask=keep_p[None, :], other=0.0)
    bq = tl.load(BIAS + wcol, mask=keep_p, other=0.0).to(tl.float32)
    bk = tl.load(BIAS + wcol + DM, mask=keep_p, other=0.0).to(tl.float32)
    bv = tl.load(BIAS + wcol + 2 * DM, mask=keep_p, other=0.0).to(tl.float32)

    # --- the projection, in registers. Nothing below ever reaches HBM. ----------------
    xbase = X + b * stride_x_b
    xq = tl.load(xbase + rm[:, None] * stride_x_s + rd[None, :],
                 mask=keep_m[:, None], other=0.0)
    q = (tl.dot(xq, wq, out_dtype=tl.float32) + bq[None, :]).to(X.dtype.element_ty)

    if SAME:
        # BM == BN == next_pow2(S) means cdiv(S, BM) == 1, so m_block is 0 and rm == rn.
        # One [BN, DB] tile serves the query, key and value projections alike.
        xn = xq
    else:
        xn = tl.load(xbase + rn[:, None] * stride_x_s + rd[None, :],
                     mask=keep_n[:, None], other=0.0)
    k = (tl.dot(xn, wk, out_dtype=tl.float32) + bk[None, :]).to(X.dtype.element_ty)
    v = (tl.dot(xn, wv, out_dtype=tl.float32) + bv[None, :]).to(X.dtype.element_ty)

    # --- from here down this is v23's single tile, unchanged --------------------------
    s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

    # Causal, exact. `rn < S` also kills the columns that exist only because BN was
    # rounded up to a power of two. Rows with rm >= S keep every column valid, so no row
    # is entirely -inf and no NaN reaches the lanes we are about to discard.
    valid = (rn[None, :] <= rm[:, None]) & keep_n[None, :]
    s = tl.where(valid, s, float("-inf"))

    row_max = tl.max(s, 1)
    p = tl.exp(s - row_max[:, None])
    row_sum = tl.sum(p, 1)

    acc = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32) / row_sum[:, None]

    tl.store(OUT + b * stride_o_b + rm[:, None] * stride_o_s + h * DH + rp[None, :],
             acc.to(OUT.dtype.element_ty),
             mask=keep_m[:, None] & keep_p[None, :])


# ------------------------------------------------------- what a program must hold

def smem_resident_bytes(seq_len: int, d_model: int, head_dim: int, block_m: int,
                        elem_size: int = 2) -> int:
    """Operands that stay LIVE across the whole program, and therefore must coexist.

    The three weight slices are read by three separate dots and the X tile feeds all
    three, so none of them can be retired early. Everything else (the score tile, P, the
    accumulator) is a dot RESULT and lives in registers, which is a different budget.

    This is the predicate that refuses config 8: at d_model 1024 the X tile alone is
    256 KB.
    """
    bn = next_pow2(seq_len)
    dp = padded_head_dim(head_dim)
    x_tile = (block_m if block_m == bn else block_m + bn) * d_model * elem_size
    weights = 3 * d_model * dp * elem_size
    return x_tile + weights


def register_bytes(seq_len: int, d_model: int, head_dim: int, block_m: int) -> int:
    """Per-program register working set: the fp32 score tile, the projected Q/K/V, and
    the fp32 output accumulator. The score tile dominates, exactly as in v23."""
    bn = next_pow2(seq_len)
    dp = padded_head_dim(head_dim)
    scores = block_m * bn * 4
    projected = (block_m + 2 * bn) * dp * 2
    acc = block_m * dp * 4
    return scores + projected + acc


def is_pow2(n: int) -> bool:
    return n > 0 and not (n & (n - 1))


def fits(seq_len: int, d_model: int, head_dim: int, block_m: int, num_warps: int,
         regs_per_sm: int, smem_optin: int, warp_size: int = 32) -> bool:
    """Dispatch predicate. Shapes and MEASURED device properties only -- no config ids,
    no announced shape literals (CLAUDE.md rule 2).

    On this card it accepts d_model 32 and 128 at head_dim 8/16/32/64 with S <= 128, and
    refuses:
      * d_model 1024 (config 8)   -- the [128, 1024] X tile is 256 KB of smem
      * head_dim 128 (config 9)   -- 96 KB of weight slice plus the X tile overflows smem
      * seq_len 1024 (config 13)  -- a 1024x1024 fp32 score tile is 4 MB
      * seq_len 100000 (config 14)-- likewise, by orders of magnitude
    """
    if seq_len < MMA_WIDTH or head_dim < 1 or block_m < MMA_WIDTH:
        return False
    if not is_pow2(d_model) or d_model < MMA_WIDTH:
        return False                     # tl.arange over the contraction axis
    bn = next_pow2(seq_len)
    if bn < MMA_WIDTH or block_m > bn:
        return False
    if smem_resident_bytes(seq_len, d_model, head_dim, block_m) > smem_optin:
        return False
    return register_bytes(seq_len, d_model, head_dim, block_m) <= register_budget(
        num_warps, regs_per_sm, warp_size)


def projection_redundancy(seq_len: int, block_m: int) -> float:
    """How many times over this tiling projects Q, K and V, against projecting them once.

    A query block needs every key row, so K and V are re-projected once per query block.
    Exactly 1.0 when block_m covers the whole (power-of-two-padded) sequence.
    """
    bn = next_pow2(seq_len)
    blocks = -(-seq_len // block_m)
    return blocks * (block_m + 2 * bn) / (3.0 * seq_len)


def fused_read_bytes(seq_len: int, d_model: int, heads: int, block_m: int,
                     elem_size: int = 2) -> int:
    """Bytes the fused kernel reads per batch element per layer, WITHOUT crediting L2.

    Every head re-reads the SAME [BN, d_model] tile of the normalized input, because each
    program owns one head and needs the whole model width to project it. That factor of
    `heads` is the fusion's one structural cost, and it is invisible until you count it.
    """
    bn = next_pow2(seq_len)
    blocks = -(-seq_len // block_m)
    x_rows = bn if block_m == bn else block_m + bn
    return blocks * heads * x_rows * d_model * elem_size


def split_io_bytes(seq_len: int, d_model: int, head_dim: int, heads: int,
                   block_m: int, elem_size: int = 2) -> int:
    """Bytes the two-kernel path moves for the same work, excluding the final output
    store, which both paths pay identically.

    The GEMM reads the normalized input once and writes [S, 3*d_model]; the attention
    kernel reads that buffer back, once per query block.
    """
    bn = next_pow2(seq_len)
    dp = padded_head_dim(head_dim)
    blocks = -(-seq_len // block_m)
    gemm = seq_len * d_model * elem_size + seq_len * 3 * heads * head_dim * elem_size
    attn_read = blocks * heads * (block_m + 2 * bn) * dp * elem_size
    return gemm + attn_read


def moves_fewer_bytes(seq_len: int, d_model: int, head_dim: int, heads: int,
                      block_m: int, elem_size: int = 2) -> bool:
    """Does the fusion actually reduce traffic, or does the per-head re-read undo it?

    THIS IS THE PREDICATE THAT CONFIG 11 FORCED. At 16 heads and d_model 128 the fused
    kernel reads the 32 KB input tile sixteen times -- 524 KB per batch element per layer
    against the 328 KB the GEMM-plus-attention pair moves. The fusion is a LOSS there and
    the op-level probe measured it as one (0.82x), while it is a win at 2, 4 and 16 heads
    with a narrower model. Counting bytes separates those cases with no fitted constant:
    it is `heads * next_pow2(S) * d_model` against `S * d_model + 6 * S * d_model` plus
    the attention read, i.e. roughly `heads <= 7` at block_m == BN, derived rather than
    tuned. No L2 credit is taken, which is the conservative direction.
    """
    return (fused_read_bytes(seq_len, d_model, heads, block_m, elem_size)
            <= split_io_bytes(seq_len, d_model, head_dim, heads, block_m, elem_size))


def pays(seq_len: int, d_model: int, head_dim: int, heads: int, block_m: int,
         ridge_flop_per_byte: float, elem_size: int = 2) -> bool:
    """Is the redundant projection arithmetic cheaper than the HBM bytes it buys back?

    Two independent conditions, both derived:

    1. **Traffic.** `moves_fewer_bytes` -- the fusion must not move more bytes than the
       pair it replaces. Refuses many-head shapes where the per-head re-read of the input
       tile costs more than the buffer it deletes.
    2. **Arithmetic.** A query block needs every key row, so K and V are re-projected once
       per query block. `ridge_flop_per_byte` is the device's own measured ridge point:
       how many FLOPs cost the same time as one byte of HBM traffic on THIS card. Below it
       the trade is profitable; above it we are buying bandwidth with arithmetic we cannot
       afford. A hardware property, not a fitted constant.
    """
    if not moves_fewer_bytes(seq_len, d_model, head_dim, heads, block_m, elem_size):
        return False
    dp = padded_head_dim(head_dim)
    bn = next_pow2(seq_len)
    blocks = -(-seq_len // block_m)
    extra_rows = blocks * (block_m + 2 * bn) - 3 * seq_len
    if extra_rows <= 0:
        return True
    added_flops = heads * extra_rows * d_model * dp * 2
    saved_bytes = 2 * seq_len * 3 * heads * head_dim * elem_size
    if saved_bytes <= 0:
        return False
    return added_flops / saved_bytes <= ridge_flop_per_byte


# --------------------------------------------------------------------- tiling
#
# block_m = next_pow2(S) is the DERIVED choice and it is derived from work, not from a
# sweep: it is the only tiling with `projection_redundancy == 1.0`. That reverses v23's
# swept preference for block_m = 64, which was correct for a kernel that read Q/K/V
# out of HBM and had no projection to redo. The tuner still times the alternatives,
# because a work argument cannot price occupancy (L: v20 lost at 0.88x on a guessed tile
# and won at 1.163x on a swept one) -- but the derived tile holds unless something beats
# it decisively.

TARGET_REGS_PER_THREAD = 128    # half the sm_89 architectural cap of 255
MAX_WARPS = 8
DECISIVE = 0.10                 # these kernels run in single-digit us; see v23


def _warps_for(seq_len: int, d_model: int, head_dim: int, block_m: int,
               warp_size: int = 32) -> int:
    need_regs = register_bytes(seq_len, d_model, head_dim, block_m) // 4
    w = 2
    while w < MAX_WARPS and need_regs > w * warp_size * TARGET_REGS_PER_THREAD:
        w *= 2
    return w


SWEEP_TILES: tuple[tuple[int, int], ...] = (
    (16, 2), (16, 4),
    (32, 2), (32, 4), (32, 8),
    (64, 2), (64, 4), (64, 8),
    (128, 4), (128, 8),
    (256, 8),
)


def viable_tiles(seq_len: int, d_model: int, head_dim: int, heads: int,
                 props) -> list[tuple[int, int, int]]:
    """Every (block_m, num_warps, num_stages) that both fits and pays on this device."""
    bn = next_pow2(seq_len)
    ridge = ridge_point(props)
    out = []
    for bm, w in SWEEP_TILES:
        if bm > bn:
            continue
        if not fits(seq_len, d_model, head_dim, bm, w,
                    props.regs_per_multiprocessor,
                    props.shared_memory_per_block_optin, props.warp_size):
            continue
        if not pays(seq_len, d_model, head_dim, heads, bm, ridge):
            continue
        out.append((bm, w, 1))
    return out


_RIDGE_CACHE: dict[str, float] = {}


def ridge_point(props) -> float:
    """FLOPs per byte at which this device's compute and bandwidth ceilings meet.

    Taken from the ORACLE's own calibration (`ratchet.oracle.device.calibrate`), which
    measures achieved bandwidth on the actual card and reads its peak from the compute
    capability -- the same number `ledger/device.json` caches and every other dispatch
    decision in this repo is entitled to use. It is a measured device property, not a
    literal, and on a different card it recalibrates rather than being retuned.

    If calibration is unavailable the trade is DECLINED (ridge 0), which falls back to
    the parent's path. Guessing a ridge would be worse than not fusing.
    """
    name = getattr(props, "name", "?")
    if name in _RIDGE_CACHE:
        return _RIDGE_CACHE[name]
    ridge = 0.0
    try:
        from ratchet.oracle.device import calibrate
        ridge = float(calibrate().ridge_point_flop_per_byte)
    except Exception:
        ridge = 0.0
    _RIDGE_CACHE[name] = ridge
    return ridge


def choose_tile(seq_len: int, d_model: int, head_dim: int, heads: int,
                props) -> tuple[int, int, int] | None:
    """The DERIVED tile, and it is always a member of `viable_tiles` so the autotuner can
    time it as the incumbent.

    Widest legal query block first -- that is the one that projects K and V exactly once
    -- and among tiles of that width, the warp count closest to the register-pressure
    target. None if nothing both fits and pays.
    """
    tiles = viable_tiles(seq_len, d_model, head_dim, heads, props)
    if not tiles:
        return None
    widest = max(t[0] for t in tiles)
    want = _warps_for(seq_len, d_model, head_dim, widest, props.warp_size)
    return min((t for t in tiles if t[0] == widest),
               key=lambda t: (abs(t[1] - want), t[1]))


def _why_no_tile(seq_len: int, d_model: int, head_dim: int, heads: int, props) -> str:
    """Name the binding constraint. A dispatcher that declines without saying WHY is a
    hardcoded table wearing a costume (L28); the reason is reported per candidate run."""
    bn = next_pow2(seq_len)
    smem = smem_resident_bytes(seq_len, d_model, head_dim, bn)
    optin = props.shared_memory_per_block_optin
    if smem > optin:
        return (f"{smem/1024:.0f} KB of resident operands (a {bn}x{d_model} input tile "
                f"plus a 3x{d_model}x{padded_head_dim(head_dim)} weight slice) against "
                f"{optin/1024:.0f} KB opt-in shared memory")
    if not moves_fewer_bytes(seq_len, d_model, head_dim, heads, bn):
        f = fused_read_bytes(seq_len, d_model, heads, bn)
        s = split_io_bytes(seq_len, d_model, head_dim, heads, bn)
        return (f"the fusion would MOVE MORE BYTES than it saves: {heads} heads each "
                f"re-read the {bn}x{d_model} input tile, {f/1024:.0f} KB per sequence "
                f"against the pair's {s/1024:.0f} KB")
    if register_bytes(seq_len, d_model, head_dim, bn) > register_budget(
            MAX_WARPS, props.regs_per_multiprocessor, props.warp_size):
        need = register_bytes(seq_len, d_model, head_dim, bn)
        return (f"a {need/1024:.0f} KB register working set (dominated by a "
                f"{bn}x{bn} fp32 score tile) past the "
                f"{props.regs_per_multiprocessor*4/1024:.0f} KB register file")
    return "no (block_m, num_warps) both fits and pays"


def applies(seq_len: int, d_model: int, head_dim: int, heads: int,
            props) -> tuple[bool, str]:
    """(use_it, human-readable reason). Every number is measured off the device."""
    if not is_pow2(d_model):
        return False, f"declined: d_model {d_model} is not a power of two"
    tile = choose_tile(seq_len, d_model, head_dim, heads, props)
    if tile is None:
        return False, "declined: " + _why_no_tile(seq_len, d_model, head_dim, heads, props)
    bm, w, _ = tile
    red = projection_redundancy(seq_len, bm)
    return True, (f"qkv-fused attention: {bm}x{w} warps, "
                  f"{smem_resident_bytes(seq_len, d_model, head_dim, bm)/1024:.0f} KB "
                  f"resident operands, projection redundancy {red:.2f}x")


def autotune_tile(seq_len: int, d_model: int, head_dim: int, heads: int, batch: int,
                  device="cuda") -> tuple[tuple[int, int, int], str]:
    """Pick the tile by TIMING it on this device, with the derived tile as the incumbent.

    Same protocol as v23's: probe batch capped from the measured SM count, the derived
    tile only displaced by a margin the timer can actually resolve. INDICATIVE ONLY --
    this proposes a tile, it concludes nothing (L41).
    """
    props = torch.cuda.get_device_properties(device)
    tiles = viable_tiles(seq_len, d_model, head_dim, heads, props)
    if not tiles:
        raise ValueError("no viable tile")
    fallback = choose_tile(seq_len, d_model, head_dim, heads, props)
    try:
        import triton.testing as tt
        probe_b = max(1, min(batch, 4 * props.multi_processor_count // max(1, heads)))
        x = torch.randn(probe_b, seq_len, d_model, device=device, dtype=torch.float16)
        w = torch.randn(d_model, 3 * heads * head_dim, device=device,
                        dtype=torch.float16) * 0.05
        bias = torch.zeros(3 * heads * head_dim, device=device, dtype=torch.float16)
        scale = head_dim ** -0.5
        timed = {}
        for bm, nw, st in tiles:
            try:
                fn = (lambda bm=bm, nw=nw, st=st:
                      fused_qkv_attention(x, w, bias, heads, head_dim, scale, bm, nw, st))
                fn()
                timed[(bm, nw, st)] = min(
                    tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
                    for _ in range(2))
            except Exception:
                continue
        del x, w, bias
        if timed:
            best, best_ms = min(timed.items(), key=lambda kv: kv[1])
            base_ms = timed.get(fallback)
            if base_ms is None or best_ms < base_ms * (1.0 - DECISIVE):
                return best, (f"autotuned over {len(tiles)} tiles at batch {probe_b}: "
                              f"{best} beat the derived tile decisively")
            return fallback, (f"derived tile {fallback}, confirmed against "
                              f"{len(tiles)} tiles at batch {probe_b}")
    except Exception:
        pass
    if fallback is None:
        raise ValueError("no viable tile")
    return fallback, "derived (autotune unavailable)"


# --------------------------------------------------------------------- launcher

def fused_qkv_attention(x: torch.Tensor, w_t: torch.Tensor, bias: torch.Tensor,
                        heads: int, head_dim: int, scale: float,
                        block_m: int, num_warps: int,
                        num_stages: int = 1) -> torch.Tensor:
    """Causal SDPA over `linear(x, W, b)`, with the projection never reaching HBM.

    `x`      -- [B, S, d_model] fp16, the already-normalized layer input.
    `w_t`    -- [d_model, 3*d_model] fp16: `cat([Wq, Wk, Wv]).t().contiguous()`.
    `bias`   -- [3*d_model] fp16: `cat([bq, bk, bv])`.

    Returns [B, S, d_model] fp16 head-major -- the same bytes as
    `sdpa(...).transpose(1, 2).reshape(B, S, d_model)`, which is the layout `out_proj`
    already wants.
    """
    b, s, db = x.shape
    dm = heads * head_dim
    assert w_t.shape == (db, 3 * dm), f"expected W^T [{db}, {3*dm}], got {tuple(w_t.shape)}"
    out = torch.empty((b, s, dm), device=x.device, dtype=x.dtype)
    bn = next_pow2(s)
    grid = (triton.cdiv(s, block_m), heads, b)
    _attn_qkv_fused[grid](
        x, w_t, bias, out,
        x.stride(0), x.stride(1),
        w_t.stride(0),
        out.stride(0), out.stride(1),
        scale,
        S=s, DB=db, DH=head_dim, DP=padded_head_dim(head_dim), DM=dm,
        BM=block_m, BN=bn, SAME=(block_m == bn),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


# --------------------------------------------------------------- offline sweep

def sweep_tile(seq_len: int, d_model: int, head_dim: int, heads: int, batch: int,
               device: str = "cuda", reps: int = 5) -> list[tuple]:
    """Time every legal tile on a real tensor. INDICATIVE ONLY (L41). Take the GPU lock."""
    import triton.testing as tt

    props = torch.cuda.get_device_properties(device)
    dm = heads * head_dim
    x = torch.randn(batch, seq_len, d_model, device=device, dtype=torch.float16)
    w = torch.randn(d_model, 3 * dm, device=device, dtype=torch.float16) * 0.05
    bias = torch.zeros(3 * dm, device=device, dtype=torch.float16)
    scale = head_dim ** -0.5
    rows = []
    for bm, nw, st in viable_tiles(seq_len, d_model, head_dim, heads, props):
        try:
            fn = (lambda bm=bm, nw=nw, st=st:
                  fused_qkv_attention(x, w, bias, heads, head_dim, scale, bm, nw, st))
            fn()
            ms = min(tt.do_bench(fn, warmup=25, rep=50, return_mode="min")
                     for _ in range(reps))
            rows.append((bm, nw, st, ms))
        except Exception:
            rows.append((bm, nw, st, float("inf")))
    rows.sort(key=lambda r: r[3])
    return rows
