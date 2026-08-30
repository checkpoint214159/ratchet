"""One Triton kernel for a WHOLE transformer layer: attention and FFN in a single launch.

    LN1 -> Q|K|V -> causal attention -> out-proj -> +residual(fp32)
        -> LN2 -> FFN1 -> GELU -> FFN2 -> +residual(fp32)

Six kernels per layer on the frontier become one, and nothing between the first load and
the final store touches HBM.

READ THIS BEFORE BELIEVING ANY NUMBER FROM IT
=============================================
`v19_norm_fused` fused three of those six and measured **FLAT** on config 6 (finding 29).
The traffic model says it should have won 23%:

    v18's three kernels        4588 MB / layer   (add+LN2+cast, fused_ffn, next LN1)
    v19's one megakernel       2294 MB / layer
    saved                      2294 MB / layer  x4 layers = 9.18 GB = 15.0 ms at
                                                613.7 GB/s, on a 65 ms config

    predicted   -23%          measured   +0.4%

The saving was real and something ate it: the achieved fraction of the roofline. The
frontier's six-kernel layer runs at ~79% of the 613.7 GB/s roofline (31.5 GB against
51.3 ms of roofline for a 65 ms measured call); a monolithic CTA holding a whole sequence
does not. **A megakernel does not delete work, it moves it into a kernel with worse
occupancy** -- and on this card that trade has already been measured once at break-even.

WHAT THE WHOLE-LAYER CASE CHANGES, AND WHERE IT IS DECIDED
==========================================================
Fusing the whole layer removes traffic v19 could not (config 6, per layer):

    LN1                        read x32, write xn16          983 MB
    QKV gemm                   read xn16, write qkv16*3     1311 MB   <- untouched by v17/v19
    attention                  read qkv16*3, write ctx16    1311 MB
    out-proj                   read ctx16, write o16         655 MB
    add + LN2 + cast                                        1966 MB
    fused_ffn                                               1638 MB
    ------------------------------------------------------ --------
    frontier total                                          7864 MB
    this kernel (x read once per query tile + reloaded for
    the residual, y written once)                          ~1966 MB

Four layers: 31.5 GB -> ~7.9 GB, i.e. 51.3 ms -> 12.8 ms of roofline. **But that inverts
which side of the ridge the model sits on.** Config 6 is 1.174 TFLOP against 7.9 GB =
149 FLOP/B, past this card's measured ridge of 144 FLOP/B. The fused layer is COMPUTE
bound, and its outcome is decided by one number: what fraction of the 88.2 BF16-TFLOP/s
peak one monolithic CTA reaches while running eight dependent `tl.dot` chains with no
loop to software-pipeline.

    10% of peak  133.2 ms  0.49x      30% of peak   44.4 ms  1.46x
    15% of peak   88.8 ms  0.73x      40% of peak   33.3 ms  1.95x
    20% of peak   66.6 ms  0.98x      50% of peak   26.6 ms  2.44x

**Break-even is 20% of tensor peak.** Pre-registered (L33): the sign of this candidate is
set by achieved tensor-core utilisation, not by the traffic argument that motivated it,
and anything outside 0.5x-2.5x should be disbelieved before it is celebrated.

THE REGISTER FILE IS THE BINDING CONSTRAINT, AND IT IS MEASURED, NOT ARGUED
==========================================================================
Compiled for the announced shapes and read straight off the Triton compiler (no timing):

    shape                 BM   warps   n_regs   n_spills   smem      blocks/SM
    S128 D128 F128 H4     128     8      255       166     67584         1
    S128 D128 F128 H4     128    16      128       546     67584         1
    S128 D32  F32  H4     128     8      254         0     18432         1
    S128 D128 F128 H2     128     8      255       212     81920         1

At `BM = seq_len` -- one program per sequence, the literal reading of A-03/B-01 -- the
d_model=128 shapes SPILL, and more warps makes it worse (Triton caps n_regs to fit the
threads, so each thread holds less and spills more). Spilled registers go to local memory,
which is HBM-backed: the fusion's entire premise is to stop touching HBM, and at BM=128 it
starts touching it again through the back door.

So `BM` is a real tiling axis, not a constant. A program owns BM query rows and computes
K and V for the WHOLE sequence (attention is a reduction over all keys), so splitting a
sequence across programs costs a redundant K/V projection per extra program and buys back
register headroom and occupancy. That trade cannot be argued -- it is swept, at prime
time, with `n_spills` read from the compiler as a hard filter (v20: 0.88x guessed,
1.163x tuned).

WHY IT FITS ON THIS CARD AT ALL
===============================
`ffn_dim == d_model` and `seq_len <= 128` on eleven of the fourteen announced rows. The
working set lives in REGISTERS -- 256 KB per SM, measured (65536 32-bit registers) --
not in the 99 KB opt-in shared memory, which is why the predicate below is written against
`regs_per_multiprocessor` rather than against smem.

THREE THINGS THAT WOULD BE EASY TO GET WRONG
============================================
1. **The residual is fp32 and never rounds.** Finding 08: an fp16 residual failed 12 of
   14 configs. Here it is an fp32 register value from load to store, and unlike the
   frontier the attention output is never rounded to fp16 on its way into it -- so this
   kernel is strictly MORE accurate than the path it replaces. Measured on identical
   inputs against an fp32 reference, one layer, batch 66:

       shape                megakernel max_abs   frontier max_abs
       S128 D128 H4              2.403e-03           2.502e-03
       S128 D32  H4              2.564e-03           2.949e-03
       S128 D128 H16             2.185e-03           2.618e-03
       S128 D128 H2              2.395e-03           2.664e-03

2. **The softmax accumulates in fp32** and causality is exact: `tl.where` puts -inf above
   the diagonal, which is what the reference's `triu(diagonal=1)` computes, and a masked
   entry carries exactly zero softmax weight.
3. **The out-projection is accumulated per head.** `out_proj(concat_h ctx_h)` equals
   `sum_h ctx_h @ WO[h*head_dim:(h+1)*head_dim, :]` -- the same contraction re-associated
   over the head axis. It is done this way because **Triton cannot slice a register-
   resident block value**: each head's Q/K/V comes from a `tl.dot` against a COLUMN SLICE
   OF THE WEIGHT, which is a memory offset and therefore legal. That one restriction
   dictates the entire shape of this kernel.

`head_dim` below the MMA width (configs 7 and 11 at 8) is padded to 16 with masked loads
that read exactly 0.0, so padded lanes contribute exactly zero to every contraction --
the same argument, and the same measured-free mechanism, as `attn_single_tile`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .attn_single_tile import MMA_WIDTH, MAX_REGS_PER_THREAD, next_pow2, padded_head_dim


# ------------------------------------------------------------------------- the kernel

@triton.jit
def _layer_block(
    X,                    # fp32 [B, S, D]   residual stream in
    Y,                    # fp32 [B, S, D]   residual stream out
    N1W, N1B,             # fp32 [D]         norm1 weight / bias
    N2W, N2B,             # fp32 [D]         norm2 weight / bias
    WQKV, BQKV,           # fp16 [D, 3D] / [3D]   already transposed from nn.Linear
    WO, BO,               # fp16 [D, D]  / [D]
    W1, B1,               # fp16 [D, F]  / [F]
    W2, B2,               # fp16 [F, D]  / [D]
    stride_b, stride_s,
    scale, eps,
    H,
    S: tl.constexpr, D: tl.constexpr, FD: tl.constexpr,
    DH: tl.constexpr, DP: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    """One program owns BM query rows of one sequence, and every key of that sequence.

    Attention is a reduction over the whole sequence, so a program cannot own fewer keys
    than there are. When BM < BN the K/V projections are recomputed by each program that
    shares the sequence; that redundancy is the price of register headroom, and which
    side of it wins is swept in `select_tile`, never assumed.
    """
    m_block = tl.program_id(0)
    seq = tl.program_id(1)

    rm = m_block * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    rf = tl.arange(0, FD)
    rp = tl.arange(0, DP)
    keep_m = rm < S
    keep_n = rn < S
    keep_p = rp < DH                      # False lanes load 0.0 -> exact zero padding

    n1w = tl.load(N1W + rd)[None, :]
    n1b = tl.load(N1B + rd)[None, :]

    # ---- LayerNorm 1 over EVERY key row: this is what K and V are projected from -------
    # The fp32 load dies here. Keeping it live across the head loop would add BN*D*4
    # bytes to the peak working set, and a spill to local memory costs far more than a
    # re-read -- local memory is HBM-backed, which is the very traffic this kernel exists
    # to remove.
    xk = tl.load(X + seq * stride_b + rn[:, None] * stride_s + rd[None, :],
                 mask=keep_n[:, None], other=0.0)
    mu = tl.sum(xk, 1) / D
    xc = xk - mu[:, None]
    var = tl.sum(xc * xc, 1) / D
    xnk = (xc * tl.rsqrt(var[:, None] + eps) * n1w + n1b).to(WQKV.dtype.element_ty)

    # ---- LayerNorm 1 over MY query rows -----------------------------------------------
    # Recomputed rather than sliced out of `xnk`, because Triton block values cannot be
    # indexed by a runtime range. When BM == BN this is the same rows and the compiler
    # sees it as such; the arithmetic is a few flops per element either way.
    off = seq * stride_b + rm[:, None] * stride_s + rd[None, :]
    xq = tl.load(X + off, mask=keep_m[:, None], other=0.0)
    muq = tl.sum(xq, 1) / D
    xcq = xq - muq[:, None]
    varq = tl.sum(xcq * xcq, 1) / D
    xnq = (xcq * tl.rsqrt(varq[:, None] + eps) * n1w + n1b).to(WQKV.dtype.element_ty)

    # ---- attention, one head at a time, accumulating into the out-projection ----------
    acc = tl.zeros((BM, D), dtype=tl.float32)
    for h in tl.range(0, H):
        col = h * DH + rp
        wq = tl.load(WQKV + rd[:, None] * (3 * D) + col[None, :],
                     mask=keep_p[None, :], other=0.0)
        wk = tl.load(WQKV + rd[:, None] * (3 * D) + (D + col)[None, :],
                     mask=keep_p[None, :], other=0.0)
        wv = tl.load(WQKV + rd[:, None] * (3 * D) + (2 * D + col)[None, :],
                     mask=keep_p[None, :], other=0.0)
        bq = tl.load(BQKV + col, mask=keep_p, other=0.0).to(tl.float32)
        bk = tl.load(BQKV + D + col, mask=keep_p, other=0.0).to(tl.float32)
        bv = tl.load(BQKV + 2 * D + col, mask=keep_p, other=0.0).to(tl.float32)

        # Bias added in fp32 BEFORE the rounding to fp16, as F.linear's GEMM epilogue does.
        q = (tl.dot(xnq, wq, out_dtype=tl.float32) + bq[None, :]).to(xnq.dtype)
        k = (tl.dot(xnk, wk, out_dtype=tl.float32) + bk[None, :]).to(xnk.dtype)
        v = (tl.dot(xnk, wv, out_dtype=tl.float32) + bv[None, :]).to(xnk.dtype)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        # Exact causality. `keep_n` also kills the columns that exist only because BN was
        # rounded up to a power of two. Rows with rm >= S keep every column valid, so no
        # row is entirely -inf and no NaN is produced in lanes we discard.
        s = tl.where((rn[None, :] <= rm[:, None]) & keep_n[None, :], s, float("-inf"))
        row_max = tl.max(s, 1)
        p = tl.exp(s - row_max[:, None])
        row_sum = tl.sum(p, 1)
        ctx = tl.dot(p.to(xnq.dtype), v, out_dtype=tl.float32) / row_sum[:, None]

        # out_proj(concat_h ctx_h) == sum_h ctx_h @ WO[h*DH:(h+1)*DH, :]. Padded rows of
        # WO load as 0.0, so padded lanes of ctx contribute exactly nothing.
        wo = tl.load(WO + col[:, None] * D + rd[None, :],
                     mask=keep_p[:, None], other=0.0)
        acc += tl.dot(ctx.to(xnq.dtype), wo, out_dtype=tl.float32)

    # ---- attention residual, in fp32 and in registers ---------------------------------
    # The attention output is NOT rounded to fp16 on the way in. The frontier's
    # `F.linear(...).float()` does round it; this is one fp16 rounding step fewer on the
    # accumulating path (finding 08).
    res = xq + acc + tl.load(BO + rd)[None, :].to(tl.float32)

    # ---- LayerNorm 2 ------------------------------------------------------------------
    mu2 = tl.sum(res, 1) / D
    xc2 = res - mu2[:, None]
    var2 = tl.sum(xc2 * xc2, 1) / D
    xn2 = (xc2 * tl.rsqrt(var2[:, None] + eps)
           * tl.load(N2W + rd)[None, :] + tl.load(N2B + rd)[None, :]).to(xnq.dtype)

    # ---- FFN: two GEMMs, the hidden activation never leaving registers -----------------
    w1 = tl.load(W1 + rd[:, None] * FD + rf[None, :])
    hid = tl.dot(xn2, w1, out_dtype=tl.float32) + tl.load(B1 + rf)[None, :].to(tl.float32)
    # Exact erf GELU, matching the reference's approximate="none". The tanh form differs
    # by up to ~1e-3 relative, half the entire 2e-3 budget spent on an approximation
    # nobody asked for.
    hid = hid * 0.5 * (1.0 + tl.erf(hid * 0.70710678118654752440))
    w2 = tl.load(W2 + rf[:, None] * D + rd[None, :])
    y = (res + tl.dot(hid.to(xnq.dtype), w2, out_dtype=tl.float32)
         + tl.load(B2 + rd)[None, :].to(tl.float32))

    tl.store(Y + off, y, mask=keep_m[:, None])


# ------------------------------------------------------------------ shape / device fit

def register_bytes(seq_len: int, d_model: int, ffn_dim: int, head_dim: int,
                   block_m: int) -> int:
    """Peak on-chip working set of one program, in bytes.

    Two phases; the peak is whichever is larger. Written as an explicit sum rather than a
    fitted constant, so another card evaluates the same arithmetic instead of inheriting
    a number tuned here. It is an ESTIMATE and deliberately not the dispatch's last word
    -- `select_tile` reads the compiler's own `n_spills` for the exact shape.
    """
    bn = next_pow2(seq_len)
    dp = padded_head_dim(head_dim)
    xnk = bn * d_model * 2            # fp16 keys' normalized input, live across the loop
    xnq = block_m * d_model * 2       # fp16 queries' normalized input
    resid = block_m * d_model * 4     # fp32 residual for my rows
    acc = block_m * d_model * 4       # fp32 out-projection accumulator
    score = block_m * bn * 4          # fp32 score tile for one head
    probs = block_m * bn * 2          # fp16 probabilities for the P@V dot
    qkv = (block_m + 2 * bn) * dp * 2  # one head's Q (mine) and K, V (all)
    attn_phase = xnk + xnq + resid + acc + score + probs + qkv

    hidden = block_m * ffn_dim * 4    # fp32 FFN hidden activation
    weights = (d_model * ffn_dim + ffn_dim * d_model) * 2
    ffn_phase = resid + xnq + hidden + weights
    return max(attn_phase, ffn_phase)


def register_budget(num_warps: int, regs_per_sm: int, warp_size: int = 32) -> int:
    """Bytes one thread block may hold in registers on the MEASURED device.

    Two independent ceilings -- the SM's whole register file, and the architectural
    per-thread cap times the threads in the block. Both are hardware properties.
    """
    return min(regs_per_sm, num_warps * warp_size * MAX_REGS_PER_THREAD) * 4


TILE_ROWS: tuple[int, ...] = (128, 64, 32, 16)
WARP_CHOICES: tuple[int, ...] = (4, 8, 16)


def fits(seq_len: int, d_model: int, ffn_dim: int, head_dim: int, num_heads: int,
         block_m: int, num_warps: int, regs_per_sm: int, warp_size: int = 32) -> bool:
    """Legality gate: shapes and MEASURED device properties only -- no config ids and no
    announced shape literals (CLAUDE.md rule 2).

    On this card it refuses, whatever the tile:

      * d_model 1024 (config 8)  -- the key rows' normalized tile alone is 256 KB
      * seq_len 1024 (config 13) -- a 1024-wide fp32 score row is 4 KB per query row
      * seq_len 100000 (14)      -- likewise, by two further orders of magnitude

    and on a card with a smaller register file it refuses more, without being retuned.
    """
    if seq_len < MMA_WIDTH or d_model < MMA_WIDTH or ffn_dim < MMA_WIDTH:
        return False
    if block_m < MMA_WIDTH or block_m > next_pow2(seq_len):
        return False
    if head_dim < 1 or num_heads < 1 or head_dim * num_heads != d_model:
        return False
    for n in (seq_len, d_model, ffn_dim):
        if n & (n - 1):
            return False              # tl.arange needs powers of two
    return register_bytes(seq_len, d_model, ffn_dim, head_dim, block_m) <= register_budget(
        num_warps, regs_per_sm, warp_size)


# A program owns BM query rows of ONE sequence, so the grid is batch * ceil(S / BM).
# Below one full wave the kernel leaves SMs idle for the whole call and cannot win however
# good the tile is -- config 2 (batch 1) would run on one or two of this card's 66 SMs.
# The threshold is a fraction of the LAST wave, so it is scale-free: 64 programs on 66 SMs
# is 97% utilised and accepted, 16 programs is 24% and declined, and a 132-SM card
# declines what this one accepts.
MIN_SM_UTILIZATION = 0.75

# A margin below which the timer cannot separate two tiles (L29: the noise floor is +/-7%
# and these probes run in microseconds). The derived tile holds the ground unless a rival
# beats it decisively -- a candidate whose own tile varies run to run adds that variance
# to every measurement taken of it.
DECISIVE = 0.10

# The sweep runs at prime time, before compilation and graph capture, and every tile in it
# costs a Triton compile. Cap it: the tiles are already ordered by the compiler's own
# spill verdict, so the tail of the list is the part least worth timing.
MAX_SWEEP = 6


def programs(batch_size: int, seq_len: int, block_m: int) -> int:
    return batch_size * -(-seq_len // block_m)


def sm_utilization(n_programs: int, sm_count: int) -> float:
    """Fraction of the grid's SM-slots that do work, under whole-wave scheduling."""
    if n_programs <= 0 or sm_count <= 0:
        return 0.0
    waves = -(-n_programs // sm_count)
    return n_programs / (waves * sm_count)


def pays(batch_size: int, seq_len: int, block_m: int, sm_count: int) -> bool:
    """Is there enough program-level parallelism to fill the measured machine?"""
    return sm_utilization(programs(batch_size, seq_len, block_m),
                          sm_count) >= MIN_SM_UTILIZATION


def viable_tiles(seq_len: int, d_model: int, ffn_dim: int, head_dim: int, num_heads: int,
                 batch_size: int, regs_per_sm: int, sm_count: int,
                 warp_size: int = 32) -> list[tuple[int, int]]:
    """Every (block_m, num_warps) that is legal on this device and fills it.

    Largest tile first: a bigger BM recomputes K/V fewer times, so the least redundant
    tile that still fills the machine is the one to prefer before any timing happens.
    """
    out = []
    for bm in TILE_ROWS:
        if not pays(batch_size, seq_len, bm, sm_count):
            continue
        for w in WARP_CHOICES:
            if fits(seq_len, d_model, ffn_dim, head_dim, num_heads, bm, w, regs_per_sm,
                    warp_size):
                out.append((bm, w))
    return out


# --------------------------------------------------------------------------- launcher

def fused_layer(x: torch.Tensor,
                n1w: torch.Tensor, n1b: torch.Tensor,
                n2w: torch.Tensor, n2b: torch.Tensor,
                wqkv: torch.Tensor, bqkv: torch.Tensor,
                wo: torch.Tensor, bo: torch.Tensor,
                w1: torch.Tensor, b1: torch.Tensor,
                w2: torch.Tensor, b2: torch.Tensor,
                num_heads: int, head_dim: int, scale: float, eps: float,
                block_m: int = 64, num_warps: int = 8,
                grid_m: int = 0, bn: int = 0, dp: int = 0):
    """A whole transformer layer in one launch. `x` is fp32 [B, S, D]; so is the result.

    Every weight is already TRANSPOSED relative to nn.Linear's [out, in] layout, because
    the kernel contracts over the leading axis.

    `grid_m`, `bn` and `dp` may be passed in already resolved. That is not a convenience:
    the candidate calls this from inside `torch.compile`'s traced region, and a Python
    `bit_length()` or an integer ceil-divide resolved in there is a graph break waiting to
    happen -- a sibling candidate's first screen read -18.9% for exactly that reason. All
    three are decided once, at prime time, and arrive here as constants.
    """
    b, s, d = x.shape
    f = w1.shape[1]
    y = torch.empty_like(x)
    _layer_block[(grid_m or -(-s // block_m), b)](
        x, y, n1w, n1b, n2w, n2b, wqkv, bqkv, wo, bo, w1, b1, w2, b2,
        x.stride(0), x.stride(1), scale, eps, num_heads,
        S=s, D=d, FD=f, DH=head_dim, DP=dp or padded_head_dim(head_dim),
        BM=block_m, BN=bn or next_pow2(s), num_warps=num_warps, num_stages=1,
    )
    return y


def _probe_args(batch, seq_len, d_model, ffn_dim, head_dim, num_heads, device):
    """Randomly initialised arguments of exactly the shapes `fused_layer` expects."""
    f32 = dict(device=device, dtype=torch.float32)
    f16 = dict(device=device, dtype=torch.float16)
    return (
        torch.randn(batch, seq_len, d_model, **f32),
        torch.ones(d_model, **f32), torch.zeros(d_model, **f32),
        torch.ones(d_model, **f32), torch.zeros(d_model, **f32),
        torch.randn(d_model, 3 * d_model, **f16) * 0.05,
        torch.zeros(3 * d_model, **f16),
        torch.randn(d_model, d_model, **f16) * 0.05, torch.zeros(d_model, **f16),
        torch.randn(d_model, ffn_dim, **f16) * 0.05, torch.zeros(ffn_dim, **f16),
        torch.randn(ffn_dim, d_model, **f16) * 0.05, torch.zeros(d_model, **f16),
    )


def spills_for(tile, seq_len, d_model, ffn_dim, head_dim, num_heads, args) -> int | None:
    """Spilled bytes per thread, read off the Triton compiler for this exact shape.

    This is a MEASURED device property -- the register allocator's own verdict on this
    kernel on this card -- not a modelled estimate and not a fitted constant. It is the
    one number that separates "the working set fits" from "the working set fits and the
    compiler did not have to push half of it to local memory", and local memory is
    HBM-backed, so a spilling megakernel reintroduces exactly the traffic it exists to
    remove. Returns None if the compiler will not say.
    """
    bm, w = tile
    x = args[0]
    y = torch.empty_like(x)
    try:
        k = _layer_block[(-(-seq_len // bm), x.shape[0])](
            x, y, *args[1:], x.stride(0), x.stride(1), head_dim ** -0.5, 1e-5, num_heads,
            S=seq_len, D=d_model, FD=ffn_dim, DH=head_dim,
            DP=padded_head_dim(head_dim), BM=bm, BN=next_pow2(seq_len),
            num_warps=w, num_stages=1)
        return int(k.n_spills)
    except Exception:
        return None


def select_tile(seq_len: int, d_model: int, ffn_dim: int, head_dim: int, num_heads: int,
                batch: int, device="cuda") -> tuple[tuple[int, int], str]:
    """Choose (block_m, num_warps) by COMPILING and TIMING on this device.

    Three stages, cheapest first:

      1. legality and occupancy, from measured device properties;
      2. `n_spills == 0`, read from the compiler -- a hard filter, because a spilling
         program pays HBM traffic for its own registers and the whole premise is gone;
      3. a timed sweep of the survivors on a probe batch capped from the measured SM
         count, since per-program work does not depend on batch once the grid fills the
         machine.

    The largest surviving tile is the fallback and holds the ground unless a rival beats
    it by more than DECISIVE.
    """
    props = torch.cuda.get_device_properties(device)
    tiles = viable_tiles(seq_len, d_model, ffn_dim, head_dim, num_heads, batch,
                         props.regs_per_multiprocessor, props.multi_processor_count,
                         props.warp_size)
    if not tiles:
        raise ValueError("no viable tile")

    probe_b = max(1, min(batch, 2 * props.multi_processor_count))
    args = _probe_args(probe_b, seq_len, d_model, ffn_dim, head_dim, num_heads, device)
    spills = {t: spills_for(t, seq_len, d_model, ffn_dim, head_dim, num_heads, args)
              for t in tiles}

    # A tile the COMPILER rejects is not a tile. `register_bytes` models the register
    # file; it does not model the shared memory Triton stages `tl.dot` operands through,
    # and at head_dim 128 (config 9) that is what refuses: "Required: 147456, Hardware
    # limit: 101376" -- the measured 99 KB opt-in smem, on a shape whose registers fit.
    # The compiler is the authority on its own resources, so its verdict is the filter.
    compiles = [t for t in tiles if spills.get(t) is not None]
    if not compiles:
        raise ValueError("no tile the compiler accepts")

    clean = [t for t in compiles if spills[t] == 0]
    # Spill-free first, then LARGEST BM: a bigger query tile recomputes K and V fewer
    # times, and that redundancy is the price this kernel pays for register headroom.
    pool = sorted(clean or compiles, key=lambda t: (spills[t] > 0, spills[t], -t[0]))
    pool = pool[:MAX_SWEEP]
    fallback = pool[0]

    note = (f"{len(tiles)} viable, {len(compiles)} compile, {len(clean)} spill-free"
            if clean else
            f"{len(tiles)} viable, {len(compiles)} compile, NONE spill-free "
            f"(best {spills[fallback]} B/thread)")
    try:
        import triton.testing as tt
        timed = {}
        for t in pool:
            bm, w = t
            try:
                fn = (lambda t=t: fused_layer(*args, num_heads, head_dim,
                                              head_dim ** -0.5, 1e-5, t[0], t[1]))
                fn()
                timed[t] = min(tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
                               for _ in range(2))
            except Exception:
                continue
        if timed:
            best = min(timed, key=timed.get)
            base = timed.get(fallback)
            if base is None or timed[best] < base * (1.0 - DECISIVE):
                return best, (f"swept at batch {probe_b} ({note}): {best} beat the "
                              f"largest spill-free tile {fallback} decisively")
            return fallback, (f"largest spill-free tile {fallback} at batch {probe_b} "
                              f"({note}), confirmed against {len(timed)} tiles")
    except Exception:
        pass
    finally:
        del args
    return fallback, f"derived {fallback} ({note}; sweep unavailable)"


def applies(seq_len: int, d_model: int, ffn_dim: int, head_dim: int, num_heads: int,
            batch_size: int, props) -> tuple[bool, str]:
    """The dispatch predicate as the candidate asks it. (use_it, human-readable reason).

    `props` is a `torch.cuda.get_device_properties` result: every number consulted is
    measured off the device at run time.
    """
    tiles = viable_tiles(seq_len, d_model, ffn_dim, head_dim, num_heads, batch_size,
                         props.regs_per_multiprocessor, props.multi_processor_count,
                         props.warp_size)
    if tiles:
        bm, w = tiles[0]
        need = register_bytes(seq_len, d_model, ffn_dim, head_dim, bm)
        util = sm_utilization(programs(batch_size, seq_len, bm),
                              props.multi_processor_count)
        return True, (f"fused layer: {len(tiles)} viable tiles, largest {bm}x{w} warps at "
                      f"{need/1024:.0f} KB per program, {util:.0%} of SMs busy")

    # Say WHICH constraint refused, so a decline is diagnosable rather than opaque.
    any_fit = any(
        fits(seq_len, d_model, ffn_dim, head_dim, num_heads, bm, w,
             props.regs_per_multiprocessor, props.warp_size)
        for bm in TILE_ROWS for w in WARP_CHOICES)
    if not any_fit:
        need = register_bytes(seq_len, d_model, ffn_dim, head_dim,
                              min(TILE_ROWS[-1], next_pow2(seq_len)))
        return False, (f"declined: no tile fits -- even {min(TILE_ROWS[-1], next_pow2(seq_len))} "
                       f"query rows need {need/1024:.0f} KB against a "
                       f"{props.regs_per_multiprocessor*4/1024:.0f} KB register file")
    best_util = max(
        sm_utilization(programs(batch_size, seq_len, bm), props.multi_processor_count)
        for bm in TILE_ROWS)
    return False, (f"declined: one program per {TILE_ROWS[-1]}-row query tile puts at most "
                   f"{programs(batch_size, seq_len, TILE_ROWS[-1])} programs on "
                   f"{props.multi_processor_count} SMs, {best_util:.0%} utilised")
