"""Single-tile causal attention with the OUT-PROJECTION, the fp32 widen and the fp32
residual add absorbed into its epilogue. One kernel where the frontier runs three.

WHAT THIS REPLACES
------------------
`v23_single_tile_attn`'s attention half of a layer is

    ctx = single_tile_attention(qkv, ...)          # Triton: writes fp16 [B, S, D] to HBM
    o   = F.linear(ctx, out_w, out_b).float()      # cuBLAS GEMM + a widening pointwise
    x   = x + o                                    # Inductor fuses the widen and the add

Three kernels. The `ctx` tile is already sitting in the attention kernel's REGISTERS when
it finishes; the projection immediately reads it back out of HBM, and the pointwise pass
reads the projection's own fp16 temporary back again. This kernel finishes the contraction
where the data already is.

Per token, bytes moved by the two paths (D = d_model, per layer):

    v23 path   write ctx 2D | read ctx 2D + write o 2D | read o 2D + read x 4D + write y 4D
               = 16D bytes and 3 launches
    fused      read x 4D + write y 4D
               = 8D bytes and 1 launch

Half the traffic of the attention epilogue and two launches of three. (The Q/K/V reads are
common to both and are excluded from both sides.) That is a bigger claim than `g24`'s
`outproj_resid`, which fused the projection with the residual add but still had to read
`ctx` back from HBM: it removed 4D of 14D and one launch of two.

WHY IT NEEDS A DIFFERENT PROGRAM SHAPE
--------------------------------------
v23 runs one program per (batch, HEAD, query block) and each program owns one head's
`[BM, head_dim]` context. The out-projection contracts over the WHOLE of `d_model`, i.e.
over every head at once, so no v23 program holds enough of the row to finish it. Two ways
out, and only one of them is any good:

  * keep v23's grid and `tl.atomic_add` each head's partial `[BM, D]` into the output.
    That is a split-K GEMM with `heads` read-modify-write passes over an fp32 `[M, D]`
    buffer -- 8D bytes per token per head, which is MORE traffic than the thing being
    removed, plus a separate kernel to seed the output with the residual. Rejected on
    arithmetic, not tried.
  * one program per (batch, query block), looping over heads INSIDE. Each head's context
    is projected through its own `[head_dim, D]` slice of the weight and accumulated in
    one fp32 `[BM, D]` register block. No atomics, no seeding pass, one store. This is
    what the kernel does.

That choice has a price and the predicate below is about paying it. The program's register
working set gains the `[BM, D]` fp32 accumulator and a `[head_dim, D]` weight tile, and the
grid loses a factor of `heads`. Both cut occupancy, which for a kernel of this shape is
the only latency hiding there is (see v23's own analysis).

PRECISION -- STRICTLY BETTER, AND IN THE SAME DIRECTION AS EVERY OTHER FUSION HERE
----------------------------------------------------------------------------------
The rounding steps are not the same on the two sides:

    v23    softmax fp32 -> ctx rounded to FP16 (stored) -> GEMM fp32-accumulate
           -> projection rounded to FP16 -> widened to fp32 -> + residual fp32
    fused  softmax fp32 -> ctx rounded to FP16 (registers) -> dot fp32-accumulate
           -> + bias fp32 -> + residual fp32

The fp16 rounding of `ctx` is common to both and is unavoidable: the tensor cores take
fp16 operands and FlashAttention rounds `P` the same way. What the fusion DELETES is the
fp16 rounding of the projection OUTPUT. So this path cannot be less accurate than the one
it replaces, and finding 08's fp32 residual is preserved exactly -- accumulator, bias,
residual load, add and store are all fp32.

THE LAYOUT PREMISE THAT `g24` HAD TO KILL DOES NOT ARISE HERE
-------------------------------------------------------------
Finding 30 established that `F.scaled_dot_product_attention` returns a `[B, S, H, hd]`-
CONTIGUOUS buffer wearing a head-major view, so the `transpose(1, 2).reshape` that feeds
the out-projection is a free view and the gather `g24` was commissioned to absorb never
existed. The same question asked of OUR kernel has an even flatter answer: v23 allocates
`torch.empty((B, S, D))` and writes head `h` at column offset `h * head_dim`, so `ctx` is
plainly token-major contiguous by construction -- there is no view, no transpose and no
stride to check. Nothing here is being won on layout. The win is materialization, and
`bench/probe_outproj_epilogue.py` prints the strides rather than arguing about them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .attn_single_tile import MMA_WIDTH, next_pow2, padded_head_dim


# ------------------------------------------------------------------ the kernel

@triton.jit
def _attn_outproj(
    QKV,                 # [B, S, 3*DM] fp16, contiguous in the last axis
    WT,                  # [DM, DM] fp16 -- out_proj.weight TRANSPOSED (contract axis 0)
    BIAS,                # [DM] fp16
    RES,                 # [B*S, DM] fp32   the residual
    MASK,                # [B, S] bool, read only when HAS_MASK
    OUT,                 # [B*S, DM] fp32
    stride_qkv_b, stride_qkv_s,
    stride_res_m, stride_out_m, stride_mask_b,
    scale,
    S: tl.constexpr,     # sequence length
    H: tl.constexpr,     # heads
    DH: tl.constexpr,    # true head_dim
    DP: tl.constexpr,    # head_dim padded up to the MMA width, power of two
    DM: tl.constexpr,    # heads * head_dim == d_model, power of two
    BM: tl.constexpr,    # query rows per program
    BN: tl.constexpr,    # key columns == next_pow2(S); ONE tile, no K/V loop
    HAS_MASK: tl.constexpr,
):
    m_block = tl.program_id(0)
    b = tl.program_id(1)

    rm = m_block * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, DP)
    rk = tl.arange(0, DM)                 # the out-projection's output columns

    keep_m = rm < S
    keep_n = rn < S
    keep_d = rd < DH                      # False lanes load 0.0 -> exact zero padding

    base = QKV + b * stride_qkv_b
    acc = tl.zeros((BM, DM), dtype=tl.float32)

    # ONE program owns every head of this query block, because the out-projection
    # contracts across all of them. `h` is the split-K index of that contraction, and the
    # partial products are summed in registers instead of in HBM.
    for h in tl.range(0, H):
        head = base + h * DH
        q = tl.load(head + rm[:, None] * stride_qkv_s + rd[None, :],
                    mask=keep_m[:, None] & keep_d[None, :], other=0.0)
        k = tl.load(head + DM + rn[:, None] * stride_qkv_s + rd[None, :],
                    mask=keep_n[:, None] & keep_d[None, :], other=0.0)
        v = tl.load(head + 2 * DM + rn[:, None] * stride_qkv_s + rd[None, :],
                    mask=keep_n[:, None] & keep_d[None, :], other=0.0)

        # v23's single tile, unchanged: whole score matrix in fp32 registers, one
        # `tl.where` for the causal triangle, ONE textbook softmax, no rescale.
        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        valid = (rn[None, :] <= rm[:, None]) & keep_n[None, :]
        s = tl.where(valid, s, float("-inf"))
        row_max = tl.max(s, 1)
        p = tl.exp(s - row_max[:, None])
        row_sum = tl.sum(p, 1)
        ctx = tl.dot(p.to(q.dtype), v, out_dtype=tl.float32) / row_sum[:, None]

        # THE EPILOGUE. `ctx` never leaves the register file. Rows of the weight beyond
        # `DH` are the zero padding's partners and are masked to 0, so the padded lanes
        # contribute exactly zero on both sides of the product.
        w = tl.load(WT + (h * DH + rd)[:, None] * DM + rk[None, :],
                    mask=keep_d[:, None], other=0.0)
        acc += tl.dot(ctx.to(q.dtype), w, out_dtype=tl.float32)

    acc += tl.load(BIAS + rk)[None, :].to(tl.float32)

    if HAS_MASK:
        # Exactly `o.masked_fill(~mask[..., None], 0)` before the residual add, which is
        # what the fallback path does. Invalid tokens keep the residual untouched.
        keep_tok = tl.load(MASK + b * stride_mask_b + rm, mask=keep_m, other=0) != 0
        acc = tl.where(keep_tok[:, None], acc, 0.0)

    # fp32 residual, fp32 add, fp32 store (finding 08). No fp16 temporary anywhere.
    row = b * S + rm
    res = tl.load(RES + row[:, None] * stride_res_m + rk[None, :],
                  mask=keep_m[:, None], other=0.0)
    tl.store(OUT + row[:, None] * stride_out_m + rk[None, :], res + acc,
             mask=keep_m[:, None])


# --------------------------------------------------------------- the working set

MAX_REGS_PER_THREAD = 255       # architectural cap, sm_89
MAX_WARPS = 8


def register_bytes(seq_len: int, head_dim: int, d_model: int, block_m: int) -> int:
    """Per-program on-chip working set.

    Two terms more than `attn_single_tile.register_bytes`, and they are the whole cost of
    the fusion: the fp32 `[BM, d_model]` projection accumulator, which is live across the
    entire head loop, and one head's `[head_dim, d_model]` slice of the weight.
    """
    bn = next_pow2(seq_len)
    dp = padded_head_dim(head_dim)
    scores = block_m * bn * 4
    operands = (block_m + 2 * bn) * dp * 2
    accumulator = block_m * d_model * 4
    weight = dp * d_model * 2
    return scores + operands + accumulator + weight


def register_budget(num_warps: int, regs_per_sm: int, warp_size: int = 32) -> int:
    """Bytes one thread block may hold in registers on the measured device: the SM's whole
    register file, or the per-thread cap times the block's threads, whichever binds."""
    return min(regs_per_sm, num_warps * warp_size * MAX_REGS_PER_THREAD) * 4


def resident_blocks(seq_len: int, head_dim: int, d_model: int, block_m: int,
                    num_warps: int, regs_per_sm: int, max_threads_per_sm: int,
                    warp_size: int = 32) -> int:
    """How many of these blocks an SM can hold at once, by the two binding limits."""
    need = register_bytes(seq_len, head_dim, d_model, block_m)
    if need <= 0:
        return 0
    return min((regs_per_sm * 4) // need,
               max_threads_per_sm // (num_warps * warp_size))


def programs(seq_len: int, batch: int, block_m: int) -> int:
    """CTAs this launch produces. Fusing the epilogue costs a factor of `heads` here --
    v23 emits `ceil(S/BM) * heads * B` and this emits `ceil(S/BM) * B` -- so whether the
    grid still covers the machine is a first-class question, not an afterthought."""
    return -(-seq_len // block_m) * batch


# ----------------------------------------------------------------- the predicate
#
# Three conditions, each a function of shapes and MEASURED device properties only
# (CLAUDE.md rule 2). None of them is a config id or an announced constant.
#
# 1. LEGALITY. The tile must be an `mma.sync.m16n8k16` shape and the working set must fit
#    the register budget.
#
# 2. RESIDENCY. Inherited unchanged from `attn_single_tile`: this kernel's per-program
#    dependent chain is v23's plus a projection, and the fp32 accumulator caps residency
#    directly. v23 MEASURED the sign flip on this card between 2.3 and 4.9 resident
#    blocks per SM and set the threshold at 4. Reusing that number here is CONSERVATIVE
#    rather than fitted: unlike v23 this kernel has a real loop (over heads) whose
#    iterations are independent, so it has intra-program latency hiding v23 has none of,
#    and its true crossover should be LOWER. It is not lowered without a measurement.
#
# 3. SATURATION. The grid shrinks by a factor of `heads`, so it can stop covering the SMs
#    at a shape where v23's does not. `g24` measured this exact crossover for the
#    out-projection GEMM and found it was SM saturation and not a token count -- a wide
#    tile won above 8,192 tokens and lost below 2,048 because it emitted 128 programs
#    against 32 on this card's 66 SMs. The predicate is that rule, restated for this
#    grid: `programs >= props.multi_processor_count`.
#
# Where any of the three fails the candidate falls back to v23's split path, which is the
# frontier and is already fast. Declining costs nothing that was being won.
MIN_RESIDENT_BLOCKS = 4


def fits(seq_len: int, head_dim: int, d_model: int, heads: int, block_m: int,
         num_warps: int, regs_per_sm: int, warp_size: int = 32) -> bool:
    """Legality: an `mma.sync`-shaped tile whose working set fits the register budget."""
    if heads <= 0 or head_dim <= 0 or d_model != heads * head_dim:
        return False
    if d_model < MMA_WIDTH or (d_model & (d_model - 1)):
        return False                      # `tl.arange` needs a power of two >= the MMA width
    if seq_len < MMA_WIDTH or block_m < MMA_WIDTH:
        return False
    if block_m > next_pow2(seq_len):
        return False
    return register_bytes(seq_len, head_dim, d_model, block_m) <= register_budget(
        num_warps, regs_per_sm, warp_size)


def pays(seq_len: int, head_dim: int, d_model: int, batch: int, block_m: int,
         num_warps: int, regs_per_sm: int, max_threads_per_sm: int, sm_count: int,
         warp_size: int = 32) -> bool:
    """Residency AND saturation. See the block above for why both are needed here when
    v23 needed only the first."""
    if programs(seq_len, batch, block_m) < sm_count:
        return False
    return resident_blocks(seq_len, head_dim, d_model, block_m, num_warps, regs_per_sm,
                           max_threads_per_sm, warp_size) >= MIN_RESIDENT_BLOCKS


# --------------------------------------------------------------------- tiling

TILE_ROWS = 64                  # v23's swept starting point; narrowed by the budget below
TARGET_REGS_PER_THREAD = 128    # half the sm_89 architectural cap of 255


def _warps_for(seq_len: int, head_dim: int, d_model: int, block_m: int,
               warp_size: int = 32) -> int:
    """Fewest warps (power of two) that keep the working set near TARGET_REGS_PER_THREAD.
    Derived exactly as v23 derives it, over the larger working set."""
    need_regs = register_bytes(seq_len, head_dim, d_model, block_m) // 4
    w = 2
    while w < MAX_WARPS and need_regs > w * warp_size * TARGET_REGS_PER_THREAD:
        w *= 2
    return w


SWEEP_TILES: tuple[tuple[int, int, int], ...] = (
    (16, 2, 1), (16, 4, 1), (16, 8, 1),
    (32, 2, 1), (32, 4, 1), (32, 8, 1),
    (64, 2, 1), (64, 4, 1), (64, 8, 1),
    (128, 4, 1), (128, 8, 1),
)

DECISIVE = 0.10                 # v23's margin: below it the CUDA event timer cannot rank


def choose_tile(seq_len: int, head_dim: int, d_model: int, heads: int, batch: int,
                regs_per_sm: int, max_threads_per_sm: int, sm_count: int,
                warp_size: int = 32) -> tuple[int, int, int] | None:
    """(block_m, num_warps, num_stages), or None if no tile both fits and pays."""
    bm = min(TILE_ROWS, next_pow2(seq_len))
    while bm >= MMA_WIDTH:
        w = _warps_for(seq_len, head_dim, d_model, bm, warp_size)
        if (fits(seq_len, head_dim, d_model, heads, bm, w, regs_per_sm, warp_size)
                and pays(seq_len, head_dim, d_model, batch, bm, w, regs_per_sm,
                         max_threads_per_sm, sm_count, warp_size)):
            return (bm, w, 1)
        bm //= 2
    return None


def viable_tiles(seq_len: int, head_dim: int, d_model: int, heads: int, batch: int,
                 regs_per_sm: int, max_threads_per_sm: int, sm_count: int,
                 warp_size: int = 32) -> list[tuple[int, int, int]]:
    """Every (block_m, num_warps, num_stages) that both fits and pays on this device."""
    out = []
    for bm, w, st in SWEEP_TILES:
        if not fits(seq_len, head_dim, d_model, heads, bm, w, regs_per_sm, warp_size):
            continue
        if not pays(seq_len, head_dim, d_model, batch, bm, w, regs_per_sm,
                    max_threads_per_sm, sm_count, warp_size):
            continue
        out.append((bm, w, st))
    return out


def autotune_tile(seq_len: int, head_dim: int, heads: int, batch: int,
                  device="cuda") -> tuple[tuple[int, int, int], str]:
    """Pick the tile by TIMING the viable ones against EACH OTHER, at prime time.

    The v20 lesson: a guessed tile lost at 0.88x where a swept one won at 1.163x. What is
    deliberately NOT done here is timing this kernel against the path it replaces -- L41
    and L33: an op-level probe of `attn + F.linear + add` called separately in eager
    measures an isolation the real candidate never runs, because Inductor fuses the widen
    and the add into a neighbouring kernel. A like-for-like comparison BETWEEN tiles of
    the same kernel has no such bias; a gate built on the biased one would.

    The probe batch is capped from the measured SM count for the reason v23 caps it: per
    program work does not depend on batch once the grid fills the machine.
    """
    props = torch.cuda.get_device_properties(device)
    d_model = heads * head_dim
    tiles = viable_tiles(seq_len, head_dim, d_model, heads, batch,
                         props.regs_per_multiprocessor,
                         props.max_threads_per_multi_processor,
                         props.multi_processor_count, props.warp_size)
    if not tiles:
        raise ValueError("no viable tile")
    fallback = choose_tile(seq_len, head_dim, d_model, heads, batch,
                           props.regs_per_multiprocessor,
                           props.max_threads_per_multi_processor,
                           props.multi_processor_count, props.warp_size)
    try:
        import triton.testing as tt
        probe_b = max(1, min(batch, 4 * props.multi_processor_count))
        qkv = torch.randn(probe_b, seq_len, 3 * d_model, device=device,
                          dtype=torch.float16)
        wt = torch.randn(d_model, d_model, device=device, dtype=torch.float16) * 0.05
        bias = torch.zeros(d_model, device=device, dtype=torch.float16)
        res = torch.randn(probe_b * seq_len, d_model, device=device, dtype=torch.float32)
        scale = head_dim ** -0.5
        timed = {}
        for bm, w, st in tiles:
            try:
                fn = (lambda bm=bm, w=w, st=st:
                      attn_outproj(qkv, res, wt, bias, None, heads, head_dim, scale,
                                   bm, w, st))
                fn()
                timed[(bm, w, st)] = min(
                    tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
                    for _ in range(2))
            except Exception:
                continue
        del qkv, wt, bias, res
        if timed:
            best, best_ms = min(timed.items(), key=lambda kv: kv[1])
            base_ms = timed.get(fallback)
            # THE DERIVED TILE HOLDS THE GROUND unless something beats it DECISIVELY --
            # v23's rule, for v23's reason: a candidate whose own tile varies run to run
            # adds that variance to every measurement taken of it (L29).
            if base_ms is None or best_ms < base_ms * (1.0 - DECISIVE):
                return best, (f"autotuned over {len(tiles)} tiles at batch {probe_b}: "
                              f"{best} beat the derived tile decisively")
            return fallback, (f"derived tile {fallback}, confirmed against "
                              f"{len(tiles)} tiles at batch {probe_b}")
    except Exception:
        pass
    return fallback, "derived (autotune unavailable)"


def applies(seq_len: int, head_dim: int, heads: int, batch: int,
            props) -> tuple[bool, str]:
    """The dispatch predicate as the candidate asks it. (use_it, human-readable reason).

    `props` is a `torch.cuda.get_device_properties` result -- every number consulted is
    measured off the device at run time, none is a config id or an announced shape.
    """
    d_model = heads * head_dim
    tile = choose_tile(seq_len, head_dim, d_model, heads, batch,
                       props.regs_per_multiprocessor,
                       props.max_threads_per_multi_processor,
                       props.multi_processor_count, props.warp_size)
    if tile is not None:
        bm, w, _ = tile
        n = resident_blocks(seq_len, head_dim, d_model, bm, w,
                            props.regs_per_multiprocessor,
                            props.max_threads_per_multi_processor, props.warp_size)
        return True, (f"attention+out-projection fused: {bm}x{w} warps, {n} blocks "
                      f"resident per SM, {programs(seq_len, batch, bm)} programs on "
                      f"{props.multi_processor_count} SMs")

    # Say WHICH of the three conditions refused, so a declined path is never mistaken for
    # an untested one (the v14 lesson). `choose_tile` narrows block_m until it runs out,
    # so the informative failure is the one at the NARROWEST tile that was still legal --
    # reporting the first one instead names a condition a narrower tile would have met.
    last = None
    bm = min(TILE_ROWS, next_pow2(seq_len))
    while bm >= MMA_WIDTH:
        w = _warps_for(seq_len, head_dim, d_model, bm, props.warp_size)
        if fits(seq_len, head_dim, d_model, heads, bm, w,
                props.regs_per_multiprocessor, props.warp_size):
            last = (bm, w)
        bm //= 2
    if last is None:
        return False, (f"declined: no legal tile for seq_len={seq_len} "
                       f"head_dim={head_dim} d_model={d_model} on this register file")
    bm, w = last
    n = resident_blocks(seq_len, head_dim, d_model, bm, w,
                        props.regs_per_multiprocessor,
                        props.max_threads_per_multi_processor, props.warp_size)
    if n < MIN_RESIDENT_BLOCKS:
        need = register_bytes(seq_len, head_dim, d_model, bm)
        return False, (
            f"declined: {need/1024:.0f} KB per program at block_m={bm} leaves {n} "
            f"blocks resident on a "
            f"{props.regs_per_multiprocessor*4/1024:.0f} KB register file, under the "
            f"measured crossover of {MIN_RESIDENT_BLOCKS}")
    return False, (
        f"declined: {programs(seq_len, batch, bm)} programs at block_m={bm} do not "
        f"cover {props.multi_processor_count} SMs; the split path emits "
        f"{programs(seq_len, batch, bm) * heads} and does")


# --------------------------------------------------------------------- launcher

def attn_outproj(qkv: torch.Tensor, res: torch.Tensor, w_t: torch.Tensor,
                 bias: torch.Tensor, mask: torch.Tensor | None,
                 heads: int, head_dim: int, scale: float,
                 block_m: int, num_warps: int, num_stages: int = 1) -> torch.Tensor:
    """`res + (causal_sdpa(qkv) @ w_t + bias).float()`, masked, in ONE launch.

    `qkv` is the fused `[B, S, 3*d_model]` projection buffer; `res` is the fp32
    `[B*S, d_model]` residual; `w_t` is `out_proj.weight` already transposed to
    `[d_model, d_model]` because the kernel contracts over the leading axis; `mask` is
    `[B, S]` bool or None. Returns fp32 `[B*S, d_model]`.

    Deliberately free of any data-dependent Python: no `.item()`, no `bool(tensor)`, no
    `is_contiguous()` branch. Everything that decides the launch was decided at prime time
    (`_decide_outproj`), so nothing here can graph-break the traced region or drop a
    compiled frame back to eager.
    """
    b, s, three_dm = qkv.shape
    dm = heads * head_dim
    assert three_dm == 3 * dm, f"expected [B, S, 3*{dm}], got {tuple(qkv.shape)}"
    out = torch.empty((b * s, dm), device=qkv.device, dtype=torch.float32)
    has_mask = mask is not None
    mask_arg = mask if has_mask else res            # never dereferenced when HAS_MASK
    _attn_outproj[(triton.cdiv(s, block_m), b)](
        qkv, w_t, bias, res, mask_arg, out,
        qkv.stride(0), qkv.stride(1),
        res.stride(0), out.stride(0),
        mask.stride(0) if has_mask else 0,
        scale,
        S=s, H=heads, DH=head_dim, DP=padded_head_dim(head_dim), DM=dm,
        BM=block_m, BN=next_pow2(s), HAS_MASK=has_mask,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
