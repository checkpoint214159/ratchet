"""A Triton GEMM for the narrow-K projections, with the GELU in the epilogue.

WHY THE VENDOR KERNEL IS NOT THE END OF THE STORY AT K = 128
------------------------------------------------------------
Censused on the real frontier (`v34_launch_bound`), config 9, per-forward device time
inside the replayed graph -- 220.0 us total, 35 kernel nodes:

    ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn   x12   68.03 us  30.9%
    ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_tn          x4    53.09 us  24.1%
    pytorch_flash::flash_fwd_kernel<...>                         x4    35.72 us  16.2%
    triton_per_fused__to_copy_add_native_layer_norm_*            x9    43.45 us  19.8%
    triton_poi_fused_gelu_2                                      x4    12.14 us   5.5%
    Memcpy DtoD                                                  x2     7.61 us   3.5%

**The sixteen projection GEMMs are 55.0% of the device time and they are not at the
roofline.** The twelve `[8192,128] x [128,128]` calls deliver 3.22 GFLOP in 68.03 us =
47.3 TFLOP/s, **53.6% of this card's measured 88.2 BF16-TFLOP/s peak**; the four QKV
`[8192,128] x [128,384]` reach 60.6 TFLOP/s (68.8%).

Three hardware reasons, all readable off the kernel name and the shape:

1. **`s1688` is `mma.sync.m16n8k8`.** sm_89 also issues `m16n8k16` -- twice the K-depth
   per instruction for fp16 operands -- and that is what `tl.dot` emits. cuBLAS's
   heuristic picks the k8 kernel on all twelve narrow-K calls. It does NOT do this at
   d_model 1024 (config 8), where it selects `cutlass_80_tensorop_f16_s16816gemm` and
   reaches 100.4% of measured peak. The bad selection is specific to narrow K.
2. **K = d_model = 128 is four BK=32 steps.** The main loop is too short to amortize the
   prologue and there is no second k-iteration to pipeline against.
3. **128x128 tiles over [8192,128] give 64 CTAs against 66 SMs** -- one wave with nothing
   to overlap the tail against (the structural problem finding 31 diagnosed for the
   loop-free attention kernel).

AND THE GELU IS A SEPARATE KERNEL PURELY BECAUSE OF THE EPILOGUE VACUUM
-----------------------------------------------------------------------
GELU sits between two cuBLAS GEMMs, so it has no pointwise or reduction neighbour for
Inductor to fuse it into, and cuBLAS takes no epilogue -- finding 22's 68-SM veto leaves
every GEMM in this stack on cuBLAS/CUTLASS. Finding 39 already corrected finding 22 on
exactly this point: that veto reading "was right about LayerNorm and wrong about GELU;
GELU has no LayerNorm to hide in and is its own kernel on every layer of every config."
It costs 12.14 us of device time AND four of the 35 graph nodes (~3.2 us at finding 33's
0.798 us/node).

THE PREDICATE IS A MEASUREMENT, NOT A SHAPE LITERAL
----------------------------------------------------
`plan` times the vendor call and the swept Triton tile ONCE, at prime time, on the actual
operands, and keeps the vendor unless Triton beats it by more than `DECISIVE`. No config
id, no announced shape constant (CLAUDE.md rule 2): a card where cuBLAS picks the right
kernel declines everywhere without being retuned, and config 8 -- where the vendor is
already at 100.4% of peak -- declines on this card. This is `v23_single_tile_attn`'s
already-shipped discipline (finding 31): the tile is swept, and the vendor holds the
ground unless something beats it decisively.

PRECISION
---------
This replaces one fp16-operand / fp32-accumulate GEMM with another: `tl.dot` over the
same fp16 operands into an fp32 accumulator, bias added in fp32, result rounded to fp16
exactly once -- the same single rounding `F.linear` performs. It is an IDENTITY argument,
not a tolerance argument.

The one semantic change is the GELU epilogue, and it is strictly BETTER than the path it
replaces. The split path computes `h = linear(...)` in fp32, rounds it to fp16, writes it
to HBM, reads it back and applies GELU to the fp16 value. The epilogue applies GELU to the
fp32 accumulator and rounds once. That is finding 35's argument for `outproj_resid`, and
it is the same expression `ffn_fused._ffn_block_normed` already uses across 611 ledger
rows: **the exact erf form**, `h * 0.5 * (1 + erf(h/sqrt(2)))`, matching the reference's
`approximate="none"`. The tanh approximation differs by up to ~1e-3, half the entire 2e-3
budget spent on an approximation nobody asked for. Do not substitute it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# The L2 swizzle width: how many M-tiles a program group walks before advancing in N.
GROUP_M = 8


# ------------------------------------------------------------------------ the kernel

@triton.jit
def _proj_gemm(
    A,                      # [M, K] fp16, row-major
    B,                      # [K, N] fp16, row-major (the weight, PRE-TRANSPOSED)
    BIAS,                   # [N]    fp16, or A again when HAS_BIAS is False
    C,                      # [M, N] fp16 or fp32, row-major
    M,
    N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    GELU: tl.constexpr, HAS_BIAS: tl.constexpr,
):
    # L2-friendly ordering: walk GROUP_M rows of tiles before advancing in N, so the
    # column panel of the weight matrix that every program in the group reads stays hot.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n: tl.constexpr = N // BN
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    keep = rm < M

    a_ptrs = A + rm[:, None] * K + rk[None, :]
    b_ptrs = B + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in tl.range(0, K // BK):
        # N and K are exact multiples of BN and BK (enforced by `legal`), so only the M
        # edge needs a mask.
        a = tl.load(a_ptrs, mask=keep[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK
        b_ptrs += BK * N

    if HAS_BIAS:
        acc += tl.load(BIAS + rn)[None, :].to(tl.float32)
    if GELU:
        # The exact erf form, applied to the fp32 accumulator BEFORE any downcast.
        acc = acc * 0.5 * (1.0 + tl.erf(acc * 0.70710678118654752440))

    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty),
             mask=keep[:, None])


# ---------------------------------------------------------------------- the launcher

def proj_gemm(a: torch.Tensor, bt: torch.Tensor, bias: torch.Tensor | None,
              tile: tuple[int, int, int, int, int], gelu: bool = False,
              out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """`gelu(a @ bt + bias)` in one launch. `bt` is [K, N] -- already transposed relative
    to `nn.Linear`'s [out, in] layout, exactly as `ffn_fused` wants its weights.

    `tile` is (BM, BN, BK, num_warps, num_stages), resolved to Python ints by `plan`
    BEFORE anything traces this call.
    """
    m, k = a.shape
    kb, n = bt.shape
    assert kb == k, f"inner dims disagree: {tuple(a.shape)} @ {tuple(bt.shape)}"
    bm, bn, bk, warps, stages = tile
    out = torch.empty((m, n), device=a.device,
                      dtype=out_dtype if out_dtype is not None else a.dtype)
    grid = (triton.cdiv(m, bm) * (n // bn),)
    _proj_gemm[grid](
        a, bt, bias if bias is not None else a, out, m,
        N=n, K=k, BM=bm, BN=bn, BK=bk, GROUP_M=GROUP_M,
        GELU=gelu, HAS_BIAS=bias is not None,
        num_warps=warps, num_stages=stages,
    )
    return out


# ------------------------------------------------------------------ the swept tiles

# Every tile the prime-time sweep considers. A mechanism argument cannot pick one: v20
# lost at 0.88x on a guessed tile and won at 1.163x on a swept one, and the g28
# megakernel measured 1.52x spill-free against 2.28x SLOWER once it spilled. So the tile
# is TIMED, and `n_spills` is recorded alongside it so a spilling winner is visible
# rather than mysterious (ncu is unavailable under WSL2 -- it denies GPU counters -- but
# every CompiledKernel carries n_regs/n_spills/shared for free).
SWEEP_TILES: tuple[tuple[int, int, int, int, int], ...] = (
    # BM,   BN,  BK, warps, stages
    (16,    64,  32,  2, 4),
    (32,    64,  32,  2, 4),
    (32,   128,  32,  4, 4),
    (64,    32,  32,  2, 4),
    (64,    64,  32,  4, 4),
    (64,    64,  64,  4, 3),
    (64,   128,  32,  4, 4),
    (64,   128,  64,  4, 3),
    (64,   128, 128,  8, 2),
    (128,   32,  32,  4, 4),
    (128,   64,  32,  4, 4),
    (128,   64,  64,  4, 3),
    (128,  128,  32,  8, 4),
    (128,  128,  64,  8, 3),
    (128,  128, 128,  8, 2),
    (256,   64,  32,  8, 3),
    (256,  128,  32,  8, 3),
    (256,  128,  64,  8, 2),
)

# sm_89's tensor-core instruction is mma.sync.m16n8k16, so `tl.dot` needs every dimension
# at 16 or more. Below that the kernel is not merely slow, it does not compile.
MMA_MIN = 16

# The accumulator lives in registers: BM*BN fp32 values spread over num_warps*32 threads.
# This card has 65536 32-bit registers per SM and a hard 255 per thread. A tile whose
# accumulator ALONE needs more than this many registers per thread is dropped before it
# is compiled, because it will spill -- and a spilling tile is not a candidate, it is a
# measurement of the spill.
MAX_ACC_REGS_PER_THREAD = 168


def legal(m: int, k: int, n: int, tile: tuple[int, int, int, int, int]) -> bool:
    """Is this tile runnable for this shape? Shapes and the MMA width only -- no ids."""
    bm, bn, bk, warps, _stages = tile
    if bm < MMA_MIN or bn < MMA_MIN or bk < MMA_MIN:
        return False
    if n % bn or k % bk:
        return False                      # the kernel masks only the M edge
    if m >= MMA_MIN and bm > 2 * m:
        return False                      # more than half the tile would be padding
    return (bm * bn) / (warps * 32) <= MAX_ACC_REGS_PER_THREAD


def viable_tiles(m: int, k: int, n: int) -> list[tuple[int, int, int, int, int]]:
    return [t for t in SWEEP_TILES if legal(m, k, n, t)]


def kernel_stats(a, bt, bias, tile, gelu) -> dict:
    """n_regs / n_spills / shared for a tile, straight off the CompiledKernel.

    Free, always available, and the only occupancy instrument we have on WSL2.
    """
    m, k = a.shape
    _kb, n = bt.shape
    bm, bn, bk, warps, stages = tile
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, bm) * (n // bn),)
    binary = _proj_gemm[grid](
        a, bt, bias if bias is not None else a, out, m,
        N=n, K=k, BM=bm, BN=bn, BK=bk, GROUP_M=GROUP_M,
        GELU=gelu, HAS_BIAS=bias is not None,
        num_warps=warps, num_stages=stages,
    )
    meta = getattr(binary, "metadata", None)
    return {"n_regs": getattr(binary, "n_regs", None),
            "n_spills": getattr(binary, "n_spills", None),
            "shared": getattr(binary, "shared", getattr(meta, "shared", None))}


# ------------------------------------------------------------------- the predicate

# The VENDOR HOLDS THE GROUND unless Triton beats it by more than this. These kernels run
# in 6-20 us and the CUDA event timer resolves ~1 us, so inside the margin the ranking is
# noise -- and a candidate whose own kernel selection varies run to run injects that noise
# into every measurement taken of it (L29). Deliberately the same 10% v23 and v34 use.
DECISIVE = 0.10

# The probe's M is CAPPED, for the reason v23's `autotune_tile` caps its probe batch: per
# -program work stops depending on M once the grid is several waves deep, so timing config
# 6's 1.28M rows would allocate gigabytes to learn what a few thousand already say -- and
# config 14's 3.2M rows at d_model 1024 would simply OOM the tuner. The cap is derived
# from the measured SM count and the widest tile in the sweep, not from a config id.
PROBE_WAVES = 4
MAX_TILE_M = max(t[0] for t in SWEEP_TILES)
# Never let the probe's own buffers claim more than this share of free device memory.
PROBE_MEM_FRACTION = 0.15


def probe_rows(m: int, k: int, n: int, device) -> int:
    """How many rows to time on. Never more than the real M; never enough to OOM."""
    props = torch.cuda.get_device_properties(device)
    capped = min(m, PROBE_WAVES * props.multi_processor_count * MAX_TILE_M)
    try:
        free, _total = torch.cuda.mem_get_info(device)
    except Exception:
        return max(MMA_MIN, capped)
    # a [M,K] fp16 + two outputs [M,N] (fp16 vendor, fp16/fp32 triton) is the M-scaling
    # part; the weights are M-independent.
    per_row = 2 * k + 6 * n
    affordable = int(free * PROBE_MEM_FRACTION) // max(1, per_row)
    return max(MMA_MIN, min(capped, max(MMA_MIN, affordable)))


def plan(m: int, k: int, n: int, gelu: bool, device, dtype=torch.float16,
         out_dtype: torch.dtype | None = None):
    """Should this projection go to Triton, and on which tile?

    Returns `(tile_or_None, why, stats)`. `tile_or_None` is None whenever the vendor
    wins, which is the entire point: the decision is a MEASUREMENT of both paths on the
    real operands, not a claim about shapes. Nothing here can see a config id.
    """
    import torch.nn.functional as F

    stats: dict = {}
    if not viable_tiles(m, k, n):
        return None, f"vendor: no legal tile for [{m},{k}]x[{k},{n}]", stats
    a = bt = bias = w = None
    try:
        import triton.testing as tt
        pm = probe_rows(m, k, n, device)
        tiles = viable_tiles(pm, k, n)
        if not tiles:
            return None, f"vendor: no legal tile at probe M={pm}", stats
        note = "" if pm == m else f" [probe M={pm} of {m}]"
        a = torch.randn(pm, k, device=device, dtype=dtype)
        bt = torch.randn(k, n, device=device, dtype=dtype)
        bias = torch.randn(n, device=device, dtype=dtype)
        w = bt.t().contiguous()                      # nn.Linear's [out, in] layout

        def vendor():
            y = F.linear(a, w, bias)
            return F.gelu(y, approximate="none") if gelu else y

        def bench(fn):
            fn()
            return min(tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
                       for _ in range(2))

        # THE VENDOR ARM IS TIMED TWICE, STRADDLING THE SWEEP, AND THE MIN IS KEPT.
        # Measured while building this: the FIRST `plan` call in a process read
        # `F.linear([8192,128]x[128,384])` at 306.18 us, where a clean process reads
        # 21.50 us six times running -- cuBLASLt's first-use heuristic and workspace
        # setup landing inside the timing window. Taken at face value that is a 17.6x
        # "win" for Triton on a shape where the honest ratio is 1.24x, and the predicate
        # would have been deciding on garbage. Finding 42's lesson applied to our own
        # tuner: when two arms are timed in sequence, the order is part of the
        # measurement, so straddle it.
        vendor_pre = bench(vendor)

        timed: dict = {}
        for t in tiles:
            try:
                timed[t] = bench(lambda t=t: proj_gemm(a, bt, bias, t, gelu, out_dtype))
            except Exception:
                continue                              # illegal on this card; skip it
        if not timed:
            return None, "vendor: no tile compiled", stats

        vendor_ms = min(vendor_pre, bench(vendor))
        best = min(timed, key=timed.get)
        best_ms = min(timed[best],
                      bench(lambda: proj_gemm(a, bt, bias, best, gelu, out_dtype)))
        try:
            stats = kernel_stats(a, bt, bias, best, gelu)
        except Exception:
            stats = {}
        ratio = vendor_ms / best_ms if best_ms > 0 else 0.0
        if best_ms < vendor_ms * (1.0 - DECISIVE):
            return best, (f"triton {best} at {best_ms*1e3:.2f} us beats the vendor's "
                          f"{vendor_ms*1e3:.2f} us decisively ({ratio:.3f}x over "
                          f"{len(timed)} tiles){note}; spills={stats.get('n_spills')} "
                          f"regs={stats.get('n_regs')}"), stats
        return None, (f"vendor: {vendor_ms*1e3:.2f} us against the best of {len(timed)} "
                      f"tiles at {best_ms*1e3:.2f} us ({ratio:.3f}x, inside the "
                      f"{DECISIVE:.0%} margin){note}"), stats
    except Exception as exc:                          # never fail closed on a tuner
        return None, f"vendor: sweep unavailable ({type(exc).__name__}: {exc})", stats
    finally:
        del a, bt, bias, w
