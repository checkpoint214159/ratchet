"""Shape-aware kernel dispatch — the germane deliverable.

The problem statement is explicit that this is a dispatch problem: "all combinations of
input shapes will be told to the participants" and you "can decide different
implementations for different shapes by adding shape checks." One kernel does not win the
whole matrix, so the submission is a `select(shape) -> recipe` that picks the
implementation per shape.

Two design commitments, both taken from what the field rewards:
  * Decisions are **calibrated from device properties**, not hardcoded shape thresholds.
    The same arithmetic intensity is launch-bound on one GPU and compute-bound on the next,
    so a constant that is right here is wrong there. `select` reads a DeviceProfile
    (`ledger/device.gb10.json`) for the smem budget, launch overhead and memory.
  * Every heavy op the recipe selects is a hand-written kernel from this repo
    (`flash_attention`, `linear_tf32`), fp16 with fp32 accumulation — inside the
    rel<0.02 / abs<0.002 correctness bound the statement fixes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """One announced shape. Every row of the matrix is causal, ffn_dim == d_model."""
    id: int
    batch_size: int
    d_model: int
    heads: int
    seq_len: int
    layers: int
    ffn_dim: int
    causal: bool = True

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    @property
    def tokens(self) -> int:
        return self.batch_size * self.seq_len


# The announced TikTok TechJam 2026 matrix (13 runnable; config 14 seq=100000 OOMs the
# fp32 reference). Transcribed from the problem statement.
MATRIX: tuple[Config, ...] = (
    Config(1, 64, 128, 4, 128, 4, 128),
    Config(2, 1, 128, 4, 128, 4, 128),
    Config(3, 4, 128, 4, 128, 4, 128),
    Config(4, 16, 128, 4, 128, 4, 128),
    Config(5, 128, 128, 4, 128, 4, 128),
    Config(6, 10000, 128, 4, 128, 4, 128),
    Config(7, 64, 32, 4, 128, 4, 32),
    Config(8, 64, 1024, 4, 128, 4, 1024),
    Config(9, 64, 128, 1, 128, 4, 128),
    Config(10, 64, 128, 2, 128, 4, 128),
    Config(11, 64, 128, 16, 128, 4, 128),
    Config(12, 64, 128, 4, 32, 4, 128),
    Config(13, 64, 128, 4, 1024, 4, 128),
    Config(14, 32, 1024, 16, 100000, 2, 1024),
)


@dataclass(frozen=True)
class Recipe:
    dtype: str            # compute dtype for the GEMMs/attention (fp32 accumulate always)
    use_graph: bool       # capture the forward in a CUDA graph (kills launch overhead)
    flash_bm: int         # flash query-block; chosen so the fp16 tiles fit smem
    flash_bn: int
    reason: str


def _flash_tiles(head_dim: int, smem_optin: int) -> tuple[int, int]:
    """Query/key block for the flash kernel, sized so the fp16 working set fits the device's
    shared-memory budget. Wider heads force smaller blocks. `head_dim<16` is padded to 16
    inside the kernel (tensor-core K>=16), so it shares the small-head tiling."""
    d = 16 if head_dim < 16 else head_dim
    working = smem_optin  # bytes available per block
    # q,k,v tiles are ~ (BM+2*BN)*d*2 bytes (fp16) plus a BM*d fp32 accumulator.
    for bm, bn in ((64, 64), (32, 32), (16, 32), (16, 16)):
        need = (bm * d * 4) + (bm + 2 * bn) * d * 2
        if need <= working:
            return bm, bn
    return 16, 16


def _launch_bound(cfg: Config, prof) -> float:
    """Fraction of a graph-free forward spent in fixed launch overhead. High => a CUDA graph
    (one replay instead of ~7 launches/layer) is the dominant win; low => compute-bound and
    the graph is neutral. Derived from the measured per-launch overhead, not a shape constant."""
    n_launch = cfg.layers * 7 + 1
    fixed_s = n_launch * prof.launch_overhead_us * 1e-6
    # rough compute estimate: attention (causal ~1/2) + qkv/out proj + FFN, fp16 tensor cores.
    d, s, b, h = cfg.d_model, cfg.seq_len, cfg.batch_size, cfg.head_dim
    attn = b * cfg.heads * s * s * h * 2  # QK^T + PV, halved for causal below
    proj = b * s * (3 * d * d + d * d)
    ffn = b * s * (2 * d * cfg.ffn_dim)
    flops = cfg.layers * (0.5 * 2 * attn + 2 * (proj + ffn))
    eff_tflops = 30.0e12  # nominal fp16 effective throughput on GB10; rough, for a ratio only
    compute_s = flops / eff_tflops
    return fixed_s / (fixed_s + compute_s)


def select(cfg: Config, prof) -> Recipe:
    """Pick the per-shape recipe. Device-calibrated; every op it selects is a repo kernel."""
    bm, bn = _flash_tiles(cfg.head_dim, int(getattr(prof, "smem_per_block_optin", 101376)))

    # CUDA graph pays off when launches dominate (small batch / short seq) and the static
    # capture buffers are memory-safe. Both are read from the device, not hardcoded.
    lb = _launch_bound(cfg, prof)
    static_bytes = 2 * cfg.tokens * cfg.d_model * 4
    mem_ok = static_bytes < 0.25 * int(getattr(prof, "total_memory", 1 << 62))
    use_graph = mem_ok and lb > 0.10

    if not mem_ok:
        reason = f"launch_frac={lb:.2f} but static buffers too large -> stream, no graph"
    elif use_graph:
        reason = f"launch_frac={lb:.2f} -> graph capture (launch-bound)"
    else:
        reason = f"launch_frac={lb:.2f} -> compute-bound, graph neutral (kept off)"

    return Recipe(dtype="float16", use_graph=use_graph, flash_bm=bm, flash_bn=bn, reason=reason)
