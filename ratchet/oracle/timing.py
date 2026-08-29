"""Timing backends.

ZONE A -- IMMUTABLE. Do not edit as part of an optimization step.

Five backends behind one interface, because the single most effective defence against
"is this speedup real?" is showing the same number from two independent methods.

The critical facts, restated here because they are the ones people get wrong:

  * triton.testing.do_bench(fn, warmup=25, rep=100) -- those are MILLISECONDS OF BUDGET,
    not iteration counts. Iteration counts are derived from a 5-iteration estimate.

  * do_bench FLUSHES L2 between repetitions. do_bench_cudagraph DOES NOT, and it captures
    n_repeat unrolled calls into one graph so launch overhead is amortized away. The two
    measure different things and must never be compared directly. Subtracting them is
    legitimate and gives you launch overhead (see device.py).

  * A kernel shorter than ~5x the launch overhead is measuring the launch. We flag this
    rather than silently reporting it as a kernel time.

  * L2 must be flushed by WRITING to a buffer larger than L2, before EVERY timed
    repetition. Allocating is not enough. Sizes: A100 40MB, H100 50MB, H200 90MB,
    RTX 4090 72MB, L40S 48MB, Blackwell ~192MB -- so a hardcoded 256MB is marginal on
    recent parts. We size from props.L2_cache_size with a 256MB floor.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch

try:
    import triton.testing as tt
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


# Stop when the standard error of the mean falls below this fraction of the mean.
# 1e-3 gives tight bars on fast kernels; the wallclock caps stop slow ones running away.
_TARGET_REL_SEM = 1e-3
_MIN_RUNS = 5
_MAX_RUNS = 10_000
_MAX_WALL_S = 120.0


@dataclass
class MethodDescriptor:
    """Everything needed to reproduce or invalidate a timing. Goes into the ledger row.

    A speedup without this metadata is not a result. That is not a style rule: half the
    published kernel speedups in 2025-2026 are artifacts of one of these fields.
    """
    method: str
    l2_flushed: bool
    flush_bytes: int
    includes_launch: bool
    warmup_ms: Optional[float] = None
    budget_ms: Optional[float] = None
    warmup_iters: Optional[int] = None
    clocks_locked: bool = False
    locked_sm_clock_mhz: Optional[int] = None
    torch_version: str = ""
    triton_version: str = ""


@dataclass
class TimingStats:
    runs: int
    mean_ns: float
    std_ns: float
    sem_ns: float
    min_ns: float
    max_ns: float
    p50_ns: float
    p20_ns: float
    p80_ns: float
    stopped_because: str
    warning: Optional[str] = None

    @property
    def rel_sem(self) -> float:
        return self.sem_ns / self.mean_ns if self.mean_ns else float("inf")

    def overlaps(self, other: "TimingStats", k: float = 2.0) -> bool:
        """Do the k-sigma confidence intervals overlap?

        Promotion to the dispatch table REQUIRES this to be False. A 3% win with
        overlapping error bars is noise, and treating it as a win is how a search loop
        convinces itself it is making progress while random-walking.
        """
        lo_a, hi_a = self.mean_ns - k * self.sem_ns, self.mean_ns + k * self.sem_ns
        lo_b, hi_b = other.mean_ns - k * other.sem_ns, other.mean_ns + k * other.sem_ns
        return not (hi_a < lo_b or hi_b < lo_a)


def _summarize(samples_ns: list[float], stopped_because: str) -> TimingStats:
    n = len(samples_ns)
    mean = statistics.fmean(samples_ns)
    std = statistics.pstdev(samples_ns) if n > 1 else 0.0
    srt = sorted(samples_ns)

    def q(p: float) -> float:
        if n == 1:
            return srt[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return srt[idx]

    return TimingStats(
        runs=n,
        mean_ns=mean,
        std_ns=std,
        sem_ns=std / (n ** 0.5) if n > 1 else 0.0,
        min_ns=srt[0],
        max_ns=srt[-1],
        p50_ns=q(0.5),
        p20_ns=q(0.2),
        p80_ns=q(0.8),
        stopped_because=stopped_because,
    )


class L2Flusher:
    """Writes a buffer larger than L2 to evict it. Allocation alone does not evict."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        self._buf = torch.empty(nbytes // 4, dtype=torch.int32, device="cuda")

    def flush(self) -> None:
        self._buf.zero_()


# --------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------

def time_cuda_event(fn: Callable[[], None], *, flush_bytes: int,
                    warmup_iters: int = 25) -> tuple[TimingStats, MethodDescriptor]:
    """The default. Explicit control over everything, adaptive stopping on relative SEM.

    Correctness of the timing depends on three things being in the right order and it is
    worth spelling them out:
      1. synchronize BEFORE recording the start event, so prior work is not counted;
      2. flush L2 AFTER that sync and BEFORE the start event, so the flush is not timed;
      3. synchronize AFTER the end event, before reading elapsed_time.
    """
    flusher = L2Flusher(flush_bytes)

    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    t0 = time.perf_counter()
    stopped = "max_runs"

    while len(samples) < _MAX_RUNS:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        flusher.flush()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()

        samples.append(start.elapsed_time(end) * 1e6)  # ms -> ns

        if len(samples) >= _MIN_RUNS:
            s = _summarize(samples, "probe")
            if s.rel_sem < _TARGET_REL_SEM:
                stopped = "rel_sem"
                break
        if time.perf_counter() - t0 > _MAX_WALL_S:
            stopped = "wallclock"
            break

    stats = _summarize(samples, stopped)
    desc = MethodDescriptor(
        method="cuda_event",
        l2_flushed=True,
        flush_bytes=flush_bytes,
        includes_launch=True,
        warmup_iters=warmup_iters,
        torch_version=torch.__version__,
    )
    return stats, desc


def time_do_bench(fn: Callable[[], None], *, flush_bytes: int,
                  warmup_ms: float = 25, budget_ms: float = 100
                  ) -> tuple[TimingStats, MethodDescriptor]:
    """Triton's do_bench. Used as a CROSS-CHECK, not as the primary number.

    warmup and rep are milliseconds of budget. do_bench flushes L2 itself via
    driver.active.clear_cache(), so flush_bytes here is recorded for provenance rather
    than used -- we cannot control Triton's buffer size.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton unavailable")

    times_ms = tt.do_bench(fn, warmup=warmup_ms, rep=budget_ms, return_mode="all")
    if isinstance(times_ms, float):
        times_ms = [times_ms]
    stats = _summarize([t * 1e6 for t in times_ms], "triton_budget")
    desc = MethodDescriptor(
        method="do_bench",
        l2_flushed=True,
        flush_bytes=flush_bytes,
        includes_launch=True,
        warmup_ms=warmup_ms,
        budget_ms=budget_ms,
        torch_version=torch.__version__,
    )
    return stats, desc


def time_cudagraph(fn: Callable[[], None], *, flush_bytes: int, rep_ms: float = 20
                   ) -> tuple[TimingStats, MethodDescriptor]:
    """CUDA-graph timing: NO L2 flush, launch overhead amortized away.

    Deliberately not comparable to the others. Two legitimate uses:
      * subtract from time_cuda_event to obtain launch overhead;
      * measure the regime where the production path will itself be graph-captured.

    Never quote a speedup that compares a graphed candidate to an ungraphed baseline.
    """
    if not _HAS_TRITON:
        raise RuntimeError("triton unavailable")

    times_ms = tt.do_bench_cudagraph(fn, rep=rep_ms, return_mode="all")
    if isinstance(times_ms, float):
        times_ms = [times_ms]
    stats = _summarize([t * 1e6 for t in times_ms], "triton_budget")
    desc = MethodDescriptor(
        method="cudagraph",
        l2_flushed=False,       # <- the important asymmetry
        flush_bytes=0,
        includes_launch=False,  # <- and the other one
        budget_ms=rep_ms,
        torch_version=torch.__version__,
    )
    return stats, desc


def time_host(fn: Callable[[], None], *, flush_bytes: int, iters: int = 100
              ) -> tuple[TimingStats, MethodDescriptor]:
    """Wall-clock timing including all host overhead. Sanity check only.

    If host time and cuda_event time diverge wildly, you have a CPU bottleneck (a sync
    point, a Python hot loop, an allocator stall) and the kernel is not your problem.
    """
    flusher = L2Flusher(flush_bytes)
    for _ in range(25):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        flusher.flush()
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        samples.append(float(time.perf_counter_ns() - t0))

    stats = _summarize(samples, "fixed_iters")
    desc = MethodDescriptor(
        method="host",
        l2_flushed=True,
        flush_bytes=flush_bytes,
        includes_launch=True,
        torch_version=torch.__version__,
    )
    return stats, desc


_BACKENDS = {
    "cuda_event": time_cuda_event,
    "do_bench": time_do_bench,
    "cudagraph": time_cudagraph,
    "host": time_host,
}


def get_timer(method: str = "cuda_event") -> Callable:
    if method not in _BACKENDS:
        raise KeyError(f"unknown timing method {method!r}; have {sorted(_BACKENDS)}")
    return _BACKENDS[method]


def annotate_launch_domination(stats: TimingStats, launch_overhead_us: float) -> TimingStats:
    """Flag a measurement that is really measuring the launch.

    Below ~5x the launch overhead you are timing the driver, not the kernel. This is not
    a reason to discard the measurement -- the launch-bound regime is a real dispatch
    branch and this is exactly the signal that detects it -- but it must never be
    reported as a kernel time without the flag.
    """
    if stats.mean_ns < 5.0 * launch_overhead_us * 1e3:
        stats.warning = "launch_dominated"
    return stats


def cross_check(fn: Callable[[], None], *, flush_bytes: int,
                primary: str = "cuda_event", secondary: str = "do_bench",
                tolerance_pct: float = 10.0) -> dict:
    """Time with two independent backends and report whether they agree.

    Disagreement beyond `tolerance_pct` on a kernel long enough not to be launch-bound
    means one of the two measurements is wrong. Investigate before trusting either.
    """
    s1, d1 = get_timer(primary)(fn, flush_bytes=flush_bytes)
    try:
        s2, d2 = get_timer(secondary)(fn, flush_bytes=flush_bytes)
    except Exception as exc:
        return {"primary": asdict(s1), "primary_method": asdict(d1),
                "secondary": None, "error": str(exc), "agree": None}

    disagree_pct = abs(s1.mean_ns - s2.mean_ns) / max(s1.mean_ns, 1.0) * 100.0
    return {
        "primary": asdict(s1), "primary_method": asdict(d1),
        "secondary": asdict(s2), "secondary_method": asdict(d2),
        "disagree_pct": disagree_pct,
        "agree": disagree_pct <= tolerance_pct,
    }
