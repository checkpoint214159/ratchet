"""E6: evidence-driven dispatch with an explicit untuned fallback.

The repo's contract (docs/optimization-principles.md, specs/04-dispatch.md): a candidate is
deployed for a workload only if *measured* evidence shows it beats the baseline on that
workload; otherwise the baseline is the fallback. Dispatch never guesses -- it reads the
recorded speedup and, when nothing clears 1.0x with margin, returns the baseline.

On GB10, measured by the *authoritative evaluator* (the scored methodology; the do_bench
harness proved unreliable for fast candidates because GB10's unlockable clock idles at
266 MHz and short timing loops under-clock the fp32 baseline):

    hand-written Triton candidates:  flash 0.96x | tf32x3 0.94x | qkv 0.97x | full 0.97x
    library tensor-core path:        cublastf32  1.16x (seq128) .. 1.66x (seq512)  <-- wins

The hand-written GEMMs could not beat cuBLAS; the real lever is TF32 tensor cores, which
the baseline forgoes (true fp32) and which stay inside the 0.002 gate. Dispatch therefore
selects `cublastf32`. Entries are measured data, not aspiration -- edit only from a fresh
authoritative measurement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasuredCandidate:
    seam: str
    speedup: float          # median of per-round baseline/candidate ratios
    correct: bool


# Recorded evidence, keyed by (dtype, seq_len_regime). "short" = seq <= 512.
# Speedups from the authoritative evaluator (median-latency ratio).
_EVIDENCE: dict[tuple[str, str], tuple[MeasuredCandidate, ...]] = {
    ("float32", "short"): (
        MeasuredCandidate("cublastf32", 1.16, True),   # SDPA + cuBLAS TF32 GEMMs
        MeasuredCandidate("full", 0.974, True),         # hand-written kernels
        MeasuredCandidate("qkv", 0.971, True),
        MeasuredCandidate("flash", 0.949, True),
        MeasuredCandidate("tf32", 0.936, True),
    ),
}

# Require a real margin over the baseline before deploying a candidate: below this the
# measured "win" is inside the timer's own round-to-round spread.
_MARGIN = 1.02


def select(dtype: str, seq_len: int, margin: float = _MARGIN) -> str:
    """Return the seam to deploy: the fastest *correct* candidate that clears `margin`,
    else "baseline" (the untuned fallback)."""
    regime = "short" if seq_len <= 512 else "long"
    cands = [c for c in _EVIDENCE.get((dtype, regime), ()) if c.correct]
    if not cands:
        return "baseline"
    best = max(cands, key=lambda c: c.speedup)
    return best.seam if best.speedup >= margin else "baseline"
