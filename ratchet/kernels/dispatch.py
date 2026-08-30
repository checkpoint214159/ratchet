"""E6: evidence-driven dispatch with an explicit untuned fallback.

The repo's contract (docs/optimization-principles.md, specs/04-dispatch.md): a candidate is
deployed for a workload only if *measured* evidence shows it beats the baseline on that
workload; otherwise the baseline is the fallback. Dispatch never guesses -- it reads the
recorded speedup and, when nothing clears 1.0x with margin, returns the baseline.

On GB10 (fp32, seq=128, the authoritative config) every hand-written candidate measured
below 1.0x with the drift-robust interleaved timer (tests/manual/timed_compare.py):

    flash 0.949x | tf32x3 0.936x | qkv 0.971x | full 0.974x   (all correct=True)

so this table dispatches to the baseline there. The entries are data, not aspiration: edit
them only from a fresh measurement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasuredCandidate:
    seam: str
    speedup: float          # median of per-round baseline/candidate ratios
    correct: bool


# Recorded evidence, keyed by (dtype, seq_len_regime). "short" = seq <= 512.
_EVIDENCE: dict[tuple[str, str], tuple[MeasuredCandidate, ...]] = {
    ("float32", "short"): (
        MeasuredCandidate("flash", 0.949, True),
        MeasuredCandidate("tf32", 0.936, True),
        MeasuredCandidate("qkv", 0.971, True),
        MeasuredCandidate("full", 0.974, True),
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
