"""Empirical lane: the announced competition matrix, and measurements against it.

Kept deliberately separate from `research/` (the fail-closed, FG-01-gated evidence
archive). Nothing here claims to be ratified evidence under that hierarchy; these are
working measurements on the one machine in the team that has a CUDA device. See
`bench/README.md` for how the two relate and what would be needed to promote a result
from here into the sanctioned archive.
"""

from .matrix import MATRIX, BY_ID, Config, REGIMES, regime_of, weighted_score  # noqa: F401
