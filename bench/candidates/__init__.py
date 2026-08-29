"""The candidate archive — every implementation that has been measured, kept.

Stepping stones are preserved rather than replaced. A candidate that was superseded is
still the parent of whatever superseded it, and clade metaproductivity scores a parent by
its descendants' outcomes, so deleting a mediocre ancestor destroys the evidence that
made its successor findable.

LINEAGE. These first two were measured before the branch protocol existed, so both are
recorded in the ledger against the trunk commit that introduced them, with `parent`
stated here rather than inferred from git. Candidates generated from here on get their
own branch (`cand/<generation>/<slug>`) and their lineage comes from git ancestry — see
`bench/README.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    generation: int
    parent: Optional[str]
    build: Callable          # build(baseline_cls) -> candidate class
    summary: str


def _v1(baseline_cls):
    from .v1_fused_graph import build
    return build(baseline_cls)


def _v2(baseline_cls):
    from .v2_fp16_flash import build
    return build(baseline_cls)


REGISTRY: dict[str, CandidateSpec] = {
    "v1_fused_graph": CandidateSpec(
        name="v1_fused_graph", generation=1, parent=None, build=_v1,
        summary="Fused Q|K|V, lazy fp16 GEMM cache with fp32 round-trip, SDPA, "
                "static-buffer CUDA graph. 3.11x geomean over the matrix. Never "
                "actually reached flash attention.",
    ),
    "v2_fp16_flash": CandidateSpec(
        name="v2_fp16_flash", generation=2, parent="v1_fused_graph", build=_v2,
        summary="v1 plus: q/k/v kept in fp16 and the all-True mask elided, so "
                "FlashAttention finally qualifies. 5.64x geomean; the only variant "
                "that can run config 14 at all.",
    ),
}
