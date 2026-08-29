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


def _v3(baseline_cls):
    from .v3_chunked import build
    return build(baseline_cls)


def _v4(baseline_cls):
    from .v4_tunable import build
    return build(baseline_cls)


def _v5(baseline_cls):
    from .v5_fp16_resid import build
    return build(baseline_cls)


def _v6(baseline_cls):
    from .v6_fp16_gelu import build
    return build(baseline_cls)


def _v7(baseline_cls):
    from .v7_fused_norm import build
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
    "v3_chunked": CandidateSpec(
        name="v3_chunked", generation=3, parent="v2_fp16_flash", build=_v3,
        summary="v2 plus batch chunking sized from the measured L2 capacity, aimed at "
                "configs 6 and 13 (93.4% of all baseline time). Falls through to plain "
                "v2 wherever the whole batch already fits the residency target.",
    ),
    "v4_tunable": CandidateSpec(
        name="v4_tunable", generation=4, parent="v3_chunked", build=_v4,
        summary="v3 with its three constants read from the environment, so the search "
                "loop can evaluate a point without rewriting source. Identical to v3 at "
                "the defaults.",
    ),
    "v5_fp16_resid": CandidateSpec(
        name="v5_fp16_resid", generation=5, parent="v3_chunked", build=_v5,
        summary="Residual stream kept in fp16 for the whole stack, with fp16 LayerNorm "
                "weights and GELU in fp16, to delete the ~6 dtype conversions per layer "
                "that profiling put at 12.8-26.8% of candidate kernel time. Tests "
                "whether the accumulated fp16 error stays inside the 2e-3 budget.",
    ),
    "v6_fp16_gelu": CandidateSpec(
        name="v6_fp16_gelu", generation=6, parent="v3_chunked", build=_v6,
        summary="v3 with exactly one non-accumulating round-trip removed: GELU runs in "
                "fp16 instead of upcast-gelu-downcast. Tests the distinction v5 "
                "established -- the residual accumulates, an elementwise op does not.",
    ),
    "v7_fused_norm": CandidateSpec(
        name="v7_fused_norm", generation=7, parent="v6_fp16_gelu", build=_v7,
        summary="v6 with the LayerNorm downcast folded into the norm's own epilogue via "
                "cached fp16 norm weights, attacking the 9.7-16.8% of kernel time in "
                "native_layer_norm plus 2.5-9.6% in add. Expected to pay on the "
                "bandwidth-bound configs (6, 13) and not on the launch-bound ones.",
    ),
}
