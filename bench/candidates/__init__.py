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


def _v8(baseline_cls):
    from .v8_padfast import build
    return build(baseline_cls)


def _v9a(baseline_cls):
    from .v9_compiled_core import build
    return build(baseline_cls)
def _v9b(baseline_cls):
    from .v9b_reduce_overhead import build
    return build(baseline_cls)


def _v10a(baseline_cls):
    from .v10_ablation import build_no_chunk
    return build_no_chunk(baseline_cls)


def _v10b(baseline_cls):
    from .v10_ablation import build_no_fused_qkv
    return build_no_fused_qkv(baseline_cls)


def _v10c(baseline_cls):
    from .v10_ablation import build_no_fp16
    return build_no_fp16(baseline_cls)


def _v11(baseline_cls):
    from .v11_lean import build
    return build(baseline_cls)



def _v12(baseline_cls):
    from .v12_graph_over_compile import build
    return build(baseline_cls)



def _v13(baseline_cls):
    from .v13_safe_capture import build
    return build(baseline_cls)



def _v14(baseline_cls):
    from .v14_dispatch import build
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
    "v8_padfast": CandidateSpec(
        name="v8_padfast", generation=8, parent="v6_fp16_gelu", build=_v8,
        summary="Takes the fp16 flash path even when the input is padded, on the proof "
                "that a right-padded causal key mask is redundant. Fixes the blind spot "
                "that halved every speedup at padding_ratio>0. Guarded: the prefix shape "
                "is verified at prime time, else it falls back to v6's slow path.",
    ),
    "v9a_compiled_core": CandidateSpec(
        name="v9a_compiled_core", generation=9, parent="v8_padfast", build=_v9a,
        summary="Sibling A of the g9 fork. Keeps v8's algorithm (flash, chunking, the "
                "padding proof) and hands the op sequence to Inductor so it fuses the "
                "elementwise chains v7 could not fuse by hand without breaking precision.",
    ),
    "v9b_reduce_overhead": CandidateSpec(
        name="v9b_reduce_overhead", generation=9, parent="v8_padfast", build=_v9b,
        summary="Sibling B of the g9 fork. Same parent and hypothesis as v9a with one "
                "variable changed: reduce-overhead instead of max-autotune. Answers "
                "whether the 2-19s per-shape autotuning cost buys anything measurable.",
    ),
    "v10a_no_chunk": CandidateSpec(
        name="v10a_no_chunk", generation=10, parent="v9a_compiled_core", build=_v10a,
        summary="Ablation: v9a without L2-sized batch chunking. Does Inductor already "
                "handle the working set, making v3's chunk loop dead weight?",
    ),
    "v10b_no_fusedqkv": CandidateSpec(
        name="v10b_no_fusedqkv", generation=10, parent="v9a_compiled_core", build=_v10b,
        summary="Ablation: v9a with three separate Q/K/V projections instead of the "
                "fused cat. Does Inductor fuse them itself?",
    ),
    "v10c_no_fp16": CandidateSpec(
        name="v10c_no_fp16", generation=10, parent="v9a_compiled_core", build=_v10c,
        summary="Ablation: v9a in pure fp32, no fp16 weight cache. Does our hand-rolled "
                "mixed precision still beat the compiler's own choice?",
    ),
    "v11_lean": CandidateSpec(
        name="v11_lean", generation=11, parent="v9a_compiled_core", build=_v11,
        summary="The frontier with dead weight removed: chunking deleted after the g10 "
                "ablation showed it subsumed by the compiler, reduce-overhead instead of "
                "max-autotune. Five remaining components, each with a measurement behind "
                "it and none inherited on faith.",
    ),
    "v12_graph_over_compile": CandidateSpec(
        name="v12_graph_over_compile", generation=12, parent="v11_lean", build=_v12,
        summary="Compile for fusion (default mode, no Inductor cudagraphs) then capture "
                "the compiled callable in our own static-buffer graph, so the steady "
                "state is one replay with no Dynamo guard evaluation. Motivated by "
                "config 2 profiling: 22.5us/call of Dynamo cache lookup on a ~97us call.",
    ),
    "v13_safe_capture": CandidateSpec(
        name="v13_safe_capture", generation=13, parent="v12_graph_over_compile", build=_v13,
        summary="v12 with fail-safe capture. v12 can capture an EMPTY graph under some "
                "call patterns, after which replay() is a no-op and it returns a stale "
                "buffer -- silently wrong. v13 verifies the graph against a freshly "
                "computed reference and falls back to the compiled callable if capture "
                "is not provably real.",
    ),
    "v14_dispatch": CandidateSpec(
        name="v14_dispatch", generation=14, parent="v13_safe_capture", build=_v14,
        summary="Shape-aware dispatch with predicates derived from measured free device "
                "memory, never from config ids. Chooses a streamed path when the working "
                "set would not fit and v13 otherwise, and reports is_tuned so an untuned "
                "path is never presented as a tuned one.",
    ),
}
