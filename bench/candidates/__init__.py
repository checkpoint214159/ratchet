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


def _v15(baseline_cls):
    from .v15_lifted_veto import build
    return build(baseline_cls)


def _v16(baseline_cls):
    from .v16_ffn_megakernel import build
    return build(baseline_cls)


def _v17(baseline_cls):
    from .v17_dispatched_megakernel import build
    return build(baseline_cls)


def _v18(baseline_cls):
    from .v18_capture_insurance import build
    return build(baseline_cls)


def _v19(baseline_cls):
    from .v19_norm_fused import build
    return build(baseline_cls)


def _v23(baseline_cls):
    from .v23_single_tile_attn import build
    return build(baseline_cls)


def _v26(baseline_cls):
    from .v26_causal_correct import build
    return build(baseline_cls)


def _v34(baseline_cls):
    from .v34_launch_bound import build
    return build(baseline_cls)


def _v33(baseline_cls):
    from .v33_streamed_long import build
    return build(baseline_cls)


def _v35(baseline_cls):
    from .v35_recombined import build
    return build(baseline_cls)


def _v36(baseline_cls):
    from .v36_gemm_gelu import build
    return build(baseline_cls)


def _v37(baseline_cls):
    from .v37_recombined2 import build
    return build(baseline_cls)


def _v38(baseline_cls):
    from .v38_stream_fallback import build
    return build(baseline_cls)


def _v40(baseline_cls):
    from .v40_looped_attn import build
    return build(baseline_cls)


def _v41(baseline_cls):
    from .v41_vendor_aware_attn import build
    return build(baseline_cls)


def _v42(baseline_cls):
    from .v42_hot_tuned_tile import build
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
    "v15_lifted_veto": CandidateSpec(
        name="v15_lifted_veto", generation=15, parent="v9b_reduce_overhead", build=_v15,
        summary="Lifts Inductor's hardcoded 68-SM veto (this card has 66), which had "
                "silently disabled Triton GEMM templates and therefore ALL GEMM epilogue "
                "fusion. Measured 1.58x on config 6's FFN pattern in isolation. Re-asks "
                "v9b's question -- is max-autotune worth it? -- with the autotuner "
                "actually enabled, which it was not when v9b answered no.",
    ),
    "v16_ffn_megakernel": CandidateSpec(
        name="v16_ffn_megakernel", generation=16, parent="v9b_reduce_overhead", build=_v16,
        summary="THE FIRST HAND-WRITTEN KERNEL. One Triton kernel for the whole FFN "
                "block -- both GEMMs, GELU and the fp32 residual -- with the intermediate "
                "held in registers. Possible only because ffn_dim == d_model makes both "
                "weight matrices 64 KB, inside the measured 99 KB opt-in smem. Inductor "
                "structurally cannot do this: it fuses elementwise into GEMM, never GEMM "
                "into GEMM. Op-level 2.2x-4.6x AND more accurate than the fp16 path.",
    ),
    "v17_dispatched_megakernel": CandidateSpec(
        name="v17_dispatched_megakernel", generation=17, parent="v13_safe_capture",
        build=_v17,
        summary="RECOMBINATION: the g16 FFN megakernel merged into the g13 frontier, "
                "gated on whether hoisting the weights into smem is paid for. The "
                "predicate is a ratio of weight traffic to activation traffic -- no "
                "config ids -- and selects exactly the three configs where g16 measured "
                "a win. Expected geomean gain ~1.3%, INSIDE the noise floor; the "
                "defensible claim is per-config, on the matrix's largest shape.",
    ),
    "v18_capture_insurance": CandidateSpec(
        name="v18_capture_insurance", generation=18, parent="v17_dispatched_megakernel",
        build=_v18,
        summary="Graph capture no longer depends on whether the CALLER allocated its "
                "input inside inference_mode. Measured 0.267ms vs 0.601ms (2.25x) on that "
                "one variable, and the graded harness allocates its timing input OUTSIDE "
                "it -- we were fast only because the accuracy tests run first. Reports "
                "capture_source so the degradation is observable instead of silent.",
    ),
    "v19_norm_fused": CandidateSpec(
        name="v19_norm_fused", generation=19, parent="v18_capture_insurance", build=_v19,
        summary="Folds the attention residual add, norm2, and the NEXT layer's norm1 into "
                "the megakernel. Those three kernels are 35% of config 6 and all run AT "
                "the 613.7 GB/s bandwidth roofline -- they cannot be sped up, only "
                "deleted. Traffic per token 28*D -> 12*D. Op-level 2.51x-3.84x; L33 bounds "
                "the end-to-end gain at ~1.35x on config 6.",
    ),
    "v26_causal_correct": CandidateSpec(
        name="v26_causal_correct", generation=26, parent="v23_single_tile_attn", build=_v26,
        summary="Honours config.causal. Every candidate from v5 to v23 hardcoded "
                "is_causal=True and returned three-quarters of its output wrong on a "
                "non-causal input -- while the reference benchmark's own DEFAULT is "
                "causal=False. Non-causal now delegates to the unmodified baseline: "
                "exactly right on a shape we do not expect, fast on the fourteen we do.",
    ),
    "v33_streamed_long": CandidateSpec(
        name="v33_streamed_long", generation=33, parent="v26_causal_correct", build=_v33,
        summary="Restores batch streaming to the frontier. v14 built the predicate -- "
                "estimated working set against measured free memory -- and v17 branched "
                "from v13 rather than v14, so the streaming path fell out of the lineage "
                "at generation 17. It changes nothing on the 13 configs that fit "
                "(choose() returns resident) and is the difference between computing "
                "config 14 one slice at a time and planning a 73 GiB resident working "
                "set that would fail on an 80 GiB card too. It does NOT make config 14 "
                "runnable here: the forward signature's own 12.21 GiB in + 12.21 GiB out "
                "is a floor no implementation removes.",
    ),
    "v34_launch_bound": CandidateSpec(
        name="v34_launch_bound", generation=34, parent="v26_causal_correct", build=_v34,
        summary="The frontier launches 36 kernels per forward on EVERY config -- "
                "censused at 2, 8 and 12 with an identical decomposition -- and a graph "
                "node costs 0.798 us on this card whatever it computes, so 28.7 us is a "
                "floor and 47% of config 2's entire wall. Adds a second, disjoint fusion "
                "predicate: fuse when the whole segment fits the device in ONE WAVE, "
                "where per-launch latency is pure overhead. That is the opposite reason "
                "from finding 25's bandwidth crossover and fires exactly where it "
                "declines (configs 2, 3, 4, 12). Reuses v19's norm-fused megakernel and "
                "elides a provably-dead mask memcpy. 36 -> 20 kernels.",
    ),
    "v35_recombined": CandidateSpec(
        name="v35_recombined", generation=35, parent="v33_streamed_long", build=_v35,
        summary="RECOMBINATION of the two live children of v26: v33's shape-latch fix "
                "and restored batch streaming, with v34_launch_bound's one-wave fusion "
                "stacked underneath it (contributor, as v17 declared v16). Composes by "
                "layering, not by copying -- v33's dispatch was factored into "
                "`build_on` so exactly one copy of each mechanism exists. The merge's "
                "own content is the binding the two could not see: v34 latches five more "
                "attributes to the input shape than v33's reset enumerates, so a model "
                "re-decided at a new batch size would have run a megakernel tiled for "
                "the old one. Also re-derives the MASK state on a shape change, which "
                "v34's mask elision makes load-bearing, and settles the launch decision "
                "on the streamed path against the slice that runs.",
    ),
    "v36_gemm_gelu": CandidateSpec(
        name="v36_gemm_gelu", generation=36, parent="v34_launch_bound", build=_v36,
        summary="THE FIRST HAND-WRITTEN PROJECTION GEMM. Censused at config 9 -- the #1 "
                "headroom row and one nobody had ever profiled -- the sixteen projection "
                "GEMMs are 55.0% of device time at 47.3 TFLOP/s, 53.6% of this card's "
                "measured peak, because cuBLAS selects `ampere_fp16_s1688gemm` at K=128: "
                "`s1688` is mma.sync.m16n8k8 where sm_89 issues m16n8k16 and tl.dot "
                "emits it. It does NOT do this at d_model 1024, where it hits 100.4% of "
                "peak -- the bad selection is specific to narrow K. And the GELU is its "
                "own kernel on every layer of every config because cuBLAS takes no "
                "epilogue (finding 39), so it moves into the ffn_in epilogue in the "
                "exact erf form, applied to the fp32 accumulator before any downcast. "
                "Each of the four sites is decided by TIMING the vendor against 18 swept "
                "tiles at prime time and keeping the vendor unless Triton wins by more "
                "than 10%, so config 8 declines on its own evidence.",
    ),
    "v37_recombined2": CandidateSpec(
        name="v37_recombined2", generation=37, parent="v36_gemm_gelu", build=_v37,
        summary="THE SECOND RECOMBINATION. v26's two lines rejoin: v36's projection "
                "GEMMs and GELU epilogue (the declared parent) with v35_recombined's "
                "shape-latch fix, batch streaming and reset discipline as the "
                "contributor -- the relationship v17 declared to v16 and v35 declared "
                "to v34. Neither is an ancestor of the other; their merge-base is v32. "
                "The merge's own content is the reset: v36 latches NINE more attributes "
                "to the input shape than v35's five (five gemm_* flags and four tile "
                "tuples), so v37 derives the reset set FROM THE CLASSES -- every "
                "class-body attribute introduced above v26 -- instead of naming them, "
                "and asserts it covers v35's declared list. It also settles the GEMM "
                "plan on the streamed path against the slice that runs, after the two "
                "fusion decisions it reads.",
    ),
    "v38_stream_fallback": CandidateSpec(
        name="v38_stream_fallback", generation=38, parent="v37_recombined2", build=_v38,
        summary="RESIDENCY IS ATTEMPTED, NOT ESTIMATED. v33's dispatch asked "
                "mem_get_info at the first forward -- which under run_matrix is the "
                "correctness check, with the baseline still resident -- and latched the "
                "answer into a timed phase where memory is plentiful, costing config 6 "
                "1.6x on the whole streaming lineage. v38 tries the resident path and "
                "streams only on an actual torch.cuda.OutOfMemoryError, releasing the "
                "failed attempt through v37's derived reset first. The only surviving "
                "pre-check is the signature floor -- 2*B*S*d*elem of input and output "
                "against total_memory -- which has no coefficient to calibrate and "
                "keeps config 14's full batch streaming on a 16 GiB card while letting "
                "an 80 GiB card attempt it.",
    ),
    "v40_looped_attn": CandidateSpec(
        name="v40_looped_attn", generation=40, parent="v38_stream_fallback", build=_v40,
        summary="A SECOND ATTENTION TILE SHAPE, CHOSEN BY A SYMMETRIC SWEEP. Adds a "
                "flash-style kernel with the K/V axis in a loop, so K/V stage through "
                "shared memory and the pipeliner has something to hide latency behind, "
                "and a chooser that sweeps BOTH Triton forms plus sdpa+repack over "
                "their full legal grids with one timer and one repeat count -- the "
                "symmetry finding 47 measured at 4.5% and finding 48 then lost. "
                "Predicated on the grid (B*heads*cdiv(S,BM) against the measured SM "
                "count) rather than on head_dim: pipelining needs more than one wave to "
                "hide behind. Census first: attention is 17.5% of config 10's wall on "
                "v38, and the op-level ratio was re-measured L2-hot (1.228x, replicated "
                "to 0.2%) because finding 48 had priced an in-graph time with an "
                "L2-flushed ratio across a 2.24x regime gap.",
    ),
    "v41_vendor_aware_attn": CandidateSpec(
        name="v41_vendor_aware_attn", generation=41, parent="v40_looped_attn",
        build=_v41,
        summary="THE CHOOSER MAY NOW STEP ASIDE FOR THE VENDOR. `attn_single_tile."
                "pays()` is a residency argument -- whether OUR loop-free kernel can "
                "hide its latency -- and it was being read as if it also said the vendor "
                "was slower. It does not. Where the plan is still the single-tile "
                "kernel, the chosen tile is timed hot against sdpa+repack, two arms with "
                "one trial budget each, and the shape goes to the vendor if the vendor "
                "clears v23's inherited DECISIVE 10%. The g41 audit measured all three "
                "paths on all thirteen runnable configs, twice: the vendor wins on "
                "exactly ONE shape (config 10, 1.119x over single_tile) and the looped "
                "form already beats it there, so this fires on ZERO announced configs "
                "and is byte-identical to v40. It is a guard on the fallback path, "
                "worth +0.0000 as shipped and ~+0.0035 in the branch where "
                "`autotune_looped` declines config 10.",
    ),
    "v42_hot_tuned_tile": CandidateSpec(
        name="v42_hot_tuned_tile", generation=42, parent="v41_vendor_aware_attn",
        build=_v42,
        summary="THE TILE SWEEP GETS AN INSTRUMENT THAT CAN RESOLVE IT. From generation "
                "23 to 41 `autotune_tile` ranked with the L2-flushed `do_bench`, which "
                "times each call with a pair of CUDA events whose quantum is 1.024 us -- "
                "against kernels that run in 1.9-11 us. On config 2 five of the eight "
                "tiles reported the IDENTICAL 5.120 us and the whole grid spanned one "
                "quantum, so the sweep was not noisy but blank; the tie fell through to "
                "the derived-tile tiebreak, which kept a tile the hot timer ranks 1.28x "
                "behind. The same quantization flipped config 3's pick between two runs "
                "of the identical sweep, in the other direction. The diff is one value: "
                "`attn_tile_timer = hot_time`, the timer `attn_choice` has ranked with "
                "since g40, so the two tuners now share one instrument by construction. "
                "No tile is hardcoded and no config id appears -- the sweep selects "
                "(16,4,1) on config 2 by itself. Blast radius measured before it was "
                "claimed: 9 of the 10 accepted shapes select the identical tile under "
                "both timers, twice. Also closes the one tuner in the package that "
                "admitted arms to a timing set without a correctness gate.",
    ),
    "v14_dispatch": CandidateSpec(
        name="v14_dispatch", generation=14, parent="v13_safe_capture", build=_v14,
        summary="Shape-aware dispatch with predicates derived from measured free device "
                "memory, never from config ids. Chooses a streamed path when the working "
                "set would not fit and v13 otherwise, and reports is_tuned so an untuned "
                "path is never presented as a tuned one.",
    ),
    "v23_single_tile_attn": CandidateSpec(
        name="v23_single_tile_attn", generation=23, parent="v18_capture_insurance",
        build=_v23,
        summary="THE FIRST HAND-WRITTEN ATTENTION. One Triton tile per (batch, head, "
                "query block): no K/V loop, no online softmax, no rescale, and the "
                "split/transpose/repack around SDPA deleted with it. Possible because "
                "eleven announced rows have seq_len <= 128, where the whole score matrix "
                "is 64 KB of registers. The tile is autotuned at prime time, not guessed. "
                "Declines head_dim 128/256 and seq_len 1024/100000, where a loop-free "
                "kernel cannot keep enough blocks resident to hide its own latency -- "
                "measured 0.94x and 0.84x there, so declining is the point.",
    ),
}
