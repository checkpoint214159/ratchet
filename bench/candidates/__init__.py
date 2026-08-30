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


def _v27(baseline_cls):
    from .v27_qkv_fused_attn import build
    return build(baseline_cls)


def _v29(baseline_cls):
    from .v29_copy_elimination import build
    return build(baseline_cls)


def _v20(baseline_cls):
    from .v20_headmajor_qkv import build
    return build(baseline_cls)


def _v21(baseline_cls):
    from .v21_double_buffered import build
    return build(baseline_cls)


def _v22(baseline_cls):
    from .v22_headdim8_attn import build
    return build(baseline_cls)


def _v24(baseline_cls):
    from .v24_outproj_prologue import build
    return build(baseline_cls)


def _v25(baseline_cls):
    from .v25_fp16_accum import build
    return build(baseline_cls)


def _v28(baseline_cls):
    from .v28_layer_megakernel import build
    return build(baseline_cls)


def _v31(baseline_cls):
    from .v31_outproj_epilogue import build
    return build(baseline_cls)


def _v34(baseline_cls):
    from .v34_launch_bound import build
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
    "v29_copy_elimination": CandidateSpec(
        name="v29_copy_elimination", generation=29, parent="v26_causal_correct",
        build=_v29,
        summary="Removes the CUDA graph's output clone, which a fresh profile of v26 puts "
                "at roughly half of a 7.2% Memcpy DtoD bill on config 6. Not by deleting "
                "it -- that is finding 24 -- but by asking, before each replay, whether "
                "anything still refers to the tensor handed out last time: nothing does, "
                "no copy; the caller holds it un-aliased, rebind it onto a clone (the "
                "parent's cost); the caller holds an un-rebindable alias, stop handing the "
                "buffer out. Ports cand/g21/double-buffered onto the current frontier, "
                "drops its second buffer (structurally absent on configs 6 and 13, the "
                "only shapes where the copy is worth anything), calibrates the alias "
                "sensor instead of hardcoding its threshold, and makes an alias event cost "
                "the parent rather than the compiled fallback. Ceiling ~3.6% on config 6, "
                "INSIDE the noise floor: not resolvable by the screen or by a geomean.",
    ),
    "v14_dispatch": CandidateSpec(
        name="v14_dispatch", generation=14, parent="v13_safe_capture", build=_v14,
        summary="Shape-aware dispatch with predicates derived from measured free device "
                "memory, never from config ids. Chooses a streamed path when the working "
                "set would not fit and v13 otherwise, and reports is_tuned so an untuned "
                "path is never presented as a tuned one.",
    ),
    "v27_qkv_fused_attn": CandidateSpec(
        name="v27_qkv_fused_attn", generation=27, parent="v26_causal_correct", build=_v27,
        summary="The Q|K|V projection fused INTO the attention kernel: one launch loads "
                "the normalized input tile, projects its own head's Q, K and V in "
                "registers and attends, so the [B, S, 3*d_model] buffer -- written by a "
                "cuBLAS GEMM at 15.4% of config 6 and read straight back -- never "
                "exists. A program owns ONE head, so its weight slice is 24 KB at "
                "head_dim 32, not the 96 KB of the whole matrix; the binding operand is "
                "the input tile, which refuses d_model 1024 and seq_len 1024. Declines "
                "config 11 on a byte count: 16 heads each re-reading the input tile "
                "move more than the buffer they delete. Op-level 1.29x on config 6's "
                "shape, 1.50x on config 7; the geomean should look flat (L33).",
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
    "v20_headmajor_qkv": CandidateSpec(
        name="v20_headmajor_qkv", generation=20, parent="v18_capture_insurance", build=_v20,
        summary="SIBLING of v19 off v18. A fused QKV GEMM that scatters straight into "
                "head-major buffers, so flash reads contiguously instead of through a "
                "stride jumping by 3*D. The tax is 1.78x of attention at config 6 and "
                "cannot be repacked away (.contiguous() costs more than it saves). Tuned "
                "kernel beats cuBLAS+strided by 1.163x on the segment.",
    ),
    "v21_double_buffered": CandidateSpec(
        name="v21_double_buffered", generation=21, parent="v18_capture_insurance",
        build=_v21,
        summary="Kills the graph's output copy. The frontier clones _static_y on every "
                "call because the next replay overwrites it; a profile puts Memcpy DtoD "
                "at 6.6% of config 6's forward, about half of it that clone. Two "
                "graphExecs sharing one memory pool give two distinct output buffers, "
                "and a liveness check before each clobber preserves anything the caller "
                "still holds (rebinding it onto fresh memory) or retires the buffer and "
                "falls back to the compiled callable. Double buffering ALONE would only "
                "be safe by accident (L24); the liveness check is what makes it correct "
                "for callers the harness does not model. Ceiling ~3% on config 6, ~2.5% "
                "on 13, ~0 elsewhere -- INSIDE the noise floor by construction.",
    ),
    "v22_headdim8_attn": CandidateSpec(
        name="v22_headdim8_attn", generation=22, parent="v18_capture_insurance",
        build=_v22,
        summary="A hand-written Triton causal attention kernel for head_dim BELOW the "
                "tl.dot contraction floor the Triton backend reports for this device "
                "(16 on sm_89, mma.sync.m16n8k16). PyTorch's FlashAttention-2 has no "
                "head_dim=8 kernel and HEADDIM_SWITCH rounds it to 32, so the vendor is "
                "mis-tiled, NOT refused -- finding 23 killed the refusal premise. "
                "Op-level 1.40x on configs 7 and 11, indicative. End to end ~1.10x and "
                "~1.13x on those two configs, ~+1.7% on the 13-config geomean, which is "
                "INSIDE the noise floor, and ~zero under weighted_score because config 11 "
                "is already capped. Per-config claim only.",
    ),
    "v24_outproj_prologue": CandidateSpec(
        name="v24_outproj_prologue", generation=24, parent="v18_capture_insurance",
        build=_v24,
        summary="The attention out-projection, its fp32 widening and the fp32 residual "
                "add in one Triton kernel, so the fp16 [M, D] temporary between them "
                "never exists -- 29% of that segment's traffic and one launch of two. "
                "Measured 1.31x-1.55x on the segment against the compiled two-kernel "
                "path with no losing shape, and ~600x tighter against fp64 because the "
                "fusion DELETES an fp16 rounding step. KILLS proposal D-02's headline: "
                "SDPA returns ctx token-major-contiguous, so the 'head-major gather' it "
                "was built to absorb does not exist. Expected end-to-end effect is "
                "2-3% at config 6, inside the noise floor.",
    ),
    "v25_fp16_accum": CandidateSpec(
        name="v25_fp16_accum", generation=25, parent="v18_capture_insurance",
        build=_v25,
        summary="fp16 MMA accumulation, probed per SITE and CLOSED. The hardware reading "
                "is correct -- tl.dot(out_dtype=float16) emits f16.f16.f16.f16 mma and "
                "measures 1.569x in an MMA-saturated loop -- but the fused FFN runs at "
                "99.2% of measured HBM bandwidth and 35.3% of peak FLOPs, so the faster "
                "instruction is off the critical path and config 6 measures 1.000x. And "
                "the error fails everywhere it could pay: 3.2e-3 to 8.4e-3 against a "
                "2.0e-3 budget from ONE site in ONE layer. Ships as fp32, i.e. identical "
                "to v18, with the predicate that declines and the falsifier that measures "
                "the refused path.",
    ),
    "v28_layer_megakernel": CandidateSpec(
        name="v28_layer_megakernel", generation=28, parent="v26_causal_correct",
        build=_v28,
        summary="Attention and the FFN in ONE launch -- the whole layer as a single "
                "Triton kernel: LN1, Q|K|V, causal attention, out-proj, fp32 residual, "
                "LN2, both FFN GEMMs and the second residual, with nothing between the "
                "first load and the final store touching HBM. Reconciles A-03/B-01/B-07: "
                "B-07's scope is already built (v17/v19) and measured flat, so the open "
                "question is fusing ACROSS the GEMMs, which deletes the QKV buffer -- "
                "25% of config 6's layer traffic and the one piece nothing has reached. "
                "31.5 GB -> 7.9 GB on config 6, which moves it PAST the 144 FLOP/B ridge "
                "and makes the answer depend on achieved tensor-core utilisation, not on "
                "traffic: break-even is 20% of peak. The register file binds, not smem, "
                "and the tile is swept with the compiler's own n_spills as a filter.",
    ),
    "v31_outproj_epilogue": CandidateSpec(
        name="v31_outproj_epilogue", generation=31, parent="v26_causal_correct",
        build=_v31,
        summary="The out-projection, the fp32 widen, the padding mask and the fp32 "
                "residual add absorbed into the single-tile attention kernel's own "
                "epilogue: one program per (batch, query block) loops over heads and "
                "accumulates the projection in registers, so the context tile never "
                "reaches HBM. Halves the epilogue's traffic (16D -> 8D bytes per token "
                "per layer) and removes two launches of three, and deletes the fp16 "
                "rounding of the projection output. Costs a factor of `heads` in grid "
                "size and an fp32 [BM, d_model] accumulator in registers, so it declines "
                "where the grid stops covering the SMs (configs 2, 3) or fewer than four "
                "blocks stay resident (9, 10, 13), falling back to v23's split path.",
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
}
