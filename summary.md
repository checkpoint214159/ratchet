# Cross-Branch Transformer-Optimization Summary

**Purpose.** This document compares the experimental work on the four remote branches at
their reviewed tips. It separates hardware measurements, unqualified working measurements,
implementation-only work, and synthetic experiments so that a number is never presented as
more general than its evidence permits.

## How to read the results

| Evidence class | Branches | Meaning | What it does **not** establish |
| --- | --- | --- | --- |
| Hardware measurement | `marcus` | Results from a real NVIDIA GB10 under the branch's submission harness. | Performance on another GPU, model matrix, dtype, or baseline. |
| Working but unqualified CUDA measurement | `ben` | Reproducible RTX 4070 Ti SUPER measurements in `bench/results.jsonl`; the branch intentionally does not promote them to the fail-closed research archive. | A ratified catalogue event or an Intel-XPU result. |
| Implementation / no-run | `Aekas` | Kernel and benchmark-protocol code exists, but the CPU-only environment recorded `EVT-000001` and zero empirical events. | Correctness, latency, memory, or speedup benefit. |
| Deterministic synthetic experiment | `brian` | Seeded ranking experiments measure how well methods forecast tuning difficulty. | GPU-kernel or end-to-end transformer performance. |

> **Do not compare headline speedups across branches.** `marcus` and `ben` use different
> GPUs, software stacks, workload details, branches of the benchmark, and measurement
> protocols. `Aekas` has no performance data and `brian` measures a ranking correlation,
> not latency. A speedup is meaningful only with its branch, baseline, matrix, and evidence
> class.

## Results index

| Branch | Optimization focus | Environment / evidence | Current result to quote | Primary limitation |
| --- | --- | --- | --- | --- |
| `marcus` | Triton attention, fp16 GEMM, CUDA graphs, and shape-aware dispatch | NVIDIA GB10 (`sm_121`), torch 2.9.1+cu130, Triton 3.5.1; 13 correct measured configurations | **3.20× geomean vs eager** (official/scored baseline); **2.62× vs `torch.compile`** | Configuration 14 is not comparable because the fp32 reference OOMs; results are GB10-specific. |
| `ben` | Multi-generation, shape-specific dispatch across attention, GEMM, graphs, and FFN | NVIDIA RTX 4070 Ti SUPER (`sm_89`), WSL2, torch 2.8.0+cu128, Triton 3.4.0; working measurements, explicitly unqualified | Historical milestone: **3.25× geomean vs compiled baseline** for `v34_launch_bound`; current `v43_replicated_tile` is the submitted candidate | Unlocked clocks, protocol-specific comparisons, and no promotion to the ratified archive. |
| `Aekas` | Fused LayerNorm/residual and GELU operations plus future XPU protocol | CPU-only; `EVT-000001` is a no-run and `triton_fused.json` is `unvalidated` | **No empirical result** | No PyTorch/qualified accelerator run; implementations must not be described as optimized in performance terms. |
| `brian` | Forecast which tuning configurations deserve search budget | 100-trial seeded synthetic experiments over CUDA/HIP and fp32/fp8 scenarios | Failure-weighted PageRank has the best Spearman correlation: **0.333–0.433** | The target is synthetic difficulty ranking, not kernel latency or correctness. |

## `marcus`: GB10 measured kernel submission

**Optimization objective.** Replace the costly transformer-layer paths for the announced
causal matrix while retaining the evaluator's fp32-accumulation correctness boundary.
The comparison distinguishes the evaluator's eager baseline (the official/scored metric)
from the tougher self-imposed `torch.compile` baseline.

**Techniques and what the evidence supports.**

| Technique | Implementation | Observed benefit | Boundary / negative result |
| --- | --- | --- | --- |
| Exact FlashAttention-2 forward | Triton online-softmax streaming with exact causal-upper-triangle skip; fp32 accumulation; head dimensions 8–256 | Matched-fp32 attention-kernel speedup ranges from **1.15× to 6.25×** on the configurations where it helps; avoids materialising the score matrix. | At head dimensions 256 and 64 (configs 8 and 10), matched-fp32 attention is **0.58×** and **0.82×** respectively. Those end-to-end wins come from fp16 tensor cores, not the flash algorithm alone. |
| fp16 tensor-core GEMM | fp16 operands/tensor cores with fp32 accumulation and fused bias + GELU | Part of the composite 13-configuration result; supplies the dtype advantage on configs where matched-fp32 flash is neutral or slower. | bf16 failed the correctness gate on every configuration (26 recorded ledger rows), so it is not a valid substitute. |
| CUDA-graph capture | Per-shape graph on/off recipe | Removes launch overhead on the shapes for which the measured recipe selects it. | Near the graph threshold, the graph-on/off difference is within the branch's 3% noise floor; the dispatch table, not the analytic prior, is authoritative. |
| Shape-aware dispatch | `dispatch.select(shape, device) -> recipe`, calibrated per GB10 shape | Selects graph use and kernel recipes instead of applying one mechanism universally. | Prediction and measurement can disagree around the launch-fraction threshold; no stable winner should be inferred from a sub-noise difference. |
| Correctness-gated search loop | Propose -> correctness gate -> measure -> append provenance-rich ledger -> promote only beyond noise | Keeps failed dtype/kernel attempts visible and prevents timing an incorrect implementation. | It is a measurement/process optimization, not an independent speedup. |

**End-to-end result.** All 13 reported configurations pass `abs <= 0.002 OR rel <= 0.02`
against the fp32 baseline with fp32 accumulation. The table reports the latest clean,
exclusive-GPU submission run; `flash fp32` isolates the attention algorithm at matched
precision and is not the full-model speedup.

| Config | Shape `[B, S, d]` | Heads / head dim | CUDA graph | vs eager | vs `torch.compile` | Flash fp32 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | `[64, 128, 128]` | 4 / 32 | on | 2.97× | 2.14× | 3.61× ±0.1% |
| 2 | `[1, 128, 128]` | 4 / 32 | on | 2.77× | 2.13× | 2.36× ±0.2% |
| 3 | `[4, 128, 128]` | 4 / 32 | on | 2.88× | 2.25× | 3.02× ±0.2% |
| 4 | `[16, 128, 128]` | 4 / 32 | on | 2.84× | 2.09× | 2.66× ±0.4% |
| 5 | `[128, 128, 128]` | 4 / 32 | on | 3.02× | 2.12× | 4.04× ±0.6% |
| 6 | `[10,000, 128, 128]` | 4 / 32 | off | 2.66× | 1.77× | 4.10× ±0.5% |
| 7 | `[64, 128, 32]` | 4 / 8 | on | 3.35× | 1.65× | 3.54× ±0.1% |
| 8 | `[64, 128, 1,024]` | 4 / 256 | off | 3.79× | 3.80× | 0.58× ±0.3% |
| 9 | `[64, 128, 128]` | 1 / 128 | on | 2.06× | 2.06× | 1.15× ±0.0% |
| 10 | `[64, 128, 128]` | 2 / 64 | on | 2.30× | 2.30× | 0.82× ±0.2% |
| 11 | `[64, 128, 128]` | 16 / 8 | on | 5.19× | 5.18× | 4.45× ±0.3% |
| 12 | `[64, 32, 128]` | 4 / 32 | on | 2.23× | 2.23× | 1.32× ±0.6% |
| 13 | `[64, 1,024, 128]` | 4 / 32 | off | **9.80×** | **9.80×** | **6.25× ±0.2%** |
| **Geomean (13 configurations)** | — | — | — | **3.20×** | **2.62×** | — |

Configuration 14 (`[32, 100000, 1024]`) is excluded rather than treated as a win or loss:
the fp32 reference would materialise an approximately 1.3-TB score matrix per layer and
OOMs, so no same-machine reference comparison exists.

**Measurement safeguards.** The branch uses an exclusive-GPU guard, `do_bench` with a
median statistic, minimum warm-up/timed sample floors, and per-configuration dtype/kernel
decomposition. Earlier contamination and too-few-samples errors were retained in the
branch history; the clean table supersedes the earlier 2.68×/7.17× figures.

**Traceability.** `origin/marcus:SUBMISSION.md` (primary submission result),
`origin/marcus:ledger/bench_results.jsonl` (append-only rows), and
`origin/marcus:docs/hardware/gb10/` (toolchain, calibration, qualification, and prior
negative experiments).

## `ben`: RTX 4070 Ti SUPER search and measurement discipline

**Optimization objective.** Search a broad kernel/design space without promoting timing
noise, a weak baseline, or an incorrect implementation. This is a working empirical lane:
its branch documentation explicitly labels the CUDA results as unqualified and keeps them
outside `research/archive/` until a vendor qualification hierarchy is ratified.

**Historical end-to-end progression.** These are geometric-mean speedups versus the
branch's compiled baseline across the announced matrix. They show the measured benefit
attributable to successive *composite candidates*, not isolated effects that can be added
together.

| Generation / candidate | New mechanism | Geomean | Interpretation |
| --- | --- | ---: | --- |
| g1 `v1_fused_graph` | Fused QKV, fp16 GEMM cache, static CUDA graph | 0.79× | Initial composite regressed. |
| g2 `v2_fp16_flash` | fp16 Q/K/V plus mask elision so FlashAttention qualifies | 1.41× | First net improvement. |
| g6 `v6_fp16_gelu` | Removes one non-accumulating fp16 round trip | 1.69× | Historical honest headline after switching from eager to compiled baseline. |
| g9 `v9a_compiled_core` | Lets Inductor fuse the branch algorithm | 2.68× | Compiler fusion is a major contributor; it must be reported as part of the composite. |
| g12 `v12_graph_over_compile` | Adds branch-owned graph capture after compilation | 2.71× | Small incremental improvement. |
| g17 `v17_dispatched_megakernel` | Recombines a hand-written FFN kernel | 2.76× | Shape-specific benefit, not a universal megakernel claim. |
| g18 `v18_capture_insurance` | Makes capture independent of caller allocation context | 2.77× | Primarily a robustness improvement with a small aggregate change. |
| g23 `v23_single_tile_attn` | Uses hand-written attention where scores fit on chip | 3.02× | Attention specialization helps only in the eligible shape regime. |
| g26 `v26_causal_correct` | Honors the evaluator's causal flag | 3.10× | A correctness repair; the performance figure must not conceal the prior wrong-answer path. |
| g34 `v34_launch_bound` | Removes 16 of 36 launches per forward | **3.25×** | Latest reported aggregate milestone in `REPORT.md`. |

**Current candidate and recent result.** `v43_replicated_tile` is the current submitted
candidate. Its change is deliberately narrow: reduce replicated tile-sweep observations
per arm before selecting a tile, so one-sided timing contamination does not decide the
winner. Its value is primarily stability/correctness of dispatch: it preserves the real
config-2 tile choice and fixes config-3 plan instability. A replicated comparison against
`v40` reports a **+0.0109 weighted-score** change, entirely from config 2; this is *not* a
geometric-mean speedup and must not be quoted as one. The candidate's own A/B controls
place unresolved byte-identical-code variation at 0.9811×–1.0046×, so non-replicated
sub-percent deltas are correctly treated as noise.

**Technique-level findings that constrain the benefits.**

| Technique / hypothesis | Supported benefit | Limit or decision |
| --- | --- | --- |
| fp16 residual stream | About 1.4× faster in the investigated path | Rejected: fails 11 of 13 configurations; the fp32 residual is load-bearing. |
| Padding-aware mask reasoning | At padding ratio 0.5, a correct right-padded-causal proof restored a 5.85× path versus 2.86× before the repair | Only applies when the stated causal/padding conditions hold; zero-padding-only results were misleading. |
| L2 weight persistence | Positive control moved the kernel 42.7%, validating the instrument | Hypothesis falsified: persistence itself moved performance only 0.25%; the 768-KiB weights were negligible beside a 327-MB activation stream. |
| Single-tile / looped attention dispatch | Can choose a hand-written path in shapes with sufficient on-chip residency | Declines to the vendor implementation when register residency or margin is inadequate. |
| Triton projection GEMMs and fused FFN | Per-shape dispatch can select Triton at sites that clear its margin; FFN fusion is used only past its amortization crossover | Vendor paths remain selected when a measured tile does not clear the margin. |
| CUDA graphs and launch fusion | Reduces launch cost on one-wave/launch-bound shapes | Does not help throughput-bound multi-wave shapes; capture and allocator state require explicit verification. |

**Measurement safeguards and known caveats.** Correctness precedes timing; the branch uses
locked tolerances (`rtol=0.02`, `atol=0.002`), one configuration per subprocess, an
append-only commit-keyed ledger, GPU exclusivity, and minimum-of-N timing because clocks
cannot be locked under WSL2. It measured an approximately ±7% noise floor, uses screens
only to reject obvious losses, and records full sweeps as the decision evidence. The branch
also documents that isolated and interleaved protocols can disagree and that its benchmark
harness and the graded evaluator differ on small shapes. Therefore the historical aggregates
and the current candidate-selection evidence must remain attached to their stated protocol.

**Traceability.** `origin/ben:REPORT.md` (historical aggregates and findings),
`origin/ben:docs/findings/49-the-shipping-decision.md` (current candidate history),
`origin/ben:docs/findings/54-a-floor-not-a-vote.md` (replication result),
`origin/ben:docs/loop/method.md` (measurement rules), and
`origin/ben:bench/results.jsonl` (append-only rows).

## `Aekas`: fused-kernel implementation with no empirical claim

**Optimization objective.** Prepare an Intel/XPU-oriented transformer path and a
measurement protocol without inventing a result on a CPU-only machine.

| Technique prepared | Intended benefit | Evidence status |
| --- | --- | --- |
| Triton fused LayerNorm + residual | Reduce separate normalization/residual operations and associated launches/traffic. | Implementation exists in `ratchet/kernels/triton_fused_ops.py`; no correctness or timing result. |
| Fused GELU epilogue | Avoid a separate activation pass after the relevant linear work. | Implementation only; no measured benefit. |
| Shape-aware attention and cached fused QKV in `OptimizedTransformer` | Select an appropriate attention path and avoid repeated projection work. | Implementation only; no measured benefit. |
| ABBA/BAAB benchmark runner | Reduce ordering/thermal bias using 20 warm-up calls, 10 alternating blocks of 30 calls, device events, and a host-timer cross-check. | Protocol is defined but unrun; `triton_fused.json` is explicitly `validation: "unvalidated"`. |

The branch's defensible result is the absence of a result: the environment had no PyTorch,
`EVT-000001` is a no-run record, and the paper reports zero empirical events. This is an
important provenance outcome, not a failed benchmark. A future qualified runtime must run
the full correctness matrix and the stated protocol before any row above can acquire a
speedup, latency, or memory claim.

**Traceability.** `origin/Aekas:research/archive/events/EVT-000001.json`,
`origin/Aekas:benchmarks/runners/configurations/triton_fused.json`,
`origin/Aekas:benchmarks/runners/run_transformer_experiment.py`, and
`origin/Aekas:ratchet/kernels/triton_fused_ops.py`.

## `brian`: synthetic tuning-difficulty forecasting

**Optimization objective.** Allocate tuning effort toward configurations predicted to be
hard or expensive, rather than directly optimizing a transformer implementation.

**Protocol.** The standardized long run uses 100 seeded trials for each hardware-profile /
precision scenario. The outcome is Spearman correlation between a method's predicted
difficulty ranking and synthetic ground truth; higher is better. This is a deterministic
search-strategy experiment, not an accelerator benchmark.

| Method | CUDA / fp32 | CUDA / fp8 | HIP / fp32 | HIP / fp8 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Failure-weighted PageRank | **0.432** | **0.335** | **0.433** | **0.333** | Best ranking correlation in every reported scenario. |
| Tuning pressure | 0.393 | 0.271 | 0.399 | 0.263 | Positive but weaker than risk-aware PageRank. |
| Plain PageRank | -0.517 | -0.517 | -0.526 | -0.526 | Anti-correlated with the target; worse than a random ordering. |
| Random baseline | 0.008 | 0.008 | 0.007 | 0.007 | Near-zero expected correlation. |

The benefit is therefore *better search prioritization*, not a latency factor: adding
failure/risk information to graph propagation outperforms plain structural centrality and
remains positive under fp8, although the correlation attenuates there.

**Traceability.**
`origin/brian:research/experiment_summaries/standardized_method_benchmark_long.json` and
`origin/brian:ratchet/experiments/standardized_method_benchmark.py`.

## Cross-branch conclusions and reporting rules

1. **The strongest directly measured end-to-end claim in this comparison is `marcus`:**
   3.20× geomean versus its official eager baseline and 2.62× versus `torch.compile` on
   GB10, across 13 correct configurations. The earlier 2.68× value is not the latest clean
   submission result and is not used as the headline here.
2. **`ben` contributes both measured candidate progress and a reusable methodology.** Its
   3.25× g34 aggregate is a branch-specific historical compiled-baseline result; g43
   improves stable dispatch selection rather than establishing a new global speedup.
3. **No technique is universally beneficial.** Flash attention can lose at matched fp32;
   fp16 residuals and bf16 can violate correctness; L2 persistence can be immaterial;
   graph capture and fusion need shape- and regime-aware dispatch.
4. **`Aekas` must remain implementation/no-run work until qualified hardware runs.**
   `brian` must remain synthetic ranking evidence until a separate experiment connects its
   prioritization to realized hardware-search savings.
5. **For future additions, record the same fields used here:** optimization goal,
   technique and dispatch condition, baseline, hardware/software environment, correctness
   status, metric and aggregation, result, negative result/trade-off, evidence class, and
   exact branch artifact. This keeps a technique's benefit traceable rather than merely
   descriptive.
