# docs/findings/

Durable notes from working sessions. One file per finding, numbered in the order they
were established.

The rule that makes these worth keeping: **a finding records what was measured and how,
not what was believed.** Where a number appears, it was produced on the machine named in
the note. Where something is inferred rather than measured, it says so.

| # | Finding | Established |
|---|---|---|
| [00](00-learnings.md) | **Running learnings — the loop's long-term memory** | ongoing |
| [01](01-competition-matrix.md) | The announced shape matrix and what it implies | 2026-08-29 |
| [02](02-allowed-techniques.md) | What the rules permit, decided against measurement | 2026-08-29 |
| [03](03-baseline-measurements.md) | Baseline and candidate across the real matrix | 2026-08-29 |
| [04](04-the-flash-attention-that-never-was.md) | v1 never reached FlashAttention; the fix is worth 3.11x -> 5.58x | 2026-08-29 |
| [05](05-two-measurement-artifacts.md) | Two harness bugs that produced plausible wrong numbers | 2026-08-29 |
| [06](06-the-search-found-noise.md) | The first search run improved by 2.7% -- inside its own 3% noise floor | 2026-08-29 |
| [07](07-environment-gaps.md) | pytest import shadowing, and git 2.34 vs the 2.38 merge-tree the workspaces need | 2026-08-29 |
| [08](08-the-fp32-residual-is-load-bearing.md) | fp16 residual stream is 1.4x faster and fails 11/13 configs; v6 resolves it | 2026-08-29 |
| [09](09-config-14-runs.md) | Config 14 runs in 3.18 GB where the reference OOMs -- with a stated limit | 2026-08-29 |
| [10](10-layernorm-fusion-buys-nothing.md) | LayerNorm downcast fusion: +2.0% (below noise) and fails config 6 | 2026-08-29 |
| [11](11-the-padding-blind-spot.md) | Every prior number used padding=0; v8 recovers the padded path with a proof | 2026-08-29 |
| [12](12-the-baseline-was-wrong.md) | **Our baseline was eager. Honest speedup is 1.69x, not 7.2x, and we lose 2 configs.** | 2026-08-29 |
| [13](13-we-are-fp32-only.md) | Every candidate assumes an fp32 model; fp16 crashes or fails the gate | 2026-08-29 |
| [14](14-the-graded-protocol-agrees.md) | The benchmark's own timing agrees with ours; and max-autotune was never active | 2026-08-29 |
| [15](15-ablating-the-inherited-stack.md) | Chunking is now dead weight; fp16 cache is worth up to 5x | 2026-08-29 |
| [16](16-dynamo-guard-cost.md) | Dynamo guards cost 22.5us/call; reclaiming them is +7.9%, and our regime labels mispredicted | 2026-08-29 |
| [17](17-the-empty-graph.md) | v12 could replay an empty graph and return stale output; v13 verifies or falls back | 2026-08-29 |
| [18](18-lineage-invariant-sweep.md) | Sweeping the lineage for staleness found 3 bugs the accuracy suite could not see | 2026-08-29 |
| [19](19-the-input-scale-tail.md) | Every candidate fails at input_scale=0.01; our 6% tolerance margin is not enough | 2026-08-29 |
| [20](20-the-rubric-backtest.md) | Backtesting the proposal rubric: three defects, found before it spent a GPU minute | 2026-08-29 |
| [21](21-the-clade-sampler-was-flat.md) | The node sampler was flat, and the obvious fix made it worse | 2026-08-29 |
| [22](22-the-sm-veto-was-real-and-worthless.md) | The 68-SM veto was real, correctly diagnosed, and worth nothing | 2026-08-30 |
| [23](23-the-head-dim-8-premise-was-false.md) | The head_dim=8 premise was false, and it had steered the project for a week | 2026-08-30 |
| [24](24-the-suite-was-masking-a-live-bug.md) | Four candidates carry a silent-wrong-answer bug, and the test suite was hiding it | 2026-08-30 |
| [25](25-the-megakernel-amortizes-over-tokens.md) | The first hand-written kernel wins where there is work and loses where there is not | 2026-08-30 |
| [26](26-the-contention-guard-that-cannot-see.md) | One GPU, several agents: the guard works, the detector is blind, and v15 is suspect | 2026-08-30 |
| [27](27-two-verifications.md) | Re-measuring v15 clean, and a fix whose value the benchmark cannot show | 2026-08-30 |
| [28](28-the-tree-was-a-chain.md) | The evolutionary tree was a chain, and I rebuilt the exact degeneracy I documented | 2026-08-30 |
| [29](29-two-probes-two-known-hazards.md) | v19 is flat, and both probes that said otherwise failed by documented hazards | 2026-08-30 |
| [30](30-fp16-accumulation-has-no-window.md) | fp16 MMA accumulate is a real 1.62x and buys 1.000x: the speed and accuracy conditions are the same variable, pointing opposite ways | 2026-08-30 |
| [31](31-single-tile-attention.md) | A hand-written attention kernel beats flash where the score matrix fits, and the predicate is not the one either proposal guessed | 2026-08-30 |
| [32](32-the-causal-default.md) | **Every candidate since v5 was wrong on the harness's own default** | 2026-08-30 |
| [33](33-the-weights-were-already-in-l2.md) | L2 persistence is inert: the weights hit L2 by natural reuse, saving 94.8% of the worst case already | 2026-08-30 |
| [34](34-the-projection-fused-into-attention.md) | The QKV projection fused into attention, and the byte count that decides it | 2026-08-30 |
| [35](35-the-outprojection-gather-does-not-exist.md) | The out-projection's head-major gather does not exist, and what replaced it | 2026-08-30 |
| [36](36-head-dim-8-is-real-and-small.md) | head_dim=8 is a real 1.4x and an end-to-end nothing, and the cost is the layout | 2026-08-30 |
| [37](37-the-outprojection-epilogue.md) | The out-projection in the attention kernel's epilogue, and the accuracy claim it corrects | 2026-08-30 |
| [38](38-the-output-copy-and-what-can-see-it.md) | The graph's output copy, and the fact that nothing we run can see it | 2026-08-30 |
| [39](39-the-launch-floor.md) | Thirty-six kernels on every config, and what a kernel costs when it computes nothing | 2026-08-30 |

Numbers are identifiers, not an ordering guarantee. Findings 30 and 33 were each claimed concurrently by several agent branches; on merge the earliest-cited claimant kept the number and the rest were reassigned 35-39, so a few numbers run slightly out of chronological order.

| [33](33-config-14-protocol.md) | **Config 14: the reference is infeasible on any hardware (18.63 TiB); we compute it, verified at the real S; and there is still no speedup.** Supersedes [09]. | 2026-08-30 |

Related, and deliberately elsewhere:

- `bench/matrix.py` — the matrix as executable data; findings cite it, never restate it.
- `bench/results.jsonl` — the measurement rows behind any number quoted here.
- `research/` — the fail-closed evidence archive. Nothing in `findings/` is ratified
  evidence under that hierarchy; see `bench/README.md` for the boundary.
