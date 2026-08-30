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
| [19](19-the-input-scale-tail.md) | Every candidate fails at input_scale=0.01; our 6% tolerance margin is not enough | 2026-08-29 |
| [18](18-lineage-invariant-sweep.md) | Sweeping the lineage for staleness found 3 bugs the accuracy suite could not see | 2026-08-29 |
| [17](17-the-empty-graph.md) | v12 could replay an empty graph and return stale output; v13 verifies or falls back | 2026-08-29 |
| [16](16-dynamo-guard-cost.md) | Dynamo guards cost 22.5us/call; reclaiming them is +7.9%, and our regime labels mispredicted | 2026-08-29 |
| [15](15-ablating-the-inherited-stack.md) | Chunking is now dead weight; fp16 cache is worth up to 5x | 2026-08-29 |
| [14](14-the-graded-protocol-agrees.md) | The benchmark's own timing agrees with ours; and max-autotune was never active | 2026-08-29 |
| [13](13-we-are-fp32-only.md) | Every candidate assumes an fp32 model; fp16 crashes or fails the gate | 2026-08-29 |
| [12](12-the-baseline-was-wrong.md) | **Our baseline was eager. Honest speedup is 1.69x, not 7.2x, and we lose 2 configs.** | 2026-08-29 |
| [33](33-the-weights-were-already-in-l2.md) | L2 persistence is inert: the weights hit L2 by natural reuse, saving 94.8% of the worst case already | 2026-08-30 |

Related, and deliberately elsewhere:

- `bench/matrix.py` — the matrix as executable data; findings cite it, never restate it.
- `bench/results.jsonl` — the measurement rows behind any number quoted here.
- `research/` — the fail-closed evidence archive. Nothing in `findings/` is ratified
  evidence under that hierarchy; see `bench/README.md` for the boundary.
