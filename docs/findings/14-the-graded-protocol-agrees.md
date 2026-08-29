# Finding 14 — The benchmark's own protocol agrees with ours, and explains L11

Recorded 2026-08-29. Tool: `bench/end_to_end.py`.

## Why this needed checking

Every number in `bench/results.jsonl` came from **our** timing loop, which deliberately
differs from the benchmark's. Ours times each arm while it is the only model resident,
because holding both inflated config 6's baseline 4.1x through host-memory spill
(finding 05). The benchmark's `benchmark_models` warms up **both** models and keeps both
resident for the whole run.

Two protocols, and the graded number comes from theirs. Nobody had checked they agree.

## They agree

Running the benchmark's own `main()` unmodified, with `UserOptimizedTransformer`
monkeypatched to v9a (the pinned file untouched):

| config | benchmark reports | our runner | delta |
|---|---|---|---|
| 1 | 7.024x | 7.06x | 0.5% |
| 6 | 6.247x | 6.52x | 4.2% |
| 12 | 13.476x | 12.63x | 6.7% |
| 13 | 33.307x | 33.80x | 1.5% |

All within or near the 3% noise floor, and **config 6's baseline reads 448.9 ms under the
benchmark's co-resident protocol against 448.4 ms under our isolated one** — the spill
that made finding 05 does not reproduce here, because v9a's peak memory is far lower than
the v1-era candidate that triggered it (no manual static graph buffer, and chunking caps
the working set).

So the measurement chain is validated end to end: our numbers transfer to the protocol
that will actually be graded. Accuracy passes on all four.

## The incidental discovery, which corrects L11

Every compiled run prints:

```
torch/_inductor/utils.py:1436] Not enough SMs to use max_autotune_gemm mode
```

**Inductor disables GEMM autotuning on this card.** 66 SMs is below its threshold.

That reframes finding L11 entirely. v9a (`max-autotune`) and v9b (`reduce-overhead`) came
out 0.3% apart and I concluded "the autotuning buys nothing on this matrix". The truer
statement is narrower and more useful: **max-autotune's distinguishing feature was never
active**, so the comparison measured two modes that had silently collapsed into nearly the
same thing.

The practical conclusion survives — use `reduce-overhead`, it is cheaper for identical
results — but the *reason* is different, and the difference matters for transfer: on a
datacenter GPU with enough SMs, `max-autotune` would actually autotune, and the sibling
comparison would have to be re-run. A null result explained by "it does nothing"
generalizes; one explained by "it was disabled" does not.

## Method note

This is the fourth audit-driven finding in a row, and the first one to check the
*measurement apparatus itself* against the thing it is a proxy for. Worth stating as a
rule: **when you replace a harness's protocol with your own, you owe a comparison against
the original.** Ours diverged for a defensible reason and happened to agree; that was not
knowable without running it.
