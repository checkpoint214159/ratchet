# Finding 06 — The first search run found noise, and told us so

Recorded 2026-08-29. Loop output: 14 points, 0 infeasible, on configs 1, 2, 6, 13.
Code: `bench/loop.py`, `bench/candidates/v4_tunable.py`.

## What the loop reported

```
seed  geomean=8.860x  {use_graph: True, target_occupancy: 0.5, live_tensors: 3}
best  geomean=9.101x  {use_graph: True, target_occupancy: 1.0, live_tensors: 6}
```

A 2.7% improvement. **It is not real**, and the reason is instructive enough to be worth
more than the win would have been.

## The search space was degenerate

`solve_chunk` uses the two parameters only as a quotient:

    chunk = (L2_bytes * target_occupancy) / (seq*dim*bytes * live_tensors)

So `occupancy/live_tensors` is the only quantity that matters. The seed and the "best"
point have ratios `0.5/3 = 0.1667` and `1.0/6 = 0.1667` — **identical**. Both compute
`chunk = 128` for config 6. The optimizer moved through the space and arrived back at
exactly the configuration it started from, wearing different coordinates.

## Which accidentally handed us a noise floor

Because the space is degenerate, several evaluated points are unintentional
**replicates** — different coordinates, identical computed behaviour. Grouping the run's
own results by effective ratio:

| ratio | chunk (cfg 6) | measured geomeans | spread |
|---|---|---|---|
| 0.125 | 96 | 8.622, 8.746 | 1.4% |
| 0.1667 | 128 | 8.860, **9.101** | **2.7%** |
| 0.25 | 192 | 8.665, 8.628, 8.891 | 3.0% |

**The run-to-run spread on identical configurations is 1.4–3.0%.** The claimed
improvement is 2.7% — sitting exactly inside the replicate spread at its own ratio. It is
measurement noise that the search mistook for signal, on a card whose clocks cannot be
locked.

## What the run did establish, decisively

One factor dominates everything else in the space:

| `use_graph` | geomean range |
|---|---|
| True | 8.21x – 9.10x |
| False | 4.66x – 4.89x |

**CUDA graph capture is worth ~1.85x** and is far outside the noise floor. Every other
axis in the space moves the result by less than the measurement error. The chunk ratio
shows a weak real trend (0.5 → 8.21x is genuinely worse than 0.125–0.25 → ~8.7x), but
the differences among the good ratios are not resolvable at this sample count.

## Three corrections this forces

1. **Collapse the degenerate axes.** Two parameters that only ever appear as a quotient
   are one parameter. Searching both wastes budget re-measuring the same point and
   manufactures fake improvements.
2. **A promotion rule needs a margin.** The loop took any `fitness < best` as an
   improvement. It must instead require the gain to exceed the measured noise floor —
   the project's own architecture doc says promotion requires non-overlapping confidence
   intervals, and the loop was not honouring the rule the rest of the system is built on.
3. **Replicate deliberately, not by accident.** The noise floor here was a gift from a
   bug. It should be measured on purpose: re-evaluate the incumbent every N rounds and
   keep a running estimate of run-to-run spread.

## Why this is reported rather than quietly fixed

The dossier's skeptic paper (arXiv 2602.16805, *Simple Baselines are Competitive with
Code Evolution*) exists for exactly this case: ablate before claiming the machinery
produced the win. Here the machinery produced a 2.7% "win" that is a re-measurement of
its own starting point. Had the loop run unattended for an hour and reported a series of
such gains, the result would have looked like steady progress and been entirely fictional.

That is the failure mode this project was built to catch, and it caught it on the first
run of its own search loop.

---

## The re-run: the search validated the calibration rather than beating it

With the axes collapsed and the 3% promotion margin in place, the loop was re-run over
the same four configs (1, 2, 6, 13). It evaluated 10 of the 12 points the space contains
and **promoted nothing**. The best point is the seed.

| `chunk_ratio` (with CUDA graph) | geomean |
|---|---|
| 0.0625 | 7.036x |
| 0.125 | 8.666x |
| **0.1667** | **8.997x**  <- the seed, and the analytic L2 derivation |
| 0.25 | 8.880x |
| 0.5 | 8.327x |

Without the graph, the same ratios yield 2.81x - 5.08x.

Two things this establishes:

**The curve is real and physically shaped.** It peaks at 0.1667 and falls off in both
directions — too-small chunks lose parallelism, too-large chunks lose the L2 residency
the chunking exists to buy. This is not a flat space with noise on top.

**The peak sits exactly where the analytic calibration put it.** `solve_chunk` derives
0.1667 from measured cache capacity with no search at all. The optimizer explored the
space around it and found nothing better by more than the noise floor. The hand-derived
predicate was already optimal.

That is a **null result for the search and a positive result for the calibration**, and
it is the ablation the dossier's skeptic paper (arXiv 2602.16805) demands before claiming
evolutionary machinery produced a win. Here it demonstrably did not: the win came from
the roofline reasoning, and the search's contribution was to confirm it and to refuse to
promote 2.7% of noise on top.

**A methodological note for the next run.** The space holds 12 points and the loop
visited 10 of them, repeatedly bouncing off already-cached points during restarts.
First-improvement iterated local search is designed for spaces too large to enumerate;
below a few hundred points, exhaustive evaluation is cheaper and gives every point a
confidence interval instead of one visit. ILS should be reserved for the space that
actually needs it — the one that opens up when the loop can propose kernel source rather
than turn existing knobs.
