# Finding 10 — LayerNorm downcast fusion spends the precision budget and buys nothing

Recorded 2026-08-29. Candidate: `bench/candidates/v7_fused_norm.py` @ `fc2e159`.
Rows: `bench/results.jsonl`.

## The experiment

`native_layer_norm` was 9.7-16.8% of candidate kernel time and `add` a further 2.5-9.6%,
both pure memory traffic — the last named bottleneck in the profile. v6 must normalize in
fp32 (the harness's `nn.LayerNorm` holds fp32 parameters) and then downcast in a separate
kernel. v7 caches fp16 norm weights so the downcast folds into the norm's own epilogue:
one fewer full pass over the activation per norm, two per layer. The fp32 residual was
left untouched, per finding 08.

## The result: rejected on both counts

**Speed: +2.0% geomean on the 12 configs both passed — below the 3% noise floor.**
Not an improvement by this project's own promotion rule.

**Correctness: config 6 FAILS**, at `max_abs` 2.013e-3 against a 2.0e-3 budget. Over by
0.65%.

And the margin everywhere else is gone:

| config | v6 `max_abs` | v7 `max_abs` | budget used |
|---|---|---|---|
| 1, 4, 5, 9, 10, 11, 12 | 1.31–1.59e-3 | **1.938e-3** | **96.9%** |
| 6 | 1.716e-3 (85.8%) | **2.013e-3** | **100.7% — FAIL** |
| 7 | 1.882e-3 | 2.317e-3 | 115.9% (passes only via the OR rule) |

**The clustering is the diagnostic.** Seven different configs land on *exactly*
1.9384026527404785e-3. That is not seven coincidences — it is the fp16 rounding of the
normalized output hitting a fixed representation limit, independent of shape. The change
does not add error proportional to the work; it adds a floor.

## Why this is a clean reject rather than a tuning problem

Correctness is a gate, not a term in the objective. v7 is nominally faster and **scores
nothing**, because a candidate that fails any config has not produced a result. Its
weighted score is 2.666 against v6's 2.809 — lower, despite the higher geomean on the
subset it survived, precisely because the failed config counts against it.

There is also no version of this to tune toward. The error is a representation floor, not
a magnitude that a smaller step would reduce, and it consumes 97% of budget for a gain
inside the noise. The correct action is to discard it and keep v6.

## What it says about the remaining headroom

This exhausts the bottlenecks the profile named. The three that were measurable are now
resolved:

| bottleneck | outcome |
|---|---|
| launch overhead (36% of wall) | **taken** — CUDA graphs, worth ~1.85x |
| fp32<->fp16 conversions (12.8-26.8%) | **partly taken** — GELU round-trip free (v6); residual immovable (v5) |
| LayerNorm + add (12-26%) | **not available** — costs the precision budget for noise |

What is left is not elementwise traffic. The remaining time sits in matmuls already
running at 89-92% of this card's measured ceiling, and in the attention kernels the vendor
already tuned. Further gains need an architectural change — a different decomposition, not
a cheaper spelling of the same one — which is beyond what the parametric loop can propose.
