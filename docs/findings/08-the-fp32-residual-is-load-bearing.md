# Finding 08 — The fp32 residual stream is load-bearing, and costs ~1.4x

Recorded 2026-08-29. Candidate: `bench/candidates/v5_fp16_resid.py`. Rows:
`bench/results.jsonl` @ `c328814`.

## The experiment

Profiling put fp32<->fp16 conversion at 12.8-26.8% of candidate kernel time. v2-v4
round-trip roughly six times per layer. v5 removed all of them: the residual stream stays
fp16 for the whole stack, with cached fp16 LayerNorm weights and GELU in fp16, one
downcast at entry and one upcast at exit.

## The result: fast, and decisively wrong

**11 of 13 configs fail correctness.** Failing `max_abs` runs 6.52e-3 to 1.06e-2 against
the 2.0e-3 budget — **3.3x to 5.3x over**, not marginal.

Where it did pass, the prize was real:

| config | v3 | v5 | gain |
|---|---|---|---|
| 2 | 15.22x | **21.85x** | 1.44x |
| 3 | 13.33x | **17.87x** | 1.34x |

So the conversions genuinely cost ~1.4x on the launch-bound configs. The fp32 residual is
not incidental overhead that a tidier implementation could remove — **it is buying the
precision the tolerance requires**, and there is no version of this trade that passes.

## Why configs 2 and 3 passed with errors far over atol

This is the subtlety worth carrying forward. Config 2 passed while reporting
`max_abs = 5.16e-3`, which is 2.6x the absolute budget.

The benchmark's criterion is **OR**, not AND: an element passes if
`|diff| <= atol` **or** `|diff| <= rtol * |ref|`. An absolute error of 5.16e-3 is
forgiven wherever the reference value exceeds `5.16e-3 / 0.02 = 0.258`, because the 2%
relative bound covers it there.

So configs 2 and 3 pass not because they are more accurate but because their large errors
happen to land on large-magnitude reference elements. **`max_abs` alone does not predict
pass/fail** — the joint distribution of error and reference magnitude does. Any future
report quoting `max_abs` as a safety margin is quoting the wrong statistic.

It also means this result is fragile in the wrong direction: configs 2 and 3 pass by
luck of distribution, not by design, and a different seed could flip them.

## What this closes and what it opens

**Closed:** the fp16 residual stream. Not "needs tuning" — 3.3x over budget is not a
tuning distance.

**Open, and the natural next experiment:** the six conversions per layer are not equally
load-bearing. The residual accumulation is what compounds; the GELU round-trip
(`h.float() -> gelu -> .to(fp16)`) touches a single elementwise op and does not accumulate
across layers. Isolating the subset that can be removed while keeping the fp32 residual is
a strictly smaller, testable change — and this run establishes the ceiling it is working
against (~1.4x) and the wall it must not hit.

## Method note

The candidate was committed before measurement, so its rows carry a sha that describes the
exact source that produced them, and it failed **correctness before any timing was taken**
— the harness never timed 11 of the 13 configs because a candidate that fails the gate is
not timed. The two passing configs were timed normally. Failures are recorded as rows with
`status="incorrect"` rather than discarded, because "this is where the precision floor is"
is the most reusable thing this run produced.
