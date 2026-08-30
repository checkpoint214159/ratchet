# 42 — Our harness measures a different quantity than the grader, and it inverted two signs

**Date:** 2026-08-30. **Found by:** the launch-bound research agent, confirmed here with
`bench/end_to_end.py`, which runs the graded benchmark's own `main()` unmodified.

## Three protocols, and we were using the wrong one

    GRADED (benchmarks/reference/..., what we are scored on)
        interleaved ABBA/BAAB rounds -- "Alternate measurement order to reduce
        thermal/clock-order bias" -- samples POOLED across rounds, then
        speedup = baseline.median / optimized.median

    CLAUDE.md's prescription
        minimum-of-N, interleave candidate/baseline (clocks are not lockable under WSL)

    bench/run_matrix.py, what actually produced all ~600 ledger rows
        min(median_ms(base), median_ms(base)) ... then correctness ... then
        min(median_ms(cand), median_ms(cand))
        NOT interleaved; the candidate is compiled and autotuned BETWEEN the two arms

All three are different. The row even records `interleaved: false` honestly; nobody had
checked what that cost.

## It is not a technicality. It inverted the sign.

`v34_launch_bound` against `v26_causal_correct`:

    cfg    our ledger        graded protocol
      1    +6.1% WORSE       -0.8% better
      2   -26.3% better     -23.8% better
      9    +5.6% WORSE       -6.9% better
     10    -0.8% better      -0.4% better

Configs 1 and 9 flipped. The research agent predicted exactly this before the check ran,
from the kernel census: v34 launches **strictly fewer kernels** on 1, 8 and 9, so a
measured regression there was mechanically implausible.

The cause is the ordering. `run_matrix` times the baseline, then BUILDS the candidate --
`torch.compile`, Inductor autotuning, Triton JIT, all of which run the GPU hard -- and
only then times the candidate, on a hotter device with no clock lock. The graded harness
interleaves specifically to cancel that. Our comment says arms are timed in isolation to
avoid finding 05's co-residency spill, which is correct and necessary; the mistake was
concluding that isolation therefore made ordering irrelevant.

## What this invalidates, and what it does not

**Ranking between candidates measured in separate runs is unsafe on sub-millisecond
configs**, which is precisely where all remaining score lives (configs 9, 4, 12, 2, 10, 1
— six of the seven with headroom). The large configs are far less exposed: L42 measured
>1 ms rows reproducing within 0.6%, and config 6/8/13 deltas have been stable across
re-measurements all session.

**Correctness rows are unaffected** — accuracy is checked per trial against a fresh
reference, not by comparing runs.

The ledger is append-only and the rows stay. They are a faithful record of a protocol we
now know differs from the graded one.

## L52 — Measure with the protocol you will be scored by, or prove they agree

`bench/end_to_end.py` was written days ago for exactly this purpose. Its own docstring
says: *"our numbers and the graded numbers are produced by different protocols, and
nobody has ever checked they agree."* It was then never run systematically, and the
project spent a full session ranking candidates on a quantity that was not the score.

Two hours before this check, an executor reported that its candidate won on min-of-N and
not on the median, and asked for a better measurement. That was the signal, and the
correct response was not to argue about noise floors — it was to run the grader.

**When a harness of your own differs from the one that scores you, the burden is on you
to demonstrate agreement, per config, before trusting a single ranking.** A difference
you have written down in a comment is still a difference.
