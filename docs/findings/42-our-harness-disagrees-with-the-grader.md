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

---

## Addendum — the graded harness cannot rank either, and the noise has a shape

Running `bench/end_to_end.py` on three candidates across nine configs shows the problem is
not "our harness is wrong, use theirs". **The BASELINE arm — byte-identical unmodified
reference code in all three runs — moves like this:**

    cfg  8   16.2580  16.2652  16.2662     spread  0.1%
    cfg  5    3.2461   3.2737   3.2737             0.9%
    cfg  1    1.8848   1.7958   1.8396             5.0%
    cfg  9    1.7316   1.6794   1.5928             8.7%
    cfg  4    1.8381   2.0054   2.2486            22.3%
    cfg  2    2.1646   2.4544   1.8427            33.2%
    cfg  3    1.8024   2.5104   2.0343            39.3%

**The noise scales inversely with config size, and the worst rows are exactly the ones
carrying all the remaining score.** Meanwhile the OPTIMIZED arm is stable to the last
digit: v34 and v35 both read 0.0481 on config 2 and 0.0922 on config 12, differing only
where their predicates differ.

So the reported `speedup` inherits a noisy denominator and is unusable for ranking, while
the candidate's own time is sound. **Rank candidates by their optimized time against a
FIXED reference, never by a per-run speedup ratio.** The score is unaffected — the graders
compute one ratio per submission, they do not compare two of ours.

The g36 executor independently diagnosed the mechanism: **round 1 of 100 timed calls reads
932.9 us on config 1 where rounds 2-3 read 250.9 us stable to 0.1 us** — roughly 130 calls
of settling after CUDA-graph capture, against the harness's 20 warmup iterations. Both arms
pay it, so a single submission's score is fair; a five-sample median of it cannot separate
two candidates 5% apart on a 0.25 ms row.

**The protocol that did work** (used for v36, +0.082 of weighted_score with no regression):
parent and child ABBA-interleaved, both models resident, cold round discarded, min of four
— with configs 8 and 13 running byte-identical code as an in-run control, establishing the
floor at +/-0.4%. Configs 1, 9 and 10 have identical GEMM shapes and agreed to 0.15pp,
which is the check that the protocol is not measuring itself.

## L53 — A tuner that times two arms in sequence is a benchmark, and inherits every benchmark's bugs

From the g36 executor, which found it the hard way twice: its predicate used `do_bench`
(which flushes L2 and pays a launch) to choose tiles for kernels that run **L2-hot inside a
replayed graph**, and its first `plan()` call in a process measured `F.linear` at 306 us
where a clean process reads 21.5 us — cuBLASLt one-time setup captured inside the timing
window, reporting a fake 17.6x.

**Interleave, discard the cold round, and use a timer whose regime matches the call site's
— or write in the code why not.**
