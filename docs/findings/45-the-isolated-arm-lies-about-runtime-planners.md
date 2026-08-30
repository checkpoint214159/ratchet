# 45 — The isolated protocol reports 2-4x errors on candidates that plan at run time

**Date:** 2026-08-30. **Found by:** v36's swept score contradicting its own branch measurement.

## The contradiction

`v36_gemm_gelu` measured **+0.082 of weighted_score (2.489 -> 2.571)** on its branch, with
ABBA-interleaved timing, both models resident, the cold round discarded, min of four, and
byte-identical configs as an in-run control. Its sweep through `bench/run_matrix.py`
returned **2.1893** — far below its parent v34's 2.4973.

Both arms are on the same row, because `run_matrix` now records an interleaved second
opinion (finding 42). They disagree by up to 4x:

    cfg    isolated   interleaved   ratio
      4     0.3021       0.0819     3.69x
      7     0.3246       0.0778     4.17x
     12     0.1802       0.0891     2.02x
      9     0.2243       0.2243     1.00x
      8     6.5761       6.5843     1.00x
     13     3.2983       3.3055     1.00x

The interleaved numbers reproduce the branch measurement. The isolated ones do not.

## Why, and it is specific to this candidate class

v36's `plan()` predicate **times the vendor GEMM against swept Triton tiles on the real
operand shapes at first forward**, then keeps whichever wins. That is a good design — it is
how the candidate avoids the guessed-tile failure that made v20 lose at 0.88x before
tuning. But it means the candidate does heavy GPU work during its own construction.

`run_matrix`'s isolated path times the baseline, then BUILDS the candidate — running the
tile sweep — then times the candidate, with 20 warmup iterations in between. The
interleaved path builds both first and alternates rounds, so planning is finished and
settled before any timing starts.

The three configs that disagree (4, 7, 12) are precisely those where v36's predicate
selects Triton at several sites. Configs 8, 9 and 13, where it selects nothing or where the
work dwarfs the planning, agree to 1.00x.

## The general shape

**A measurement protocol that builds an arm immediately before timing it will misreport any
candidate that does significant work at construction.** That is not a rare class: it covers
every autotuning candidate, every JIT-warming candidate, and anything that probes the
device to choose a strategy — which is most of what this project builds, because
CLAUDE.md's rule 2 *requires* predicates be derived from measured device properties.

Finding 42 established that the isolated protocol drifts. This is worse than drift: it is a
systematic 2-4x error correlated with the very design pattern the contract mandates.

## Disposition

Both arms stay recorded. The isolated arm remains the only one comparable with the ~600
rows that predate finding 42. **For ranking, use the interleaved arm**, and re-sweep any
candidate whose ranking matters so that both arms exist on it.

## L54 — If a candidate does work at construction, do not time it immediately after constructing it

The harness's job is to measure the steady state a grader will see, and a grader does not
rebuild the model between the two arms it compares. Our isolated protocol was built to
avoid finding 05's co-residency spill and did so correctly; the cost was never "a few
percent of drift" but a structural misreport of every candidate that plans, tunes or warms
at build time.

Corollary for authors: if your predicate times anything, say so in the docstring, because
it changes which measurement of your candidate is meaningful.
