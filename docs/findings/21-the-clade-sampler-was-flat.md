# 21 — The node sampler was flat, and the obvious fix made it worse

**Date:** 2026-08-29. **Changed:** `bench/ledger.py::clade_stats`. **Tests:** 92 pass (2 new).
**Found by:** the user asking whether the idea sampler and the node sampler are the same thing.
They are not — and checking in order to answer surfaced that the node sampler had the
same defect that finding 20 had just fixed on the idea side.

## The defect

`clade_stats` decided a row was a clade success with `speedup > 1.0`, against the **eager**
baseline. Measured over the archive:

    clean candidate rows                                344
    counted as SUCCESS                                  303  = 88.1%
    median speedup vs eager                            6.93x

A Beta posterior fed 88% wins is nearly flat. **The sampler that decides where every
expansion attaches was barely discriminating between lineages at all** — it had been that
way all week, silently.

Finding 12 established in the morning that the eager baseline is the wrong reference and
that speedups must be quoted against `torch.compile`. The objective function was fixed.
Nobody propagated it to `clade_stats`. A lesson recorded in one place is not applied in
another unless someone walks the call sites.

## The obvious fix makes it WORSE, which is why it was measured

Swapping in the compiled baseline gives a healthy-looking 70.6% success rate. But the
resulting clade ranking correlates with **commit age** at rho = +0.660 — considerably
worse than the +0.269 it replaced.

The reason is the L1 degeneracy re-entering through the success criterion rather than the
topology. Every late commit clears the compiled bar, so an old node's subtree accumulates
credit for wins its descendants merely INHERITED. That is precisely the confusion finding
20 fixed on the idea side: **crediting a candidate with work its ancestors did.** Same
error, different sampler, found the same day.

    criterion                       success rate    rho(clade rank, commit age)
    speedup > 1.0 vs eager              88.1%   <- vacuous          +0.269
    beats compiled baseline             70.6%                       +0.660  <- WORSE
    improves on nearest ancestor        19.2%                       -0.192
    BOTH (adopted)                      15.4%                       -0.130

## What was adopted

A success must clear **both** bars:

  1. it beats the **compiled** baseline for its config (the real competitive reference), and
  2. it improves on the nearest **ancestor commit** that measured the same config, by more
     than `CLADE_NOISE = 0.07` (the measured run-to-run spread on short configs, L29).

The ancestor is found by walking git parents breadth-first, so a merge compares against
the closer of its two lineages. A descendant that merely carries its parent's win forward
now scores nothing — which is the entire point of metaproductivity, since the parent
already holds that credit.

## A second, pre-existing bug found while fixing the first

`clade_stats` skipped rows named `baseline` but **not** rows named `baseline_compiled`.
The eager row was excluded with a comment explaining that a guaranteed 1.0x speedup books
a guaranteed failure per config; the compiled row got no such treatment.

So under the old rule the compiled baseline — the yardstick itself — scored ~2.63x against
eager and was counted as **13 candidate successes**, inflating the very statistic it exists
to be measured against. Under the new rule it would have flipped to 13 guaranteed failures,
since nothing can beat itself. Neither is right: it is not a candidate. Now excluded, with
the reasoning written down so it does not regress.

## Result

    pooled successes            471 / 3194 = 14.7%     (was 88.1%)
    rho(clade rank, commit age)      -0.158            (was +0.269)

The Thompson draw now concentrates on `c0808609` (v8_padfast), `27d27114` (v9a) and
`2e855f81` (v9b) — the fork that produced the single largest real jump in the project
(+58% marginal). It ranks `ded3d7a3` (v13) last at a posterior mean of 0.029, which is
correct: v13's own marginal contribution is -0.0%. It is a bug fix that PROTECTED v12's
win, not a candidate that produced one.

## Learning

**L31 — The same miscalibration will exist in every consumer of a number until each one is
walked.** Finding 12 fixed the objective; the identical error survived in `clade_stats`
for a week because nobody enumerated the call sites. Both samplers in this system were
miscalibrated simultaneously, for the same reason, and neither would have been caught by
a test — they produced plausible numbers, just not meaningful ones.

**L32 — Measure the fix, not just the bug.** The intuitive repair here (use the compiled
baseline) is a 2.5x REGRESSION on the property that actually matters, and it looks like an
improvement on the property that is easy to check. Four candidate criteria took ten
minutes to evaluate and cost no GPU time at all.
