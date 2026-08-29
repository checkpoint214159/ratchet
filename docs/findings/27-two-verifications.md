# 27 — Re-measuring v15 clean, and a fix whose value the benchmark cannot show

**Date:** 2026-08-30. Two verifications, opposite in character.

## 1. Finding 22 stands. The contention was real and immaterial.

Finding 26 established that the v15 sweep (15:57-16:00 UTC) overlapped research agent C,
which was benchmarking Triton kernels on the same GPU until 16:01, and marked finding 22's
quantitative conclusion PROVISIONAL. Re-measured on an idle GPU under the lock:

    v15 CONTENDED   2.618x        config 6:  70.72 ms
    v15 CLEAN       2.634x        config 6:  68.08 ms
    parent v9b      2.655x        config 6:  69.04 ms

The contention cost about **0.6% on the geomean and 3.7% on config 6** — real, and far too
small to move the verdict. Clean, v15 is still slightly below its own parent and the
difference is inside the noise floor. **Lifting Inductor's 68-SM veto still buys nothing
measurable, and finding 22's retraction was correct.**

Worth noting what changed sign and what did not. Config 6 went from "+2.4% worse than
v9b" to "1.4% better" — the per-config claim in finding 22 was contamination. The
conclusion was not. A result can be reported for the wrong reason and still be right, and
the only way to know which happened is to re-run it.

The contended rows are kept. The ledger is append-only, and a contended row is data about
a contended run — it is now the measured size of the contention hazard, which is the only
number of its kind we have.

## 2. v18's benefit is invisible to the benchmark that should validate it

    v17   2.759x     total wall 77.0 ms
    v18   2.765x     total wall 76.9 ms      (+0.2%, i.e. identical)

**This is the correct result and it is not a disappointment.** v18 changes nothing when
capture already succeeds, and in our harness it always does, because `run_accuracy_tests`
runs before the timing loop and allocates its input inside `inference_mode`.

The defect v18 fixes costs **2.25x** (0.267 ms vs 0.601 ms) and only appears when the
timing input is allocated outside `inference_mode` — which is what the graded harness does
at line 529. We are fast today purely because of test ordering upstream of us.

So the standard sweep is structurally incapable of showing v18's value: it exercises only
the path where the bug is masked. Promoting on the sweep alone would rank v18 as "no
change, why bother". The evidence for it is a dedicated experiment holding one variable,
plus a test that pins the parent's degradation so the insurance cannot rot into dead code.

v18 is merged as the frontier: not faster, strictly more robust, and identical in numerics
by construction.

## L39 — Some fixes are invisible to the measurement that ranks them

A candidate's score comes from a benchmark that exercises one calling pattern. A defect
that only fires under a different calling pattern is worth nothing on the scoreboard and a
great deal in the graded run. **A search that promotes strictly on measured score will
never find, and will actively discard, this entire class of fix.**

Both instances found this week arrived the same way: not from the loop, but from asking
"what does this depend on that we never varied?" — the audit rule that went 4-for-4 in
finding 27's predecessors (padding, baseline, dtype, input_scale) and is now 6-for-6 with
allocation context and process contention.

The practical consequence for spec 07: **A3 (time-to-signal) and the screen both assume
the sweep can see the effect.** For a robustness proposal that assumption is false, and
such a proposal needs a bespoke falsifier rather than a screen verdict. The rubric does not
currently distinguish these, and it should.
