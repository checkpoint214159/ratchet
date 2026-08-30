# 32 — Every candidate since v5 was wrong on the harness's own default

**Date:** 2026-08-30. **Found by:** the g22 executor, while building something else.
**Fixed by:** `v26_causal_correct` (gen 26, parent v23), now the frontier.

## The defect

Every candidate from v5 to v23 calls
`F.scaled_dot_product_attention(q, k, v, is_causal=True)` with the flag **hardcoded**.
Only `v1_fused_graph` ever read `self.config.causal`. On a non-causal config:

    v1_fused_graph          max_abs 7.05e-04   failed     0 / 32768   OK
    v8_padfast              max_abs 1.43e+00   failed 24846 / 32768   WRONG
    v13_safe_capture        max_abs 1.33e+00   failed 25249 / 32768   WRONG
    v18_capture_insurance   max_abs 1.58e+00   failed 24942 / 32768   WRONG
    v23_single_tile_attn    max_abs 1.67e+00   failed 25064 / 32768   WRONG

Three quarters of the output wrong, three orders of magnitude past the locked tolerance.

## Why 177 green tests never saw it

`bench/matrix.py` states that all 14 announced configs are causal, and that is correct. So
no ledger row is invalid, and nothing in the suite ever exercised the other branch.

But **the reference benchmark defaults to `causal: bool = False`** (line 89), with
`--causal` as an opt-in flag. Every measurement this project has ever taken used a setting
the harness does not default to. If the graders run the reference as-shipped on any shape
we did not anticipate, the submission returns garbage and passes none of its own tests
first.

This is L24 at its most literal — *correct because of how the harness was invoked*. And
the information was already present: v8's `_fastpath` consults `self.config.causal`,
because the redundant-key-mask proof (finding 11) depends on causality. The flag sat one
line above the call that ignored it.

## The fix, deliberately conservative

Non-causal delegates to the **unmodified baseline forward**. Not a fast non-causal path.

Several optimizations in this lineage depend on causality for CORRECTNESS, not speed:
v8's proof that a right-padded key mask is redundant is derived from causal masking, and
v23's kernel skips the causal triangle structurally. Re-deriving both against a case the
announced matrix never exercises, under a deadline, to speed up a shape we do not expect,
is not a trade this project should make. Exactly right on the unexpected shape; fast on
the fourteen expected ones.

## An accidental noise-floor measurement

v26's causal path is byte-identical to v23's, so the sweep re-measured the same code:

    cfg  6, 8, 13  (the large ones)    +0.0%,  -0.1%,  +0.6%
    cfg  2,3,4,9,11,12 (sub-ms)        -4.8% .. -6.9%
    geomean                            3.015x -> 3.103x   (+2.9%)
    total wall time                    69.2 ms -> 69.2 ms  (identical)

**The same code measured 2.9% apart on the geomean.** Every config above a millisecond
reproduced to within 0.6%; every deviation came from configs whose absolute time is a few
tens of microseconds. This is the clearest direct evidence yet for L29's +/-7% floor, and
a caution about the geomean specifically: it weights a 0.06 ms config equally with a 57 ms
one, so noise on the cheap rows dominates it.

**Reported honestly: v23 and v26 are the same speed. v26 is strictly more correct.**
The frontier is ~3.0x-3.1x; the total-wall-time figure (69.2 ms, unchanged) is the stable
one.

## L42 — Test the settings the harness DEFAULTS to, not the ones the problem statement uses

The problem statement says every config is causal. The harness says `causal=False` unless
told otherwise. We tested the former and shipped against the latter for eighteen
generations.

The audit rule that has now gone 7-for-7 in this project — padding ratio, eager baseline,
dtype, input scale, allocation context, process contention, and now the causal flag — is
always the same question: **what does this depend on that we never varied?** Every single
one was found by asking it, and not one was found by the search loop.

The generalization: for any flag, dtype or mode the harness exposes, the DEFAULT is a
distinct test case from the value the specification implies, and it is the more dangerous
of the two, because it is what runs when nobody passes an argument.
