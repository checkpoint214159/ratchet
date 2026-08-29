# 20 — Backtesting the proposal rubric: three defects, found before it spent a GPU minute

**Date:** 2026-08-29. **Artifact:** `bench/proposals/backtest.py` (reproducible; no GPU).
**Subject:** spec 07, the proposal rubric.

## What was done

Scored 12 already-measured candidates on spec 07's ten dimensions, using ONLY each
candidate's docstring **as written at the commit that introduced it**, recovered with
`git show <intro_sha>:bench/candidates/<file>.py`. Then compared the rubric's predicted
ranking against what the ledger actually recorded.

**This is a falsification test, not a validation.** The scorer knows every outcome.
Hindsight cannot be removed from it. Passing means only "not obviously broken"; failing
would have meant "definitely broken". That asymmetry is the whole value of running it.

Three of the twelve (v1, v2, v3) had their own measured results written into the
docstring at the introducing commit — v1 opens with "3.11x geomean". Those are outcome
reports, not hypotheses, and are excluded from the primary test.

## Defect 1 — the spec's own formula is degenerate at Q = 1.0

`beta_0 = (1 - Q) * kappa` is exactly 0 when a proposal scores perfectly, and
`random.betavariate` raises `ValueError: alpha and beta must be > 0.0`. v12 scored 1.00
and crashed the sampler on the first run.

**Fix:** Laplace pseudo-counts on both sides, `alpha_0 = Q*kappa + 1`,
`beta_0 = (1-Q)*kappa + 1`. This also matches the convention `sample_parent` already uses
(`Beta(1+s, 1+f)`), so the node-side and idea-side samplers now agree.

Worth noting how this was found: not by review, but by running the thing on real data.
The formula reads fine.

## Defect 2 — THE IMPORTANT ONE: the rubric was scoring the wrong quantity

Initial Spearman rho between predicted quality and realized `geomean_vs_compiled` was
**+0.050**. Essentially zero. The rubric had no predictive power whatsoever.

The cause is structural, not a scoring error. **The cumulative geomean measures the whole
stack a candidate INHERITS, not what the candidate itself contributed.** v13 reads 2.711x
almost entirely because it inherited v12; v11 reads 2.514x because it inherited v9a. A
rubric that scores one IDEA cannot correlate with a metric that scores an ancestry.

Re-targeting to **marginal gain over the parent** — the candidate's own contribution —
raised rho from +0.050 to **+0.267** with no change to any score.

    v9a_compiled_core       parent v8_padfast              +58.3%
    v9b_reduce_overhead     parent v8_padfast              +56.9%
    v12_graph_over_compile  parent v11_lean                 +7.9%
    v6_fp16_gelu            parent v3_chunked               +7.6%
    v8_padfast              parent v6_fp16_gelu             +0.0%
    v13_safe_capture        parent v12_graph_over_compile   -0.0%
    v11_lean                parent v9a_compiled_core        -6.1%
    v5_fp16_resid           FAILED correctness (2/14)
    v7_fused_norm           FAILED correctness (12/14)

**This generalizes past the rubric.** Any selection mechanism scoring candidates by their
absolute ledger number is scoring their ancestors. CMP already gets this right by
construction — it scores a node by its descendants — but the scoreboard we have been
reading all week does not, and "v13 is the frontier at 2.711x" credits v13 with work v12
and v9a did.

## Defect 3 — modest framing talked the rubric down 20 points

The diagnostic row: **v9a scored Q=0.92 and v9b scored Q=0.72, and they realized +58.3%
and +56.9%.** Same parent, same mechanism, essentially the same outcome.

v9a's docstring frames itself as a division of labour between our algorithm and
Inductor's fusion. v9b's frames itself modestly — "a compile-COST question, not a speed
one" — and the rubric believed it, scoring A2 (headroom) at 2 instead of 5. But the two
candidates make the SAME move: hand the decomposition to a compiler. The gain came from
the mechanism, which both share, not from the framing, which differs.

A rubric that can be talked down by modest framing can be talked up by grandiose framing.
That is a direct exploit for a proposer agent scoring its own work.

**Fix — the sibling rule:** A2 scores the MECHANISM, not the proposal's presentation of
it. Proposals sharing a mechanism inherit the same A2. Applying it took rho from
**+0.267 to +0.483**, nearly doubling it, for a one-line change.

## Calibration — entropy was being swamped

At the spec's original `KAPPA=(20, 2)` the top-ranked idea won **50.5%** of Thompson
draws. With five expander agents drawing concurrently, two would receive the same idea.
The entropy axis was present but not doing work: the Q spread (0.68-1.00) dominated the
kappa spread entirely.

    KAPPA=(20, 2)      top idea takes 50.5%
    KAPPA=(12, 1)      top idea takes 36.1%
    KAPPA=( 8, 0.5)    top idea takes 27.5%   <- adopted
    KAPPA=( 4, 0.25)   top idea takes 19.1%

Adopted **(8.0, 0.5)**. At that setting a five-draw queue pulls roughly four distinct
ideas, which is the behaviour the architecture needs. (4, 0.25) explores more but starts
drawing v11 — a candidate the rubric correctly ranked last — often enough to waste GPU.

## Verdict, stated honestly

The rubric passes its four pairwise acceptance checks (v9a and v12 both rank above v5 and
v7) and reaches rho = +0.483 against marginal gain.

**That is NOT significant.** For n=9 the two-tailed 5% critical value of Spearman's rho is
about 0.68; +0.483 does not reach it. The honest claim is: *the rubric is not obviously
broken, and it is now free of three specific defects it demonstrably had an hour ago.*
It has not been shown to work.

Two things it still gets wrong, left standing deliberately:

  * **v5_fp16_resid, Q=0.88, ranked 4th, failed correctness on 12 of 14 configs.** The
    rubric rates it highly and I am NOT fixing that. v5 produced finding 08 (the fp32
    residual is load-bearing), which redirected three subsequent generations. Ranking it
    highly is correct for a rubric that prices information; it is only wrong against a
    metric that counts speedups. This is the tension the two axes exist to hold, and the
    marginal-gain target scores it -100% precisely because that target is speed-only.
  * **v7_fused_norm, Q=0.84, failed.** Its A2 was reduced from 5 to 3 on the text's own
    admission that most configs would see "likely no win at all". A rubric reading the
    proposal's own hedges is working as intended, but 0.84 is still too high for
    something that bought nothing.

## Learning

**L30 — A scoring function must be backtested against the thing it will select on, and
the obvious target is usually the wrong one.** The rubric had zero predictive power for
half an hour purely because it was aimed at cumulative rather than marginal outcomes. The
same class of error as finding 12 (the objective had saturated) and finding 06 (the search
found noise): in all three the machinery was fine and the QUANTITY was wrong. Check what a
number is actually measuring before trusting a ranking built from it.
