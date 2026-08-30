# Finding 41 — The recombination, and the wrong answer it had to reset

**Date:** 2026-08-30 · **Generation 35** · **Branch:** `cand/g35/recombined`
**Parent:** `v33_streamed_long` · **Recombination contributor:** `v34_launch_bound`

Numbering note: this branch is cut from `cand/g33/config14`, where the config-14 protocol
is still `docs/findings/33-config-14-protocol.md`. On `ben` those have already been
renumbered to 39 (the launch floor) and 40 (config 14). 41 is free in both, so the merge
does not re-collide.

---

## What was merged

v26 has two live children and they are orthogonal:

| | changes | measured before this |
|---|---|---|
| `v33_streamed_long` | which shapes are computable, and how many times a model may be called | frontier returns `(8,128,128)` for a `(1,128,128)` input; v33 returns `(1,128,128)` |
| `v34_launch_bound` | how many kernels one call costs | 36 → 20 kernels, 53.2 → 44.6 us device time at config 2 |

Composed by **layering, not by copying**. v33's streaming dispatch was factored into
`build_on(base_cls)` so that v35 is `build_on(build_v34(baseline))` and exactly one copy
of each mechanism exists in the repository. `build(baseline_cls)` for v33 is now
`build_on(build_v26(baseline_cls))` and v33's own MRO is asserted unchanged by the
factoring. This is the argument v33 already made about importing v14's predicate rather
than restating it — two copies drift, one copy cannot ([L14]) — applied to the merge.

## The merge's own content: five attributes v33 could not know about

v33's `_invalidate_shape_state` enumerates the state latched to an input shape — the CUDA
graph, its three static buffers, v23's attention tile, v17's FFN gate. **v34 adds five
more and v33 cannot know their names**: `launch_reason`, `launch_fused_used`,
`launch_bm`, `launch_warps`, `mask_capture`, all computed in `_decide_launch` from
`(b*s, d_model, ffn_dim)` and the SM count.

`tests/bench/test_v35_recombined.py` computes that difference **from the classes** rather
than typing it out, so a generation 36 that latches a sixth attribute fails a test instead
of shipping a stale plan.

## A silent wrong answer, created by one fix and closed by the merge

`_nomask` is derived once, in `_prime`. v34's `_core` gates only on `launch_fused_used`;
the mask check lives in `_decide_launch`, which without a reset runs once ever. And v34's
`_try_capture` **elides the mask buffer entirely** when `_nomask` is True.

In v34 alone that is unreachable — the model raises on the second shape before it can
matter. **v33 removes the raise.** So the recombination is the first configuration in
this project's history where a model warmed on an all-True mask can be re-called, at a new
shape, with a mask that must be honoured, and the maskless kernel runs anyway.

Measured, with v35's reset deliberately neutered, warm at `(8,128,128)` unpadded then call
at `(16,128,128)` with `padding_ratio=0.4`:

```
max_abs 3.1612e+00      elements past the locked tolerance   69407 / 262144
```

26% of the output wrong by three orders of magnitude past `atol=2e-3` — the same shape and
nearly the same magnitude as the causal defect v26 was built to fix (finding 32 measured
1.67 there). With the reset in place the mask state is re-derived, `_decide_launch` re-runs
and declines, and the same call is correct.

`_prime` is idempotent — every implementation in the chain reassigns its lists rather than
appending — so re-deriving costs one pass of weight casts on a shape change, which happens
at most once per distinct shape and never inside a timed loop.

**Residual hole, stated rather than fixed:** the latch key is the input *shape*. A mask
whose validity changes at the *same* shape is still not noticed. Closing that needs
`mask.all()` on every forward — a device reduction and a host sync per call, on exactly
the launch-bound configs this candidate exists to make cheaper. Not reachable through the
harness (one model per config, one mask), so it is documented, not paid for.

## The two predicates are NOT disjoint. Config 7 satisfies both.

v34's docstring says `amortizes` and `one_wave` are "disjoint sets, which the tests assert
rather than assume". The **selected** sets are disjoint. The **predicates** are not:

```
cfg  7  tokens=8192  d=32  f=32  bm=64  smem=12288   amortizes=True  one_wave=True
```

Config 7's weights are 32x32, so they amortize at 8192 tokens *and* eight blocks fit per
SM, so the whole grid is one wave. Both statements are true of it. v34 resolves this by
**precedence** — `_decide_launch` asks `amortizes` first and returns — which is sound, and
v34's own test does check that the launch path does not claim config 7. But it is an
ordering rule inside one function, not a property of the two predicates, and the merge is
where that gets written down rather than inherited as the stronger claim.

The third predicate, `choose` (v14/v33), is on a different axis: capacity, not occupancy.
It selects no kernel. It decides how many times `_core` runs, and each of those runs then
asks the other two about *its own slice*. Thirteen announced shapes are resident, so their
slice is the whole input and the answer is unchanged. Config 14 streams, and its slice is
declined by `fits` — `d_model = ffn_dim = 1024` needs 4.25 MB of shared memory against 99
KB opt-in — at both the whole and the sliced shape, which is the check that matters:
streaming must not turn a shape the kernel cannot hold into one it thinks it can.

On the streamed path v34's `forward` is never reached, so `launch_reason` would have read
`"undecided"` at the moment `_core` ran — not wrong, since the flag defaults to False, but
silent, which is the failure mode finding 18 is about. v33's `_settle_slice_decisions`
hook is extended so the decision is made, and made against the slice.

## The kernel count carries a constant profiler offset; the drop does not

Counting device events over ten profiled forwards, repeated five times, GPU lock held:

```
                     run alone            inside the full 279-test suite
v26_causal_correct   36.0 x5              35.3
v33_streamed_long    36.0 x5              35.3
v35_recombined       20.0 x5              19.3
```

A **constant deficit of 7 events per profiled window**, identical for both models — the
profiler drops events in a loaded process. The *difference* is exactly 16.0 in both
environments. So the test asserts the drop, and that is a measured choice rather than a
loosened threshold: the absolute count is what the environment perturbs, the difference is
what the mechanism claims.

Note also that v33's count is identical to v26's, which is the negative control for the
merge: the streaming layer adds no kernel on the resident path.

## Score, diluted honestly ([L33])

No new mechanism, so no new speed argument. v34's bound carries over unchanged: 16 nodes
at 0.798 us is 12.8 us per forward, and against each config's own wall —

```
config  2   12.8 us / 0.061 ms  = 21.0%      config  4   11.5%
config 12   12.8 us / 0.103 ms  = 12.4%      config  8    0.45%
```

— at most **+0.065 of 3.000** on the capped weighted score, with configs 1, 8, 9, 10
getting exactly nothing because the predicate declines them. Config 14 stays at 1.0 and is
not a source of score (finding 33/40).

## The measurement problem, restated because it is not solved

v34 measured that removing 16 nodes pushed config 2 **from GPU-bound to CPU-bound**:

```
min-of-N     v34 0.0440-0.0471   v26 0.0604-0.0614    NO OVERLAP, 10/10
median       v34 0.0451-0.0666   v26 0.0604-0.0676    overlapping
```

`run_matrix` scores on `min(median, median)`. The win is real and the statistic that ranks
it cannot see it. Config 2 is also one of the four SCREEN configs, so the screen inherits
the same blindness. This candidate does not fix that and does not tune to it.

**What would resolve it:** report min-of-N alongside the median for the sub-millisecond
rows, and treat "the minima do not overlap across N repeats while the medians do" as a
distinguishable state rather than as noise — the reproducibility of configs 7, 8 and 10
across the same repeats is a free control that separates a discrete CPU/GPU regime change
from variance. That is a harness change, not a candidate change, and it belongs to whoever
owns `run_matrix`.

## Lineage note: the topology test forbids a true recombination merge

v17, the previous recombination, is a real two-parent git merge (`b0168b7 c7051cf b8b0a6b`).
That is no longer available. `test_new_candidates_branch_from_their_declared_parent`
requires, for every candidate above generation 18, that its git ancestors intersected with
the registry equal its *declared* ancestors — and the registry allows one `parent`. Merging
`cand/g34/launch-bound` would make `v34_launch_bound` (which has clean ledger rows at
`5fac19f`) an undeclared git ancestor of v35 and fail that test the moment v35 is measured.

So v34's four files were **ported, not merged**, and v34 is recorded as a contributor in
v35's registry summary — the convention v17 used at the registry level. This is a genuine
tension: the tree the method describes is one where recombination has two parents, and the
invariant that keeps CMP from degenerating into age cannot currently express that. Worth
resolving before the next merge; not resolved here.

(Three `test_lineage_topology` failures on this branch — v23, v26, v33 all git-descending
from `v19_norm_fused` — are **pre-existing**, verified identical on `bb7c471` with every
change of this generation stashed.)

## Screen verdict

`bench/screen_log.jsonl` is gitignored by design — advisory, partial sweeps, never feeds
clade sampling — so the two runs are recorded here instead.

```
                    geomean   cfg 2      cfg 7      cfg 8     cfg 10
screen 1            2.7377    0.04518    0.08499    6.7553    0.24269
screen 2            2.7216    0.04710    0.08397    6.7441    0.24166
v26 parent          2.5340

PROMOTE, +8.0%  and  PROMOTE, +7.4%      (commit 77b9aef, clean tree)
```

Screened against `v26_causal_correct` rather than the declared parent: v33 has only
config-14 rows on this branch, and on the thirteen resident configs v33 *is* v26 by
construction — the streaming layer delegates and adds no kernel, which the census confirms
independently (both count 36.0 nodes per forward, five out of five). v26 is also what v34
screened against, so the siblings are comparable.

**What this settles and what it does not.** v34 screened +8.1% and then +0.7% on the same
commit and a clean tree ([L46]). Both of my config-2 readings sit inside v34's measured
min-of-N band (0.0440–0.0471) and far below v26's (0.0604–0.0614); configs 7, 8 and 10
reproduced to three or four digits across both screens, which is the free control saying
the harness was steady while I sampled. So this pair did not exhibit the instability —
and two samples cannot establish that it is gone, since v34 needed one unlucky draw to
show +0.7%. A screen is one pass and advisory ([L41]). The mechanism is intact; the
statistic is still the open question, and the fix for it belongs to `run_matrix`.
