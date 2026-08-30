# 44 — The second recombination: nine more latched attributes, and the reset stops being a list

**Date:** 2026-08-30 · **Generation 37** · **Branch:** `cand/g37/recombined2`
**Parent:** `v36_gemm_gelu` · **Recombination contributor:** `v35_recombined`

**Numbering note.** This branch is cut from `cand/g36/gemm-gelu`, whose `docs/findings/`
carries 43 (the projection GEMMs) and, from the branch it was cut from, a *differently
numbered* copy of the launch floor (33 there, 39 on `ben`). This file uses **`ben`'s
numbering** throughout — 39 the launch floor, 41 the first recombination, 42 the
harness/grader disagreement, 43 the projection GEMMs — and is numbered 44 to sit after
`ben`'s last. `00-learnings.md` and the findings `README.md` are deliberately **not**
edited here: `ben` is ahead of this branch on both and appending would collide.
**The proposed lessons are stated at the bottom for the merge to lift.**

---

## What was merged, and why neither could contain the other

`v26_causal_correct` has had two live lines since generation 33, and
`git merge-base cand/g35/recombined cand/g36/gemm-gelu` is `cand/g32/persistent-layernorm`
— so neither branch is an ancestor of the other and neither could have inherited the
other's work.

| | changes | measured before this |
|---|---|---|
| `v35_recombined` | *which* shapes are computable, how many times a model may be called, and how many kernels one call costs | frontier returns `(8,128,128)` for a `(1,128,128)` input; v35 returns `(1,128,128)`. 36 → 20 kernels |
| `v36_gemm_gelu` | what one of those kernels costs | **+0.082 of weighted_score (2.489 → 2.571), no regression on any config** |

Their win sets are close to disjoint. Under the graded harness, **by candidate time** —
never by the reported speedup, whose baseline arm spreads 33–39% on the sub-millisecond
rows (finding 42's addendum):

```
  cfg      v26      v34      v35        v35 wins config 3 outright, where v34 is
    2   0.1440   0.0481   0.0481        WORSE than the v26 it descends from
    3   0.0743   0.0886   0.0553
    4   0.1106   0.0922   0.0922
   12   0.1792   0.0922   0.0922
```

while v36's +0.082 lands on configs 1, 4, 5, 9, 10 and 12.

Composed by **layering, not by copying**: `v37 = build_on(build_v36(baseline))`, one rung
higher than v35 stacks the identical `build_on`. MRO `V37 → V33 → V36 → V34 → V26`, one
copy of every mechanism, asserted by object identity
(`M.build_streaming_on is v33.build_on`, `M.build_v36 is v36.build`). `v33_streamed_long.py`,
`v35_recombined.py` and their two test files are ported **byte for byte** from
`cand/g35/recombined`, so v35's measured rows still describe the code that produced them.

---

## The merge's own content: v35's reset stops being a list

v33's `_invalidate_shape_state` enumerates the state latched to an input shape — the CUDA
graph, its three static buffers, v23's attention tile, v17's FFN gate. **v35's entire
content** is that v34 adds five more that v33 cannot name: `launch_reason`,
`launch_fused_used`, `launch_bm`, `launch_warps`, `mask_capture`. v35 typed them into
`SHAPE_LATCHED_BY_V34` and its test computed the same set from the classes, so that — in
its own words — *"a generation 36 that latches a sixth attribute fails a test instead of
shipping a wrong answer."*

**Generation 36 latched nine.** Five flags (`gemm_used`, `gemm_reason`, `gemm_sites`,
`gemm_engaged`, `gemm_stats`) and four private tile tuples (`_tile_qkv`, `_tile_out`,
`_tile_ffn_in`, `_tile_ffn_out`), every one computed in `_decide_gemm` from `m = b * s`.
The prediction was right and it arrived one generation late and nine attributes wide.

So v37 does not name them either. `shape_latched_over(top, base)` walks the MRO and
returns **every class-body attribute introduced above v26, with its class-body default**;
the reset restores all of them. Two properties make that a claim rather than a
convenience:

* Run over `(v34, v26)` it returns **exactly** v35's hand-written five — the cross-check
  that the generic rule is the same rule v35 applied by hand, not a different claim
  wearing its clothes.
* Underscore-prefixed names are **included**. v35's helper excluded them, which was sound
  for v34 (it latches nothing private) and would have silently dropped four of v36's nine.
  That one line is the difference between a reset and a reset that looks complete.

`_proj_t` is *deleted* rather than reset: it is an instance attribute built by
`_decide_gemm` out of `self._cache`, so it has no class-body default, and `_prime` rebuilds
`_cache` underneath it.

A test asserts the derived set is a strict superset of v35's declared five and names the
nine, and a second test builds a synthetic `Generation38` subclass with a fifteenth
attribute and requires the same rule to see it — [L38]: verify a check is capable of
failing before trusting that it passed.

---

## What the v36 half of the merge does NOT create, stated precisely

v35 recorded a genuine wrong answer that exists only in the combination it made: v34's
`_try_capture` elides the mask buffer when `_nomask`, unreachable in v34 alone because v34
raises on the second shape, and **v33 removes the raise** — 69407 / 262144 elements past
the locked tolerance with the reset neutered. That fix is inherited here unchanged and
re-verified.

**The v36 half does not add a second one, and saying so precisely matters more than
claiming one.** A stale `_tile_*` is a tile sized for a batch that is no longer there:
`proj_gemm` masks its M edge and `legal()`'s only M-dependent rule is a padding-waste
rule, so the arithmetic stays right and the kernel merely runs the wrong shape's plan. A
stale `_proj_t` holds transposes of the same weights. What the reset buys on this half is
therefore **a correct plan rather than a correct answer** — and, on the streamed path, a
plan at all.

That distinction is the reason the control in
`test_a_reshape_re_decides_the_gemm_plan_and_the_control_shows_it_would_not` builds the
*naive* composition (`build_on(build_v36(...))`, no reset extension) and requires it to be
seen keeping the old shape's plan. Without the control the test would only show that v37
re-decides, never that there was anything to re-decide.

---

## Three decisions on the streamed path, and the order is load-bearing

v33's `forward` returns before it ever reaches v36's, so `_settle_slice_decisions` is where
every shape-latched decision has to be made. v33 settles the attention tile and the FFN
gate; v35 added the launch decision; this adds the GEMM plan. **`_decide_gemm` reads
`launch_fused_used` and `fused_ffn_used`** — where a megakernel already owns the FFN there
is no `F.linear` left to replace — so planning it first produces tiles that are never
launched and a `gemm_reason` that overstates what engaged, which is the failure v36's own
test file exists to pin. A test asserts the ordering on the source, after stripping the
docstring (which names them the other way round on purpose, because it is explaining the
dependency rather than obeying it).

---

## The five verifications

`tests/bench/test_v37_recombined2.py` — **43 tests, all passing**, GPU lock held
throughout. `test_v33_streaming.py`, `test_v35_recombined.py` and `test_v36_gemm_gelu.py`
are ported byte for byte and run here unchanged as the regression surface for the halves:
**74 tests, all passing**, which is the check that the ports are the siblings' code and
not an adaptation of it. `tests/bench` as a whole is green apart from the two pre-existing
`test_lineage_topology` failures (v23 and v26 git-descending from `v19_norm_fused`),
verified identical on `2b8d600` with this generation stashed.

**1 — v33's shape fix survives.** Warm at `(8,128,128)`, call at `(1,128,128)`; assert the
output is `(1,128,128)` and inside `atol` of the reference. The **v26 control still returns
`(8,128,128)`**, and the test asserts that too, so it is capable of failing.

**2 — v35's reset survives and now covers fourteen attributes.** `shape_latched_over` run
over `(v34, v26)` returns v35's hand-written five exactly; run over `(v36, v26)` it returns
those five plus v36's nine. A synthetic `Generation38` subclass with a fifteenth is caught
by the same rule with no edit to the candidate. On the device: a reshape re-derives both
`launch_bm` (16-row tile at 1024 tokens → 32-row at 2048) and `gemm_reason`, and **the
naive composition — `build_on(build_v36(...))` with no reset extension — is built inside
the test and asserted to keep the old shape's plan**, so the fix is measured against a
control rather than asserted alone. v35's mask hazard is re-verified: warm all-True, call
at a new shape with `padding_ratio=0.4`, and the fused maskless kernel must decline.

**3 — v34's kernel-count cut survives.** Counted from device events, 10 forwards, the
DROP against `v33_streamed_long` (which counts identically to v26 — the streaming layer
adds no kernel on the resident path, the negative control for the merge):

```
  cfg  2    v33 36.0  ->  v37 20.0     drop 16.0
  cfg  3    v33 35.0  ->  v37 19.0     drop 16.0
  cfg 12    v33 35.8  ->  v37 19.8     drop 16.0
```

**4 — v36's projection GEMMs fire, asserted by name.** v36's first draft ran four cuBLAS
calls on config 9 while `gemm_reason` claimed Triton and every accuracy test passed, so a
reason string is a claim and the device profile is the evidence. Two witnesses are
required: `_proj_gemm` must appear in the device events, and the four free-standing GELU
launches must be gone — with the parent asserted to still have them.

```
  cfg  9    v34 34.6  ->  v37 30.6     drop 4.0    sites out, ffn_in, ffn_out
  cfg 10    v34 34.4  ->  v37 30.4     drop 4.0    sites out, ffn_in, ffn_out
  cfg  1    v34 34.2  ->  v37 30.2     drop 4.0    sites out, ffn_in, ffn_out
```

**5 — config 3. And this is where the measurement had to change.**

---

## Config 3: the wall does not reproduce, and the device time does

Config 3 is 512 tokens — the second most launch-bound row on the matrix. Four independent
runs, all four candidates resident, GPU lock held, 400 warmup calls then 200 timed:

```
                 DEVICE us / forward (4 runs)              WALL us, min      nodes
  v34         47.61   47.33   47.60   47.75   = 47.57      54.3 - 55.3        20.0
  v35         47.90   47.57   47.13   47.33   = 47.48      55.3 - 55.5        20.0
  v36         44.57   43.40  141.53*  44.82   = (outlier)  52.4 - 161.9*      19.9
  v37         43.02   43.16   43.04   43.26   = 43.12      52.3 - 52.5        19.9
```

**v37's config-3 device time reproduces to 0.3% across four runs and is 9.2% below both
v34's and v35's.** `_proj_gemm` at 8.69 µs replaces `ampere_fp16_s16816gemm` at 11.54 µs,
and the node count is unchanged (one GEMM for one GEMM). The minimum wall agrees: 52.3–52.5
µs against v34's 54.3–55.3 and v35's 55.3–55.5.

**The median wall agrees with nothing.** Across the same four runs plus one 8-config
ABBA sweep, config 3's median read 53.3, 55.9, 56.6, 65.6, 80.1, 85.1, 92.2, 105.5 and
162.5 µs for arms whose device time never moved more than 4%; even the *minimum* read 86.0
in the interleaved 8-config run against 52.3 in every isolated one. **On this row the wall
is a measurement of the host's scheduling, not of the candidate**: the device does 43–48 µs
of work, the Python side needs 25–50 µs to submit it, and any jitter lands directly in the
number. That is finding 42's addendum one row further down the size axis, and it is why
this file ranks config 3 by device time and node count and says so.

Verification 5 therefore reads: **v37 is at least as good as v35 on config 3 on every
quantity that reproduces** — device time 43.1 vs 47.5 µs, minimum wall 52.4 vs 55.4 µs,
node count 19.9 vs 20.0 — and neither is rankable on the median.

### The outlier is the contributor's, and it is worth naming

v36's second config-3 run measured **141.53 µs of device time and a 162 µs wall**, 3.3x its
own other three runs, with the node count unchanged. The tile sweep is a *timing*, so it
lands differently run to run: v36 chose `(64,64,32,4,4)` for `qkv` in one run and v37 chose
`(128,32,32,4,4)` in another, both reported as decisive wins over the vendor. **A candidate
whose kernel selection varies run to run injects that variance into every measurement taken
of it** — [L29], stated in `proj_gemm.py`'s own `DECISIVE` comment and evidently not
sufficient at 512 rows, where the vendor and the sweep are 2.4 µs apart. v37 did not show
it in four runs; **four runs cannot establish that it is gone**, and the mechanism is
inherited unchanged, so this is a hazard of the line, not a property v37 fixed. It belongs
to whoever next touches `proj_gemm.plan`.

---

## End to end: v37 against the three candidates it descends from

ABBA-interleaved, all four arms resident (the graded harness's own condition), five rounds
of 100 timed calls after 300 warmup, **cold round discarded, minimum of the four remaining
round-medians**, one process per config, GPU lock held. Timed with the reference's own
`benchmark_once` loop — 2N events recorded around N unsynchronized calls, one sync at the
end — because syncing per call inserts a host round trip into exactly the rows being ranked.
`bench/abba.py`. Config 8 runs byte-identical code on all four arms and is the in-run
control.

| cfg | v34 | v35 | v36 | **v37** | v37 sites |
|---|---|---|---|---|---|
| 2 | 47.10 | 47.10 | 47.10 | **47.10** | — |
| 3 | 62.46 | 86.02 | 91.14 | **92.16** | qkv — *see above; not rankable* |
| 4 | 92.16 | 92.16 | 81.92 | **81.92** | qkv, out |
| 12 | 83.97 | 84.59 | 75.78 | **74.75** | qkv, out |
| 1 | 236.54 | 236.54 | 224.26 | **224.26** | out, ffn_in, ffn_out |
| 9 | 236.54 | 235.52 | 224.26 | **225.28** | out, ffn_in, ffn_out |
| 10 | 243.71 | 243.71 | 231.42 | **232.45** | out, ffn_in, ffn_out |
| 8 | 6626.8 | 6634.5 | 6633.5 | **6622.7** | — *(control: ±0.2%)* |

Read as a null, which is what it should be: **v37 and v36 agree to within 0.5% on every
config except 3**, and they should, because in the steady state at a single input shape
they execute the same code — the streaming layer's dispatch runs once, returns `resident`,
and delegates. The in-run control on config 8 puts the floor at ±0.2%, and 1/9/10 (identical
GEMM shapes) agree with each other to 0.5pp, which is the check that the protocol is not
measuring itself.

**So this merge buys no speed and is not supposed to.** It buys v36's +0.082 of
weighted_score *and* v35's shape fix, streaming, and reset — in one candidate, where before
you had to choose.

---

## Verdict: should this be the frontier?

**Yes, on correctness grounds; not on speed grounds, because it has none of its own.**

The case *for*: v37 is a strict superset of v36 in capability and a strict superset of v35
in speed, and the ABBA table shows it costs nothing to be both. It is the only candidate in
the tree that simultaneously (a) returns the right shape when a model is called twice,
(b) can compute a shape too large to hold resident, (c) re-derives mask state when the
shape changes, and (d) carries v36's +0.082. If the frontier is "the thing we would
submit", every one of those is a reason to submit this rather than either parent.

The case *against*, stated honestly:

* **It has not been swept.** Everything above is `bench/abba.py`, which is an instrument
  for *ranking* and does not write to the ledger. Eight of fourteen configs are measured
  here; 5, 6, 7, 11, 13 and 14 are not. The claim "no regression anywhere" is v36's,
  measured on v36, and inherited by argument (same code) rather than by measurement.
  **A proper `run_matrix` sweep at this commit is the gate, and I did not run one** — the
  controller owns the GPU for those ([L41]: a probe may propose, it may never conclude).
* **The two topology failures are still open.** `test_lineage_topology` fails for v23 and
  v26 on this branch, verified identical on `2b8d600` with this generation stashed, so
  they are pre-existing and not mine — but finding 41 already flagged that the topology
  invariant *cannot express a two-parent recombination at all*, and this is now the second
  recombination recorded as a one-parent lineage plus a summary string. Two of those and
  the declared tree is no longer the tree.
* **v36's tile-sweep instability is unresolved and inherited** (the 3.3x config-3 outlier
  above). It is the largest single risk in this candidate and it is not this candidate's
  bug.

What would change the verdict either way: a `run_matrix` sweep of all fourteen at this
commit, and a repeat of the config-3 device census on v36 to see how often the outlier
lands.

---

## Proposed lessons, for `00-learnings.md` to lift

**L54 — A reset that is a list of names is a list that will be out of date.**
v35 declared five shape-latched attributes and wrote a test predicting that generation 36
would add a sixth. Generation 36 added nine, four of them private — and v35's own helper
skipped private names, so the test that was supposed to catch it would have caught five of
the nine. **Derive the set from the classes at run time and let the declared list be the
audit surface, not the mechanism.** A test that fails is better than a wrong answer; a
reset that cannot be incomplete is better than both.

**L55 — Say which kind of hazard a merge closes.**
v35's merge closed a silent *wrong answer* (69407 elements past tolerance, measured). v37's
merge closes a stale *plan* — the arithmetic was never at risk, because the kernel masks
its own M edge. Both need the same fix and only one deserves the same language. Grading
every hazard as a wrong answer is how a project stops being able to tell which of its
guards are load-bearing.

**L56 — A recombination whose two halves are disjoint should be measured as a control
first.** In the steady state at a single input shape, v37 and v36 execute the same code:
the streaming layer's dispatch runs once and returns `resident`, and the reset never fires.
So the honest first measurement of v37 against v36 is a **null**, and any non-null is
evidence about the harness rather than about the candidate. That makes this merge its own
in-run control — a property worth arranging on purpose rather than discovering. It is also
what caught the config-3 story: two arms that must agree, reading 92 and 91 µs in one
protocol and 52 and 44 µs in another, is a statement about the protocol.

**L57 — On a launch-bound row, census the device before you rank the wall.**
Config 3's wall spread 53–162 µs across nine measurements of candidates whose per-forward
device time never moved more than 4%. The device does 43–48 µs of work and the Python side
needs 25–50 µs to submit it, so the wall is measuring whichever of the two lost the race.
`torch.profiler`'s device-event total is cheap, needs no interleaving, and reproduced to
**0.3% over four runs** where no wall statistic reproduced at all. **Where host submit time
is within 2x of device time, report the device census and the node count as the primary
result and the wall as context** — the wall is still what the grader scores, but it is not
what tells you whether your kernel got faster.
