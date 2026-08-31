# 54 — A floor, not a vote: the sharper timer's variance is one-sided, and it lands on the incumbent

**Date:** 2026-08-31. **Generation:** 43. **Branch:** `cand/g43/stable-tiles`.
**Parent:** `v42_hot_tuned_tile` (`5643e2c`). **Candidate:** `v43_replicated_tile`.
**Takes:** the fix finding 53 pre-registered and deliberately declined to bundle —
*"require the margin to replicate before it displaces"* — and a second, unrelated question:
*do v16's un-swept FFN megakernel constants survive the hot timer?*

**Numbering note:** taken as 54 on a candidate branch, following the 53 this branch
carries. `ben` may have renumbered; cross-references below name files rather than numbers
where it matters. `00-learnings.md` and the findings README are deliberately NOT edited
here — `ben` is ahead. Proposed lessons are at the bottom for the merge to pick up.

---

## JOB 1 — THE PRE-REGISTERED FIX WAS BUILT AS WRITTEN, MEASURED, AND IS WRONG

Finding 53's closing proposal, in its own words:

> **Require the margin to replicate before it displaces.** `autotune_tile` should run its
> sweep twice and displace the derived tile only if both sweeps clear `DECISIVE` *and*
> agree on the winner. […] Predicted effect: config 2 keeps `(16,4,1)` (7/7 agreement, 28%
> margin), config 3 reverts to `(64,4,1)` and stops moving.

That was built first, exactly as specified, and measured before anything else was tried.
**It gets config 3 right and loses config 2 — v42's entire win — in 5 of 10 fresh
processes.** Not stochastically: it loses it in precisely the 5 runs where the candidate
primed *second*, after the control arm had been built, primed and left resident.

```
prime order                      v43 (vote rule)         v42 (control)
v42 first, v43 second     x5     single_tile(64,4,1)     single_tile(16,4,1)
v43 first, v42 second     x5     single_tile(16,4,1)     single_tile(16,4,1)
```

and the tuner said, in its own reason string, exactly why:

```
2 sweeps did not agree on a winner ((16, 4, 1), (64, 4, 1)), so no margin replicated
```

on a shape whose true margin is **28%**. A 28% margin should not fail to replicate. That it
did is the finding.

### THE MECHANISM: THE NOISE IS ONE-SIDED AND IT LANDS ON THE INCUMBENT

`bench/probes/g43_stable_tiles/sweep_grids.py` dumps every arm of every sweep instead of
just the winner — [L62]'s cheap check, asked of *replicated* sweeps. Six back-to-back
sweeps in the model's own regime (a second model built, primed and resident; inside
`inference_mode`), at B=4:

```
tile          sw1     sw2     sw3     sw4     sw5     sw6     floor
(16,2,1)    2.827   2.840   2.636   4.064   2.831   2.638    2.636
(16,4,1)    2.743   4.002   2.547   4.029   2.741   2.574    2.547
(32,2,1)    2.887   4.446   2.670   4.866   2.873   2.676    2.670
(32,4,1)    2.766   4.455   2.593   2.743   2.568   2.594    2.568
(64,4,1)    2.711   2.517   3.752   2.708   2.529   4.245    2.517   <- derived
winner       (64)    (64)    (16)    (64)    (64)    (16)
margin      1.000   1.000   1.473   1.000   1.000   1.649
```

**The two sweeps that displace are not a challenger reading fast. They are the incumbent
reading slow** — 3.752 and 4.245 against its own 2.517 floor — while the challenger barely
moves. The same shape at B=1:

```
tile          sw1     sw2     sw3     sw4     sw5     sw6     floor
(16,4,1)    5.227   1.899   3.169   1.904   3.962   2.046    1.899
(64,4,1)    2.431   2.426   2.445   4.496   2.626   4.413    2.426   <- derived
winner       (64)    (16)    (32)    (16)    (32)    (16)
```

Three different winners in six sweeps **on the shape finding 53 recorded as stable in 7 of
7** — because 7 of 7 was seven *first* sweeps in seven processes, and this is six
consecutive sweeps in one.

Finding 53 had already stated the premise and not drawn this conclusion from it:

> Contamination on this harness is **one-sided**: a descheduled host thread, an unsettled
> graph or a co-resident allocation can only make a reading slower, never faster.

If the noise can only push a reading up, then it lands on whichever arm it lands on, and
`min()` over a contaminated grid returns *the arm the noise happened to miss*. **A vote
between two contaminated rankings is decided by where the contamination fell, not by which
tile is faster.** Two independent draws of that lottery agreeing is a coincidence, so
requiring agreement makes the tuner *more* likely to revert — including on margins that
are entirely real.

### THE FIX THAT THE MECHANISM DEMANDS: REDUCE PER ARM, THEN DECIDE ONCE

Sweep the grid `replicates` times, take **each arm's minimum across the sweeps**, and run
v23's decision rule once on the reduced table with nothing else changed. That is the
estimator CLAUDE.md already prescribes for a card whose clocks will not lock, applied one
level lower than it had been:

> the mean is a statistic about how often the machine misbehaved and the minimum is a
> statistic about the code

Reduced by the floor over **adjacent pairs** of the grids above — i.e. what `replicates=2`
would actually have seen — every window agrees:

```
             window (1,2)   window (3,4)   window (5,6)   verdict
  B=4          hold          hold           hold          derived tile, 0.988x, inside DECISIVE
  B=1        displace       displace       displace       (16,4,1) at 1.277x / 1.284x / 1.284x
```

and 1.277x is the margin finding 51 (twice), the g42 probe (twice) and this one all agree
on. Every arm is timed in every sweep and reduced over the same number of readings; an arm
that failed to time in any sweep is dropped rather than reduced over fewer, because an arm
with a smaller trial budget than its rivals is finding 47's best-of-N handicap inverted.

**So the pre-registration named the right defect, priced the right cost, and prescribed the
wrong remedy — and the remedy fails in the direction that would have looked like success**
(config 3 stabilised) while quietly destroying the parent's only scoring row.

### THE MEASUREMENT

`bench/probes/g43_stable_tiles/prime_stability.py --mode fresh`: **one process per
replicate, BOTH arms built and primed inside it**, prime order alternated between
replicates. That is `bench/abba.py`'s regime and therefore the regime in which every
ranking of these two candidates is taken. The observable is the selection rule's OUTPUT —
`(attn_form, attn_tile)` — not a time, because a plan that varies run to run shows up in a
timing as variance, which is the thing every protocol here exists to suppress.

```
cfg   arm                      asks   distinct plans
  2   v42_hot_tuned_tile        22    1   single_tile(16,4,1) x22
  2   v43_replicated_tile       22    1   single_tile(16,4,1) x22          <- the win survives
  3   v42_hot_tuned_tile        22    3   (64,4,1) x17, (16,4,1) x4, (32,4,1) x1
  3   v43_replicated_tile       22    1   single_tile(64,4,1) x22          <- fixed
```

Blast radius, same protocol, 4 replicates each:

```
cfg  1   both 1 plan, identical    single_tile(64,4,1)              inert here
cfg  7   both 1 plan, identical    single_tile(64,4,1)              inert
cfg 10   both 1 plan, identical    looped(64,32,4,4)                inert
cfg 11   both 1 plan, identical    single_tile(64,4,1)              inert
cfg 12   both 1 plan, identical    single_tile(32,2,1)              inert
cfg  4   BOTH ARMS UNSTABLE        looped(64,32,4,3) vs single_tile(64,4,1)
```

**Configs 1 and 4 are not fixed and are not this candidate's to fix.** Their instability is
a `looped`-versus-`single_tile` flip decided by `attn_choice.autotune_looped`, which this
change does not touch. Config 4 shows it here on both arms; config 1 shows it in the A/B
below, where v42's arm selected `looped(32,16,2,2)` in one run and `single_tile(64,4,1)` in
the other — so **4 replicates was not enough to catch it on config 1 and this table
understates the residue.** Finding 53 flagged configs 4 and 12 as coin flips through that
routine; 12 has since come out stable in 4 of 4 on both arms, 4 has not, and 1 joins them.

### THE END-TO-END A/B: ZERO, AND THE CONTROL FLOOR SAYS ZERO IS ALL THAT IS RESOLVABLE

`bench/abba.py --ids 1 2 3 4 7 9 10 11 12 --arms v42_hot_tuned_tile v43_replicated_tile
--rounds 5 --warmup 200`, GPU lock held, both arms resident, ABBA-interleaved, cold round
discarded, correctness checked against a fresh reference at the locked tolerance before any
round. Two full runs; floor of the two, per arm
(`bench/probes/g43_stable_tiles/summarize.py`):

```
cfg      v42 us     v43 us   floor ratio   plans
  1      220.16     224.26      0.9817x    DIFFER -- autotune_looped flipped, on v42's arm
  2       49.66      50.18      0.9898x    CONTROL, same plan
  3       53.25      54.27      0.9811x    CONTROL, same plan
  4       99.33     100.35      0.9898x    DIFFER -- autotune_looped flipped, on v43's arm
  7       77.82      77.82      1.0000x    CONTROL, same plan
  9      225.28     224.26      1.0046x    CONTROL, same plan
 10      223.23     223.23      1.0000x    CONTROL, same plan
 11      269.31     268.29      1.0038x    CONTROL, same plan
 12       74.75      74.75      1.0000x    CONTROL, same plan
```

**Seven of the nine configs ran byte-identical code on both arms** — same form, same tile,
in every run — and this candidate's entire diff is in the tuner, which runs at prime time
*outside the timed region*. So those seven ratios are a reading of the harness and not of
the change, and they span **0.9811x to 1.0046x**. That is the resolution of this protocol
on these runs, measured rather than assumed, and it is the number every other row has to be
read against.

The two rows that DIFFER differ through `attn_choice.autotune_looped` — a `looped` /
`single_tile` flip that this candidate does not touch and that the plan-stability probe
measured happening on **both** arms. In run 1 it fell on v43 (config 4); in run 2 it fell
on v42 (config 1). A mechanism that lands on whichever arm the coin gives it is not this
candidate's effect in either direction.

```
weighted_score delta as computed, 3.0 clip per config, /14:   -0.0068
weighted_score delta, honestly:                                0, inside a +/-1.9% floor
```

**This is the predicted result and it is the deliverable.** The prediction registered
before the A/B ran was "config 2's win must survive unchanged, config 3 is capped, so the
score delta is ~0 and the value is in the variance." Config 2 selected `(16,4,1)` on both
arms in every run and every priming; config 3 selected `(64,4,1)` on both arms here and on
v43's arm in 22 of 22 asks. Anyone quoting the -0.0068 is quoting the noise of seven
byte-identical control rows.

### TWO MEASUREMENT HAZARDS FOUND ON THE WAY, BOTH ABOUT REGIME, NEITHER THE CANDIDATE'S

Finding 53 confessed that it *"diagnosed a tuner for using a timer whose regime did not
match its call site, and then validated the fix with a probe whose regime does not match
the tuner's."* This generation walked into the same wall twice more before it stopped.

**1. A one-arm-per-process probe sees no instability at all.** The first draft of
`prime_stability.py --mode fresh` built ONE arm per process — which looks like the
strictest possible isolation, and is why it was written that way. It reports:

```
cfg 2  v42: 1 plan in 8    v43: 1 plan in 8
cfg 3  v42: 1 plan in 8    v43: 1 plan in 8
```

**Both arms perfectly stable on both shapes, including the shape finding 53 measured
moving three ways.** The instability does not exist in a one-arm process; it appears only
once a second model has been built, primed and left resident. A probe that had not modelled
`abba.py` would have concluded "there is nothing here to fix" and closed the generation.

**2. `hot_time` silently degrades to `do_bench` outside `torch.inference_mode()`.** Once
any model has been run inside an inference-mode block, `do_bench_cudagraph` raises

```
RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
```

and `hot_time`'s bare `except Exception` returns `do_bench`'s number instead. The whole
grid then comes back **quantized to the 1.024 µs event tick** — the exact instrument
generation 42 was built to remove, wearing generation 42's name:

```
tile         sw1      sw2      sw3      sw4      (outside inference_mode)
(16,4,1)  20.480   19.456   23.552   20.480      every value an exact multiple of 1.024
(64,4,1)  23.552   23.552   23.552   21.504
```

The tell needs no instrumentation: a `do_bench` reading is an integer multiple of the
quantum and a `do_bench_cudagraph` reading is not. `hot_time`'s docstring already names
this fallback — *"A device or context that refuses capture falls back to `do_bench` and the
caller is none the wiser"* — as a robustness note; it had never been observed to fire.
`_decide_attn` runs inside `inference_mode` at the real call site, so **this is a probe
hazard and not a shipping one**, and it was verified as such
(`bench/probes/g43_stable_tiles/timer_fallback.py`, both contexts, one per process). It
cost this generation two wrong probes and one wrong test first. Anything that measures this
tuner in future must hold an inference-mode context or it is measuring the old timer.

### THE COST OF THE CANDIDATE

One extra pass of the grid at prime time: no extra compilation (the second sweep hits
Triton's JIT cache), no extra allocation (one probe tensor and one reference for all
sweeps, deliberately, so the replicates differ only in *when* they ran), ~1 s against the
frontier's existing 14–67 s of tuning, entirely outside the timed region.

---

## JOB 2 — v16'S FFN CONSTANTS SURVIVE WHERE THEY CAN SCORE, AND FAIL WHERE THEY CANNOT

`bench/kernels/ffn_fused.py` has no autotuner. `_ffn_block` runs at `BLOCK_M = 64,
NUM_WARPS = 8`, two class attributes set on `v16_ffn_megakernel` and justified in the
source as *"measured best at every shape that fits"* — measured at generation 16, with
`do_bench`, which finding 53 has since shown is blind at these sizes.

### WHERE THE CONSTANTS ACTUALLY FIRE, TRACED AND NOT ASSUMED [L36]

`bench/probes/g43_stable_tiles/ffn_call_site.py` builds the frontier on every config and
reports which FFN kernel it runs and at which tile:

```
cfg   tokens   ffn kernel            bm   warps   decided by
  1     8192   none (unfused)         -       -   both predicates decline
  2      128   _ffn_block_normed     16       8   launch_tile (derived) + _pick_warps (swept)
  3      512   _ffn_block_normed     16       8   ditto
  4     2048   _ffn_block_normed     32       8   ditto
  5    16384   none (unfused)         -       -   both predicates decline
  7     8192   _ffn_block            64       8   *** v16's CONSTANTS
  9     8192   none (unfused)         -       -
 10     8192   none (unfused)         -       -
 11     8192   none (unfused)         -       -
 12     2048   _ffn_block_normed     32       8   launch_tile + _pick_warps
 13    65536   _ffn_block            64       8   *** v16's CONSTANTS
```

plus config 6 (1 280 000 tokens), which `amortizes` admits on the same arithmetic. So the
un-swept constants govern **configs 6, 7 and 13 only**; the four launch-bound rows run
`_ffn_block_normed` at a `block_m` **derived per shape** from the measured SM count, with
the warp count swept at prime time by `v34._pick_warps`. Half of the brief's premise —
"`_ffn_block` and `_ffn_block_normed` are 18.7% of config 6" — is right about the kernel
and the config; the `_ffn_block_normed` half of it is already parameterised.

### THE SWEEP

`bench/probes/g43_stable_tiles/ffn_tile_sweep.py`: every `(block_m, num_warps)` that
`ffn_fused.fits` accepts on the measured device, at the real shapes, under **both** timers,
correctness-gated against the un-fused path's own arithmetic at the locked tolerance before
any arm is timed, two passes each.

```
cfg 13  (65 536 tokens, d_model 128)
  flushed   best (64, 8) at 152.576 us   v16's (64, 8)   1.000x   14 distinct of 15
  hot       best (64, 8) at 137.980 us   v16's (64, 8)   1.000x   15 distinct of 15

cfg  6  (1 280 000 tokens, d_model 128)
  flushed   best (64,16) at 2663.424 us  v16's (64, 8)   1.000x   15 distinct of 15
  hot       best (64,16) at 2641.692 us  v16's (64, 8)   1.000x   15 distinct of 15
```

**v16's constants survive, and the survival is not an artefact of a blind instrument.** At
these sizes the kernel runs 138–2642 µs, so the 1.024 µs quantum is 0.04–0.7% of the
quantity being measured, and the flushed grid is not degenerate: 14–15 distinct values of
15, 0–3 arms on the quantum. Finding 53's resolution failure simply does not exist here.
`(64,16)` ties `(64,8)` on config 6 to within 0.01%, which is a tie and not a challenger.
**Generation 16's original sweep is validated on the two shapes where the constants carry
weight, and no change is warranted.**

### CONFIG 7 IS THE EXCEPTION, IT IS FINDING 53'S DEFECT EXACTLY, AND IT SCORES NOTHING

```
cfg  7  (8192 tokens, d_model 32)                        run 1     run 2
  flushed   best (16, 4)   v16's (64, 8)                 1.020x    1.143x
                                    distinct of 20 arms      8         8
                                    arms on the quantum     17        19
  hot       best (16, 4) at 2.322 / 2.341 us
            v16's (64, 8) at 3.122 / 3.127 us            1.345x    1.336x
                                    distinct of 20 arms     20        20
                                    arms on the quantum      0         0
```

This is finding 53's table of ties, in a different file, eight of twenty distinct values
and nineteen of twenty arms pinned to the event tick — and the flushed timer's own answer
moves between 1.020x and 1.143x across two runs of the identical sweep, which is the same
non-reproducibility that finding 53 diagnosed as quantization rather than noise. The margin
the hot timer resolves, **1.345x / 1.336x replicated across processes**, is larger than the
1.28x that justified generation 42.

**And it is worth exactly zero.** Config 7 sits at **4.302 speedup, past the 3.0 clip**, so
0.80 µs per call × 4 layers buys nothing whatsoever in `weighted_score`. The two shapes
where `_ffn_block` could pay are the two where its tile is already right.

### DISPOSITION FOR JOB 2

* **The constants survive.** On configs 6 and 13 they are optimal or tied under a timer
  with three digits of resolution, and the sweep that chose them at generation 16 was not
  blind at those sizes. Closed.
* **Config 7 is a real 1.34x op-level margin on a capped row.** It is recorded, replicated
  and **deliberately not acted on**: an FFN autotuner would be a second mechanism riding
  into this candidate's measurement, which is the error finding 53 refused to make and this
  one refuses too. Pre-registered as the next generation below.
* **`v34._pick_warps` ranks with `do_bench`, flushed**, on `_ffn_block_normed` at 128–2048
  tokens — squarely in the regime where the quantum bites. It has returned "derived 8
  warps, confirmed" on all four launch-bound configs, i.e. it has never displaced anything,
  which is exactly what a tie table does. Not measured here and not fixed here. Named.
* `proj_gemm.plan()` already uses `do_bench_cudagraph` and documents why; `attn_qkv_fused`
  and `attn_outproj` still use `do_bench` but belong to v27/v31, which do not ship.

---

## PRE-REGISTERED FOR THE NEXT GENERATION

**An `autotune_ffn_tile` consistent with this one.** Same shape as `autotune_tile` after
generation 43: enumerate the `(block_m, num_warps)` grid `ffn_fused.fits` admits on the
measured device, correctness-gate every arm at the locked tolerance, rank with
`attn_single_tile.hot_time`, replicate and reduce by the per-arm floor, and require
`DECISIVE` against the derived tile. `BLOCK_M`/`NUM_WARPS` become the fallback rather than
the answer, which is what `choose_tile` is to `autotune_tile`.

**Its expected score is zero on the announced matrix**, and that should be said before it
is built rather than discovered afterwards: the only shape it moves is capped. It is worth
building for the reason generation 42 was — a tuner that derives its tile is right on
shapes nobody has swept — and it should be measured for *stability*, not for speed.

**`v34._pick_warps` should move to `hot_time` at the same time**, since it is the same
defect in the same file's other kernel, at sizes where the quantum certainly bites.

---

## PROPOSED LESSONS

Not appended to `docs/findings/00-learnings.md` — `ben` is ahead and it would collide.

### L64 — When the noise is one-sided, replicate to REDUCE it, never to VOTE on the answer

Finding 53 established that contamination on this harness only ever makes a reading
slower, and then proposed a fix that requires two rankings to agree. Those two statements
are incompatible and it took building the fix to see it. If the noise is one-sided, it
lands on an arbitrary arm, and a `min()` over the grid returns *the arm the noise missed* —
so two sweeps disagreeing is the expected outcome even when the underlying margin is 28%,
and requiring agreement makes the rule revert more often the noisier the machine is. The
measured cost was the parent's only scoring row, lost in 5 of 10 processes.

The correct use of a replicate under one-sided noise is to take the **floor per arm** and
then decide once. That estimator is already this project's house rule for timing a card
whose clocks will not lock; the thing that was new was applying it *inside* a selection
rule rather than to the selection rule's output.

The general form: **before choosing how to combine repeated measurements, ask what shape
the noise has.** Voting, averaging and flooring are right under different noise models and
wrong under the others, and "replicate it" is not a decision until you have said which.
A vote is right for symmetric noise around a true value. A floor is right for one-sided
contamination of a true minimum. Picking the wrong one does not merely waste the
replicate — it can be worse than not replicating at all.

### L65 — An isolation that removes the phenomenon is not a control, it is a different experiment

The first stability probe of this generation gave every arm its own process: no
co-residency, no allocator sharing, no interference. It reported both candidates perfectly
stable on both shapes and would have closed the generation as "nothing to fix". The
instability being investigated **exists only when a second model is resident**, which is
the condition `bench/abba.py` creates and therefore the condition under which every number
this project ranks candidates on is produced.

Isolation is the reflex here for good reasons — finding 05's co-residency spill, finding
45's construction-time planners, the one-config-per-subprocess rule — and every one of
those is about isolating a *measurement*. This was a measurement of a *decision*, and the
decision is made in the contaminated environment on purpose. **The probe must reproduce the
call site's environment, including the parts of it that look like contamination**, or it
measures a system that does not ship.

Corollary, and the cheap check: when a probe reports the phenomenon absent, that is a
result about the probe until it is a result about the code. Ask what the probe removed.

### L66 — A silent fallback inside an instrument is a silent change of instrument

`hot_time` wraps `do_bench_cudagraph` in `except Exception: return do_bench(...)`, and says
so in its docstring, with a reason: failing closed on a tuner is worse than degrading. That
reasoning is defensible. What is not defensible is that the degradation is **invisible in
the number**: the caller gets a float, the reason string still says `hot_time`, and the
whole grid quietly reverts to the 1.024 µs instrument the previous generation was built to
remove.

It fires for a real, non-exotic reason — calling it outside `torch.inference_mode()` after
any model has run inside one — which is a condition every probe and test in this repo can
meet by accident, and three of them did in one afternoon.

The general form: an instrument that can silently become a *different instrument* must say
which one it was. A timer that falls back should return, or record, the path it took, and
anything that names an instrument in a reason string should name the one that actually ran.
The arithmetic tell here was free — a `do_bench` reading is an exact multiple of the event
quantum and a graph reading is not — and had it been checked in an assertion rather than by
eye, none of the three wrong probes would have gone anywhere.
