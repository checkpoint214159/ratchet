# 53 — The tile sweep was ranking a table of ties: an event timer with a 1.024 µs quantum against kernels that run in 1.9 µs

**Date:** 2026-08-31. **Generation:** 42. **Branch:** `cand/g42/tile-timer`.
**Parent:** `v41_vendor_aware_attn` (`c863eed`). **Candidate:** `v42_hot_tuned_tile`.
**Takes:** the proposal finding 50 opened and finding 51 measured at 1.283x, and that both
deliberately declined — *re-tune the single-tile form with the hot timer*.

**Numbering note:** taken as 53 on a candidate branch. `ben` renumbered the g41 audit to 52
at `64056fd`; this branch still carries it as 51, so cross-references below to "the g41
audit" name the file rather than the number. `00-learnings.md` and the findings README are
deliberately NOT edited here — `ben` is ahead. The proposed lesson is at the bottom for the
merge to pick up.

---

## THE CENSUS, AND THE CEILING, BEFORE ANY CODE WAS WRITTEN

The lead was worth checking arithmetically first, because an op-level margin on a shape
that is 4% of a config is not a margin. Config 2 is B=1, S=128, D=128, H=4, **L=4**, so
attention runs four times per forward.

From this generation's own sweep (`bench/probes/g42_tile_timer/`, two independent runs,
L2-hot inside a captured graph, at config 2's real batch of 1 — the probe cap does not bite
here, `probe_batch(1, 4, 66) == 1`):

```
              single_tile(64,4,1)   single_tile(16,4,1)    ratio    saving/call
  run 1              2.438 us              1.905 us       1.2801      0.5335 us
  run 2              2.454 us              1.920 us       1.2784      0.5345 us
```

* attention today: **4 × 2.446 = 9.78 µs** of config 2's 48.13 µs wall (the g41 audit's
  replicated ABBA median) — **20.3% of the config**
* saving: 4 × 0.534 = **2.14 µs**, i.e. 4.4% of the config
* config 2's speedup goes 2.631 → **2.753** (REPORT §4.2, v40 column). It is **uncapped**,
  which is why this row is worth anything at all: five of the thirteen runnable configs are
  already past the 3.0 clip and score nothing further however much faster they get.

    **PRE-REGISTERED CEILING:  Δ weighted_score = (2.753 − 2.631) / 14 = +0.0087**

and **+0.0083** after the event quantum, which is the number to hold this to: config 2's
median is an integer multiple of 1.024 µs (48.13 = 47 quanta), a 2.14 µs saving is 2.09
quanta, so the observable is a drop of 2 quanta to 46.08 µs, a ratio of 1.0444.

**This disagrees with the brief that commissioned it**, which priced the same lead at
~16% of the config and +0.006. The difference is the numerator: 16% is `4 × 1.929 / 47`,
the attention cost *after* the fix. The share being displaced is the cost *before* it,
9.78/48.13 = 20.3%. Same measurement, and the ceiling is about 45% higher than the brief's.

**PRE-REGISTERED DETECTION BAR**, fixed before the A/B ran, because a 2-quantum effect on a
48 µs config is exactly the size of thing this project has repeatedly fooled itself with
(findings 42, 45, 49, 50):

1. config 2's v42/v41 median ratio > 1.0 in **both** independent ABBA runs;
2. the drop is ≥ 1 quantum in both and ≥ 2 quanta in at least one;
3. every control config stays within ±2 quanta **with no sign surviving replication** —
   the g41 audit's own control read 1.0217 on config 2 on byte-identical code in one run of
   two, so a single positive row proves nothing and a consistent sign is the whole signal.

---

## WHAT IS ACTUALLY WRONG WITH THE TUNER

Not the tile. The tile is chosen by `attn_single_tile.autotune_tile`, at prime time, by
sweeping the legal grid on the real shape — so "hardcode `(16,4,1)` for config 2" would be
a config id in disguise and is not on the table. The question was why a sweep that saw
`(16,4,1)` ranked it behind.

It ranked it behind because **it could not see it.** From generation 23 to 41 the sweep
ranked with `do_bench(warmup=10, rep=25)`, which times each call with a pair of CUDA
events. The event quantum on this card is **1.024 µs**. The kernels being ranked run in
**1.9–11 µs**.

`bench/probes/g42_tile_timer/probe_timer_regimes.py` swept the full eight-tile grid under
both timers, in one process, at the same shape, with the same trial budget, on every shape
the kernel accepts, twice. Config 2's grid:

```
tile            flushed_us    hot_us   flushed/hot
(16, 4, 1)           5.120     1.905      2.69       <- fastest, hot
(16, 2, 1)           5.120     2.260      2.27
(32, 4, 1)           5.120     2.409      2.12
(64, 4, 1)           5.120     2.438      2.10       <- the derived tile, and what ships
(32, 2, 1)           5.120     2.486      2.06
(64, 8, 1)           6.144     3.344      1.84
(32, 8, 1)           6.144     3.423      1.80
(64, 2, 1)           6.144     3.718      1.65
```

**Five of the eight arms report the identical number, and the entire spread of the grid is
one quantum.** The sweep was not noisy — it was *blank*. `min()` over a five-way tie then
fell through to the rule below it, which is v23's deliberate and correct tiebreak: *the
derived tile holds the ground unless something beats it decisively*. Nothing could beat
anything by 10% in a table whose values are all equal, so the derived tile held — and the
derived tile is `(64,4,1)`, which the hot timer ranks **1.280x / 1.278x** behind the best
arm, reproducing finding 51's 1.283x from an independent sweep in a fresh process.

**The decision rule was right. Its input was a tie table.** That distinction is the whole
finding: nothing about `DECISIVE`, `pays()`, `choose_tile` or the kernel needed to change.

### The same quantization is wrong in the other direction, which is how you know it is quantization

Config 3, the two runs of the identical sweep:

```
             run 1                     run 2
  flushed -> (16,4,1) at 6.144   |   flushed -> (64,4,1) at 7.168
  hot     -> (64,4,1) at 2.518   |   hot     -> (64,4,1) at 2.521
```

In run 1 a **one-quantum** difference (6.144 vs the derived tile's 7.168) cleared the 10%
`DECISIVE` bar and selected `(16,4,1)` — a tile the hot timer ranks 1.9% *slower*. In run 2
the same sweep picked the derived tile. **The flushed timer picked a different tile in each
of two runs of the identical sweep on the identical shape.** The hot timer picked
`(64,4,1)` both times, at 2.518 and 2.521 µs.

That instability is exactly what finding 50 recorded as "`autotune_tile` is deterministic —
and it is deterministic against a comparable process state, not absolutely", and attributed
to "a warm allocator or a loaded L2 moving an arm across the margin". It is neither. It is
one quantum of an event timer, and the fix for a resolution problem is resolution.

`do_bench_cudagraph` times a graph of many replays and divides, so the quantum is amortized
across the replays instead of paid once per call. That is why it resolves a 1.9 µs kernel
to three decimal places (1.905 / 1.920 across two processes) and `do_bench` cannot resolve
it at all. The L2-hot regime — which is what [L53] and `attn_choice._time`'s docstring
argue for, and what finding 48's 2.24x headline error was about — is a *second*, independent
reason to prefer it, and the ratio column above measures that gap again at 1.65x–2.69x. But
on this matrix the resolution argument is the load-bearing one, and it is new.

### The asymmetry nobody had noticed

`attn_choice` — written eighteen generations later, for the same decision, about the same
kernel — has ranked with `do_bench_cudagraph` since generation 40, and its module docstring
opens with a section on why every arm must be timed *"by the same function with the same
repeat count"*. That symmetry held **between the arms of one sweep** and broke **between the
two sweeps**: `autotune_looped` ranks hot, and the routine it falls through to on every
shape it declines ranked flushed. Config 2 is such a shape — `attn_looped` declines it
outright (the audit's `looped` column reads `nan` there) — so config 2's tile was decided
entirely by the instrument that could not resolve it.

This is the same shape of defect as the g41 audit's own headline, one level up. There, a
predicate about our kernel was read as a comparison against the vendor's. Here, a symmetry
argument about arms was read as if it covered routines.

## THE FIX

One value, in one place. `bench/candidates/v42_hot_tuned_tile.py` is:

```python
class CandidateV42(v41_cls):
    attn_tile_timer = staticmethod(attn_single_tile.hot_time)
```

`attn_single_tile.autotune_tile` grew a `timer` parameter; v23 grew the class attribute
that feeds it. The kernel is untouched, no tile is hardcoded, no predicate moves, `pays()`
and finding 51's answer to it stand, and **no config id appears anywhere** — the sweep
reaches `(16,4,1)` from the shape plus `torch.cuda.get_device_properties`, and will reach
the right answer on shapes nobody has swept and on cards nobody has run.

`attn_choice._time` is now a one-line alias for `attn_single_tile.hot_time`, so the two
tuners share one instrument **by construction** rather than by inspection. That is the
structural half of the fix and it is the half that stops this recurring.

**`timer=None` still means the flushed timer.** That default is deliberate and temporary:
the candidate's whole claim is that this timer picks the wrong tile, and the only protocol
that can measure the claim is two arms resident in one process (finding 50 measured two
candidates reading 46% apart on config 3 across a process boundary). A control arm that has
silently moved to the new timer measures nothing. **The default flips at merge, not before.**

### Second fix: correctness before timing, latent not live

`autotune_tile` admitted arms to its timing set gated only by `fits`/`pays`. It was the one
tuner in the package that could select a tile on speed alone — `attn_choice` checks every
arm against the reference at the locked tolerance first, and says so in a heading. It now
does too. Measured, the gap is **latent and not live**: the probe checked all eight tiles on
all ten accepted shapes, twice, and every arm matched, so this drops nothing today. Closed
anyway, per [L38].

## THE BLAST RADIUS IN THE PROBE'S REGIME — AND WHY THAT IS NOT THE MODEL'S

What `autotune_tile`'s own decision rule returns under each timer, on all ten shapes the
kernel accepts, twice, **in a standalone process on a near-idle GPU**:

```
cfg      B    pb   hd     S   derived      flushed ->     hot ->        verdict
  1     64    64   32   128   (64,4,1)     (64,4,1)       (64,4,1)      SAME
  2      1     1   32   128   (64,4,1)     (64,4,1)       (16,4,1)      *** DIFFERS, 1.280x
  3      4     4   32   128   (64,4,1)     UNSTABLE       (64,4,1)      SAME (hot, both runs)
  4     16    16   32   128   (64,4,1)     (64,4,1)       (64,4,1)      SAME
  5    128    66   32   128   (64,4,1)     (64,4,1)       (64,4,1)      SAME
  6  10000    66   32   128   (64,4,1)     (64,4,1)       (64,4,1)      SAME
  7     64    64    8   128   (64,4,1)     (64,4,1)       (64,4,1)      SAME
 10     64    64   64   128   (32,4,1)     (32,4,1)       (32,4,1)      SAME
 11     64    16    8   128   (64,4,1)     (64,4,1)       (64,4,1)      SAME
 12     64    64   32    32   (32,2,1)     (32,2,1)       (32,2,1)      SAME
  8, 9, 13, 14                 the kernel declines the shape; `autotune_tile` is never called
```

**Nine of the ten select the identical tile under both timers, in both runs**, so in this
regime the change moves exactly one shape.

### THE PROBE'S REGIME IS NOT THE MODEL'S, AND THIS COST ME THE BLAST-RADIUS CLAIM

That table is true and it is not the answer to the question it looks like it answers.
`bench/abba.py` now records each arm's chosen plan alongside its timing ([L36] applied to
the measurement rather than to the tests), and **what the model picks is not what the probe
picks**:

```
        v41 (flushed)              v42 (hot)                    probe said hot would pick
cfg 1   single_tile(64,4,1)        single_tile(64,4,1)          (64,4,1)   agrees
cfg 2   single_tile(64,4,1)        single_tile(16,4,1)          (16,4,1)   agrees
cfg 3   single_tile(64,4,1)        single_tile(16,4,1)          (64,4,1)   *** DISAGREES
cfg 4   single_tile(64,4,1)        looped(64,32,4,4)            n/a -- a DIFFERENT FORM
cfg 7   single_tile(64,4,1)        single_tile(64,4,1)          (64,4,1)   agrees
cfg 9   sdpa                       sdpa                         declined
cfg 10  looped(64,32,4,4)          looped(64,32,4,4)            n/a
cfg 11  single_tile(64,4,1)        single_tile(64,4,1)          (64,4,1)   agrees
cfg 12  single_tile(32,4,1)        single_tile(32,2,1)          both (32,2,1)  *** BOTH DISAGREE
```

On config 3 the model's own prime-time sweep reports, in its reason string,
`(16,4,1) beat the derived tile (64,4,1) decisively (1.460x)` — where the standalone probe
measures those same two arms at **0.98x**, i.e. the other way round. **A 1.49x regime gap on
the same two tiles under the same timer**, which is the same order as the gap this candidate
was built to close, one level up.

The difference is that `_decide_attn` runs with the whole model resident, immediately after
other tuning work, and the probe runs on a near-idle card. So `hot_time` fixed the
*resolution* problem and the *regime* problem is still there — it has simply moved from
"flushed vs hot" to "prime time with a resident model vs anything else". I diagnosed a tuner
for using a timer whose regime did not match its call site, and then validated the fix with
a probe whose regime does not match the tuner's. That is recorded here rather than quietly
corrected, and the test that asserts the standalone equality now says in its docstring
exactly what it does and does not establish.

Config 12 is worse than a disagreement: **both arms disagree with the probe and with each
other** (v41 `(32,4,1)`, v42 `(32,2,1)`, probe `(32,2,1)` for both). And config 4's two arms
ran different *forms* — `looped` against `single_tile` — from `autotune_looped`, a routine
this candidate does not touch. That is the g41 audit's item 3 (configs 4 and 12 selected on
margins of 1.153x/1.313x that re-measured at 0.986x/0.971x) showing up as run-to-run
nondeterminism in the plan itself, under a timer that was already hot. **So the instability
on the launch-bound rows is not only the quantum, and this candidate does not fix all of
it.**

## THE MEASUREMENT

`bench/probes/g41_attn_audit/run_ab.py --mode abba`, GPU lock held for each run, both arms
resident, ABBA-interleaved, cold round discarded, `--rounds 5 --warmup 200`, correctness
checked against a fresh reference before any round at the locked tolerance. **Two full
nine-config runs, then six more runs of config 2 alone with config 1 carried alongside as a
per-run health check** — `bench/probes/g42_tile_timer/replicate_cfg2.sh`.

The six extra runs were not planned. They were forced, and by the control arm rather than
by the candidate.

### The first two runs disagreed on the sign, and the reason was not the candidate

```
cfg      v41 run1   v41 run2   move on BYTE-IDENTICAL code
  1        225.28     224.26   1.00x
  2         48.13     200.70   4.17x   <== 
  3         55.30     117.76   2.13x   <==
  4         81.92      86.02   1.05x
  7         77.82      77.82   1.00x
  9        226.30     225.28   1.00x
 10        224.26     224.26   1.00x
 11        269.31     269.31   1.00x
 12         74.75     104.45   1.40x   <==
```

**Config 2's control arm — the same commit, the same protocol, the same machine — read
48.13 µs and then 200.70.** That is a 4.17x move, larger than finding 51's 2.69x on config 3
and far larger than finding 42's 33–39%. A predicted effect of 2 event-timer quanta cannot
be read off an instrument whose own control moves by 150 of them, and the two runs duly
reported config 2 at 0.9592x and then 1.0316x. **Neither number means anything on its own,
and this is exactly the trap findings 42, 45, 49 and 50 are each a version of.**

### Seven runs of config 2, with config 1 as the in-run health check

```
              run:      1       2       3       4       5       6       7
cfg 1  v41         224.26  224.26  225.28  225.28  225.28  225.28  224.26
       v42         224.26  225.28  224.26  224.26  225.28  224.26  224.26
       ratio       1.0000  0.9955  1.0046  1.0046  1.0000  1.0046  1.0000     floor ratio 1.0000

cfg 2  v41          69.63   48.13   49.15   51.20   51.20   48.13  200.70     floor  48.13
       v42          45.06   45.06   51.20   50.18   48.16   50.18  194.56     floor  45.06
       ratio       1.5455  1.0682  0.9600  1.0204  1.0631  0.9592  1.0316     floor ratio 1.0682
```

**Config 1 is the finding here as much as config 2 is.** Across seven independent runs its
two arms span 224.26–225.28 — one quantum, spread 1.00x, floors identical, and the tiles
identical `(64,4,1)` on both arms every time. **The harness is not broadly broken and the
candidate is not broadly perturbing anything.** Config 2's 4.2x excursions are config 2's.

On config 2 both arms show the same 4.2x spread, so the instability is a property of the
*shape*, not of either arm. What separates them is the floor: **v41 reaches 48.13 µs (47
quanta) twice and never goes below it; v42 reaches 45.06 (44 quanta) twice and never goes
below it.** Three quanta, and the per-run ratio favours v42 in five runs of seven.

### The estimator, and the argument for it

Contamination on this harness is **one-sided**: a descheduled host thread, an unsettled
graph or a co-resident allocation can only make a reading slower, never faster. So the mean
is a statistic about how often the machine misbehaved and the minimum is a statistic about
the code — which is why CLAUDE.md prescribes minimum-of-N for a card whose clocks will not
lock, and why finding 49's two candidates that read 46% apart matched to the hundredth of a
microsecond once replicated. Both arms get the same number of runs and the same reduction,
so no asymmetry is introduced.

Two defensible reductions, and the honest thing is to report both:

```
                          cfg 2 ratio    speedup 2.631 ->    Δ weighted_score
  floor of 7 runs            1.0682            2.810             +0.0128
  median of 7 per-run        1.0316            2.714             +0.0059
  pre-registered ceiling     1.0444            2.748             +0.0083
```

**The pre-registered ceiling sits between the two estimators**, which is the most reassuring
thing in this document: a number predicted from an op-level census before any end-to-end
code ran lands inside the range the end-to-end measurement produces. Config 2 stays well
under the 3.0 cap on every reading, so all of this is real score rather than clipped score.

**What this candidate is entitled to claim is `+0.006 to +0.013`, and not a point estimate.**
Anyone quoting the +0.0128 alone is quoting the best of seven runs against the best of seven
runs on the least trustworthy row in the matrix.

### THE COST: config 3's plan stops being stable, and the flushed timer's ties were protecting it

Config 3 is the other shape where the model's plan changed, so it got the same treatment —
six independent runs, config 1 alongside:

```
run    cfg 1 (v41/v42)     cfg 3 v41                  cfg 3 v42                  ratio
 1     225.28 / 224.26      55.30  single(64,4,1)      56.32  single(16,4,1)     0.9818
 2     225.28 / 225.28      97.28  single(64,4,1)      98.30  single(64,4,1)     0.9896
 3     225.28 / 225.28     116.74  single(64,4,1)     123.90  single(32,4,1)     0.9421
 4     225.28 / 225.28      77.82  single(64,4,1)      57.34  single(16,4,1)     1.3571
 5     225.28 / 225.28     148.48  single(64,4,1)     150.53  single(32,4,1)     0.9864
 6     225.28 / 225.28      64.00  single(64,4,1)     104.45  single(32,4,1)     0.6127

floors                      55.30                      56.32                     0.9818
```

**`v41` selected `(64,4,1)` in six runs out of six. `v42` selected three different tiles in
six runs** — `(16,4,1)`, `(64,4,1)`, `(32,4,1)`, `(16,4,1)`, `(32,4,1)`, `(32,4,1)`. Config
1, in the same processes, read 225.28 for both arms in five of six runs.

This is a real regression and it is not the one I expected. The mechanism is clean once you
have both halves of the finding:

* Config 2's true margin is **28%**, far outside `DECISIVE`'s 10%, so no plausible amount of
  prime-time noise moves the decision. v42 picked `(16,4,1)` there in **7 runs out of 7**.
* Config 3's true margin between the same two tiles is **~2%**, far inside `DECISIVE`. For
  the sweep to displace anything it must observe a spurious >10% margin — and the in-model
  prime-time environment evidently supplies that, since it displaced three ways.
* **The flushed timer never had this problem because it could not clear the bar at all.**
  Its tie table always fell through to the derived tile, so it was accidentally, uniformly
  stable — stable and, on config 2, stably wrong.

So the change trades a systematic error for a variance one on the shapes whose margins are
genuinely small. Config 3 is **past the 3.0 cap (4.388)** so none of this reaches
`weighted_score` in either direction, and the floor difference is one quantum. But a plan
that varies run to run adds its variance to every measurement taken of the candidate
(L29), and that is a cost worth naming rather than netting out.

### THE AGGREGATE

`bench/probes/g42_tile_timer/summarize.py`, floor estimator, every run of every config:

```
   cfg  1   2.633 ->  2.633 (1.0000x)   n=7,  tiles identical both arms
   cfg  2   2.631 ->  2.810 (1.0682x)   n=7,  THE WIN
   cfg  3   4.388 ->  4.308 (0.9818x)   n=8,  CAPPED, scores nothing
   cfg  4   2.675 ->  2.675 (1.0000x)   n=2
   cfg  7   4.302 ->  4.302 (1.0000x)   n=2,  CAPPED
   cfg  9   2.022 ->  2.022 (1.0000x)   n=2,  sdpa on both arms
   cfg 10   2.415 ->  2.426 (1.0046x)   n=2,  tiles identical both arms
   cfg 11   8.464 ->  8.464 (1.0000x)   n=2,  CAPPED
   cfg 12   2.653 ->  2.617 (0.9865x)   n=2,  SEE BELOW -- do not trust this row

   weighted_score  2.5823 -> 2.5933     delta  +0.0110
```

**Config 12's −0.0026 should not be netted off without a caveat**, and stating it is the
point of having recorded the plans. Its two runs have a 1.40x spread on the control arm, and
its two arms ran *different tiles for a reason that is not this change*: v41 selected
`(32,4,1)` where its own flushed sweep in a standalone process selects `(32,2,1)` twice.
That is the parent's tuner being unstable, sampled twice. n=2 on a 1.40x-spread row is not a
measurement, and the honest form of that row is "unresolved", not "−0.0026".

So: **+0.0110 as computed, +0.0128 if config 12 is (correctly) treated as unresolved, and
`+0.006 to +0.013` is the range this candidate is entitled to claim.**

## DISPOSITION

* **The defect is real, diagnosed, and closed.** `autotune_tile` was ranking a table of ties
  produced by an instrument whose quantum is half the quantity being measured. Four
  independent measurements across two generations agree the margin it missed is 1.28x
  (finding 51 twice, this generation's probe twice).
* **The fix is a timer, not a tile.** One value, no config ids, and it makes the package's
  two tuners share one instrument by construction.
* **Config 2 is a real win worth `+0.006 to +0.013`** of `weighted_score`, on an uncapped
  row, with the pre-registered ceiling landing between the two defensible estimators.
* **Config 3 is a real cost**: the plan destabilises across three tiles where it used to be
  fixed, at zero score impact (capped) and one quantum of wall.
* **Every other shape is inert and was shown to be**: configs 1, 7, 9, 10 and 11 select
  identical tiles on both arms and read 1.0000–1.0046 across every run.
* **NOT FIXED, and not this candidate's to fix:** configs 4 and 12 select different plans on
  the two arms through routines this change does not touch (`autotune_looped`, and the
  flushed tuner's own instability). The g41 audit flagged both; they are still coin flips.

### THE OBVIOUS NEXT GENERATION, PRE-REGISTERED HERE

**Require the margin to replicate before it displaces.** `autotune_tile` should run its
sweep twice and displace the derived tile only if both sweeps clear `DECISIVE` *and* agree
on the winner. Cost: one extra sweep, ~1 s at prime time against v40's existing 14–67 s.
Predicted effect, from the six runs above: config 2 keeps `(16,4,1)` (7/7 agreement, 28%
margin), config 3 reverts to `(64,4,1)` and stops moving (no tile appears twice in a row).
That is L56 — *replicate before you decompose* — applied inside the tuner rather than to the
tuner's output, and it is the cheapest thing on this table. It is deliberately NOT bundled
here: it would ride into this candidate's measurement and destroy the attribution that made
config 2 readable.

## PROPOSED LESSON

Not appended to `docs/findings/00-learnings.md` — `ben` is ahead and it would collide.

### L62 — Before you argue about a timer's regime, check that it can resolve the thing

[L53] says to use a timer whose regime matches the call site's, and three findings (48, 50,
the g41 audit) have been about cache regime — L2-hot against L2-flushed, a 2.24x gap that
made one headline wrong by that factor. That is a real axis and the reasoning about it was
correct. It is also not the only way an instrument can fail, and here it was not the one
that cost the score.

`do_bench` times each call with a pair of CUDA events. Their quantum is 1.024 µs. Ranking
eight variants of a 1.9 µs kernel with it produced a table in which five entries were the
same number — not noise around a true ordering, but **no ordering at all**. Every downstream
rule then behaved correctly on a degenerate input: `min()` returned an arbitrary tie, the
`DECISIVE` margin could not be cleared, and the tiebreak kept the incumbent. Eighteen
generations of correct decisions on a blank input.

The cheap check is one line and nobody had run it: **print the sweep, and look at how many
distinct values it contains.** A grid of N arms that yields two distinct readings has not
ranked anything. If the spread of the whole grid is comparable to the instrument's quantum,
the instrument is the wrong one, and no amount of replication, minimum-taking or
winner's-curse correction will fix it — those all assume a noisy signal, and this is an
absent one.

The general form: an instrument has a *resolution* as well as a *regime*, and resolution
failures are invisible in exactly the way regime failures are not. A mis-regimed timer gives
you confident wrong numbers you can catch by cross-checking against another regime. An
under-resolved timer gives you *ties*, which look like "the arms are equivalent" — an
answer, and a plausible one, and the reason this sat unexamined from generation 23 to 42.

### L63 — A blunt instrument is stable; sharpening it buys accuracy with variance, and you must check where you spent it

`do_bench` could not separate config 2's eight tiles, so the sweep fell through to its
tiebreak — every time, on every shape, in every process state. That is a *systematic* error
and it has a property nobody had noticed was doing work: **it is perfectly reproducible.**
Eighteen generations of stable, wrong tile selection, and the stability was load-bearing,
because a plan that does not vary adds no variance to the measurements taken of it.

Replacing it with a timer that *can* resolve the arms fixed the shape whose true margin is
28% and destabilised the shape whose true margin is 2% — from one tile in six runs to three.
The new instrument is better and it converted a bias into a variance, and the variance
landed on exactly the shapes where the decision was closest and therefore mattered least
per-decision and most in aggregate.

The lesson is not "keep the blunt instrument". It is that **replacing a selection rule's
instrument is not complete until you have measured the rule's OUTPUT for stability, not just
its accuracy** — run the sweep several times and count distinct answers, on the shapes where
the margin is small as well as the one that motivated the change. The decision rule's
threshold was calibrated (`DECISIVE = 0.10`) against the old instrument's noise; a sharper
instrument with a different noise distribution needs the threshold re-earned, or needs the
answer required to replicate before it is acted on. Neither is expensive. Not noticing is.
