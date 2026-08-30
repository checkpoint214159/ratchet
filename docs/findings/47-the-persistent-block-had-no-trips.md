# 47 — The persistent FFN block never looped, and 4.5 points of its 8.2% were the size of the search

**Date:** 2026-08-30. **Generation:** 39. **Branch:** `cand/g39/persistent-ffn`.
**Parent:** `v38_stream_fallback` (`7cee27c`). **Proposal:** F-03.
**Verdict:** DECLINED. No candidate built. **F-03 never measured its own mechanism** — at
the tile its launcher derives, the grid-stride loop runs exactly once, on all four configs
that dispatch the kernel. Measured at a tile where the loop *does* run, the idea is real
but worth **+0.0063 of `weighted_score`, not +0.0138**, on two configs only, at 1.8% per
config against a ±7% floor — unresolvable by any instrument we have.

**Numbering note:** taken as 47 on a candidate branch while `ben` is ahead. Renumber at
merge if it clashes.

---

## What F-03 asked for

Rewrite `bench/kernels/ffn_fused._ffn_block_normed` in persistent grid-stride form: clamp
the grid to a multiple of the SM count and turn the row tile into a `for` loop, so `w1`,
`w2`, both biases and all four LayerNorm parameter vectors are loaded **once per program**
instead of once per row tile. The proposal measured that form at

    tokens   frontier    persistent best   ratio
       128    7.632 us       7.456 us      0.977x
      2048   14.714         13.514         0.918x
      8192   37.948         34.873         0.919x
     16384   62.712         56.577         0.902x

and priced it at **+0.0138 of `weighted_score`** on configs 2, 4 and 12.

It is a careful proposal. It checks its output against the parent kernel on every arm, it
reports the null half of its own hypothesis, and it pre-registers a region (the
8192-token rows) that must *not* be extended into. The defect is in neither the kernel nor
the honesty; it is in the sweep.

---

## 1. THE MECHANISM NEVER RAN IN THE MEASUREMENT THAT PRICED IT

The saving F-03 describes is proportional to `(trips per program - 1)`. Its launcher runs
`min(grid, ntiles)` programs over `ntiles` tiles. At the proposal's own four winning arms:

```
 tokens   frontier BM/grid   persistent best arm       programs   trips/program
    128        16 /   8      BM=16 grid=66                    8        1.00
   2048        32 /  64      BM=32 grid=66                   64        1.00
   8192        64 / 128      BM=64 grid=132                 128        1.00
  16384        64 / 256      BM=16 grid=66                   66       15.52
```

At 128, 2048 and 8192 tokens **the winning "persistent" arm is the frontier's own launch
geometry with a loop that runs exactly once.** Same tile, same program count, same weight
load per program, nothing hoisted, because there is no second trip to hoist it out of. The
only genuinely persistent winner is at 16384 tokens — which is config 5, where `one_wave`
declines and the kernel is never dispatched at all.

**This is not a badly-chosen grid; it follows from the tile the launcher derives.**

```
one_wave(...)   is True  iff   ntiles <= sm_count * blocks_per_sm
persistence     requires       ntiles >  grid,  and F-03's grid is sm_count * k
```

Both weight matrices are a 64 KB on-chip image, so `blocks_per_sm == 1` on every announced
config that reaches the branch, and `one_wave` degenerates to `ntiles <= sm_count`. At
`launch_tile`'s own answer, `grid = sm_count * k >= sm_count >= ntiles` for every k the
proposal swept, `min(grid, ntiles)` collapses to `ntiles`, and the trip count is exactly
one. **At the derived tile, `one_wave` is precisely the condition "there are no more tiles
than there are SMs" and persistence is precisely the technique for "there are more tiles
than there are SMs".**

The escape — which F-03 did not take and which §3 measures — is to *narrow* the tile below
the derived one, raising the tile count above the SM count. That works on configs 4 and 12
and on nothing else in the matrix:

```
tokens        BM=16          BM=32          BM=64        persistent arm exists?
   128     8 tiles        4 tiles        2 tiles        NO -- at any legal tile
   512    32              16              8             NO -- at any legal tile
  2048   128 tiles       64              32             yes, at BM=16 ONLY
```

`MIN_TILE_ROWS` is 16 because sm_89's `mma.sync` is `m16n8k16` and `tl.dot` needs every
dimension >= 16, so 16 rows is a hardware floor rather than a tuning choice. **On configs 2
and 3 the mechanism is unreachable at every legal tile, and the symmetric sweep found zero
persistent arms at 128 tokens to confirm it.**

Enumerated over the announced matrix (`fits`, then `amortizes`, then `one_wave`, in v34's
own order):

```
cfg  tokens     BM  fits  amortizes  one_wave  per_sm  ntiles  DISPATCHES
  2     128     16  True      False      True       1       8      YES     <- 1 trip
  3     512     16  True      False      True       1      32      YES     <- 1 trip
  4    2048     32  True      False      True       1      64      YES     <- 1 trip
 12    2048     32  True      False      True       1      64      YES     <- 1 trip
  1    8192     64  True      False     False       1     128      no      (would loop)
  5   16384     64  True      False     False       1     256      no      (would loop)
  9,10,11 as config 1;  6, 13 taken by amortizes;  7 taken by amortizes (per_sm = 8);
  8, 14 declined by fits (d_model 1024 needs 4.19 MB of weight image)
```

`tests/bench/test_persistent_ffn_closure.py` asserts all three boundaries — the derived
tile, the two configs with no persistent tile at all, and the two that have exactly one —
over `ffn_fused`'s own public functions with no GPU, so the closure cannot rot silently
[L40] and a future card, width or tile rule that moves any boundary fails the matching
test by name.

## 2. AND 4.5 POINTS OF THE REMAINING 8.2% WERE THE SIZE OF THE SEARCH

F-03 times the **frontier** once (min of 5 `do_bench`) and the **persistent** form as the
minimum over **96 arms** of min-of-3. Under unlockable clocks the minimum of 96 noisy draws
sits below a single draw for free. `bench/probes/g39_persistent_ffn/probe_one_trip.py`
applies the candidate-side protocol to the frontier kernel — 96 repeats of the identical
call, min of 3 each, take the min — and asks what the frontier appears to gain over itself:

```
 tokens   single min-of-5   min of 96 min-of-3   frontier's "speedup" over ITSELF
    128       7.716 us            7.608 us              0.9859x
   2048      14.782             14.118                  0.9551x     <- 4.5%
   8192      36.790             36.710                  0.9978x
  16384      63.058             62.078                  0.9845x
```

**At 2048 tokens — configs 4 and 12, the rows the proposal is actually for — searching 96
arms buys 4.5% with no kernel change at all.**

## 3. WHAT IS LEFT WHEN BOTH ARE REMOVED

Same probe, geometry-matched arms (same BM, same warps, same program count), ABBA-
interleaved, cold round discarded, min of five kept rounds, outputs checked at the locked
tolerance, `n_spills = 0` on every arm:

```
 tokens   frontier    persistent    ratio     F-03 claimed
    128    7.657 us     7.872 us   1.0281x       0.977x     <- SLOWER
   2048   14.167       13.817      0.9753x       0.918x
   8192   36.840       35.041      0.9512x       0.919x
  16384   62.393       57.727      0.9252x       0.902x
```

`0.9753 x 0.955 = 0.931` against the claimed `0.918` at 2048 tokens, so the one-trip
comparison and the selection bias together account for essentially the whole reported gap.

## 3. THE PROPOSAL MEASURED FAIRLY, WITH THE MECHANISM ACTUALLY ENGAGED

`probe_symmetric_sweep.py` sweeps `BM x warps x stages` for the frontier and
`BM x warps x stages x grid` for the persistent form, at equal arm counts, and — this is
the part F-03 was missing — **separates the persistent arms by trip count.** It also
reports the frontier *as shipped*, because `fused_ffn_normed` never passes `num_stages`
and takes Triton's default, which was a second confound in F-03's winning arms.

```
                                            128 tok    2048 tok    8192 tok   16384 tok
frontier AS SHIPPED (derived BM, 8w)         7.791      14.354      35.936     61.170
frontier best of 32 arms                     7.677      14.185      34.832     61.495
persistent best, ONE trip                    7.734      13.848      32.545     57.635
persistent best, >1 trip (MECHANISM ON)     (0 arms)    13.732      32.989     56.736
                                                        BM=16       BM=64      BM=32
                                                        1.94 trips  1.94       7.76
mechanism ratio vs frontier-best              --        0.9681x     0.9471x    0.9226x
```

Three things fall out:

* **At 128 tokens there are zero persistent arms**, exactly as §1 predicts. Config 2 gets
  nothing from this proposal, ever.
* **`num_stages` is a real but small confound.** The frontier's own best arm is
  `stages=1` at 2048 and 8192 tokens, worth 1.2% and 3.1% over what it ships — available
  from one keyword on the existing launcher, with no persistent rewrite at all.
* **With the mechanism genuinely engaged the persistent form wins 3.2%** over the best
  frontier arm at 2048 tokens (4.3% over the frontier as shipped). Real, reproducible,
  and less than half of what was claimed.

## 4. L33 — THE DILUTED FIGURE, IN UNITS OF `weighted_score`

Priced on the **mechanism-engaged** arm against the frontier as shipped
(`13.732 / 14.354 = 0.9567`), against each config's own wall:

| cfg | block share | wall (v34) | ratio | saved | speedup | Δ weighted |
|---|---|---|---|---|---|---|
| 4 | 41.18 µs of 99.6 (41.3%) | 0.0973 ms | 0.9567 | 1.78 µs (1.83%) | 2.382 → 2.427 | **+0.0032** |
| 12 | same decomposition | 0.085 | 0.9567 | 1.52 µs (1.79%) | 2.421 → 2.465 | **+0.0032** |
| 2 | 15.19 of 43.6 (34.8%) | 0.0502 | — | **no persistent arm exists** | — | 0 |
| 3 | past the 3.0 cap | 0.059 | — | no persistent arm exists | — | 0 |

**Total: +0.0063 of `weighted_score`, on two configs, against a claimed +0.0138 on three.**

And two reasons to treat even that as an upper bound:

1. **Every per-config delta is 1.8% against L29's ±7% floor.** No sweep, screen or graded
   run can resolve it; it would need a census, and finding 38 already records two screens
   disagreeing with each other by 95% on config 2 at a *larger* effect size.
2. **The probe is L2-cold and the model is not.** `do_bench` re-reads the weights from L2
   or HBM on each call, whereas in the real forward the same 64 KB is read by four layers
   back to back and is already resident — finding 33 measured exactly that. Weight-load
   amortization is what the persistent form sells and what L2 residency already provides,
   so the in-model figure should be *smaller* than 4.3%, not equal to it.

A kernel rewrite plus a second launcher path, carried on the frontier forever, for at most
+0.0063 that nothing in the harness can see, is the complication L17 warns the loop only
ever adds and never removes. **Declined.**

### And the instrument itself moves further than the effect

The two probes were run half an hour apart, in separate processes, both holding the GPU
lock, both min-of-3 `do_bench`. At 8192 tokens they measure a **byte-identical arm** —
BM=64, 8 warps, `num_stages=1`, grid=264, 128 programs:

```
                                probe_one_trip   probe_symmetric_sweep   apart
persistent, one trip, BM=64        35.041 us            32.545 us         7.1%
frontier as shipped                36.840               35.936            2.5%
```

**The candidate arm moved 7.1% between two runs of the same code on the same card.** That
is L29's ±7% floor, reproduced at op level, on the instrument F-03 used to report an 8.2%
win. Every ratio in this finding — mine included — should be read against it. It is the
reason the disposition below hands the one live residue to a tuner that re-measures at
prime time under a `DECISIVE` margin, rather than to a constant chosen from any of these
tables.

## 5. DISPOSITION

* **F-03 is declined.** Not "measured flat": it was priced from arms in which its own
  mechanism never ran, and when the mechanism is switched on it is worth less than half
  the claim on two thirds of the rows, entirely below the noise floor.
* **The 8192-token decline stands, and is now over-determined.** F-03 pre-registers that
  the fusion must not be extended to configs 1, 5, 9, 10 because four layers of the
  persistent block costs ~139.5 µs against ~95 µs for the split path it would replace.
  That holds; and the persistent form's own advantage there is only 5.3%, which does not
  begin to close a 45 µs gap.
* **The one live residue is `num_stages`, and it is a knob-turn, not a generation.**
  `fused_ffn_normed` never passes `num_stages`; `stages=1` is the best arm at 2048 and
  8192 tokens, worth 1.2% and 3.1% on the kernel. The principled form is to widen v34's
  existing `_pick_warps` prime-time sweep — which already times candidates under a
  `DECISIVE` margin and keeps the derived value inside it — from `warps` to
  `(warps, stages)`. That introduces no constant and no new kernel. It is worth roughly
  **+0.002** and should be judged on whether the loop wants another tuned parameter, not
  on a number no instrument can confirm. **Handed to spec 03; not built here.**

## PROPOSED LESSONS

Not appended to `docs/findings/00-learnings.md` — `ben` is ahead and it would collide.

### L55 — Best-of-N against best-of-1 is not a comparison, it is a handicap


A sweep that searches N arms on the candidate and one arm on the baseline reports the
minimum of N noisy draws against a single draw, and under unlockable clocks that gap is
free. Measured here at **4.5% for N = 96** on the shape that mattered, with the *frontier
kernel on both sides.* Any probe that sweeps must sweep both arms over the same grid, or
report the baseline's own best-of-N as a control. This is L29's noise floor arriving
through a door nobody was watching: the floor was never crossed, it was searched around.

### L56 — A knob a mechanism cannot turn is a mechanism that did not run [L36, sharpened]

L36 says assert the mechanism engaged before asserting speed. F-03 did check something —
`matches=True` on every arm — but that is a *correctness* assertion, and a kernel that does
nothing is trivially correct. The mechanism assertion here costs one line, `ntiles >
programs`, and it is the difference between an 8.2% result and a closed region. **When a
proposal's saving is proportional to some count, print the count.**

### L57 — A new technique inherits the incumbent's tuned parameters, and they were tuned against it

`one_wave` and persistence are both functions of `ntiles` against `sm_count`, written eight
generations apart, and neither docstring mentions the other. `launch_tile` derives its tile
by *spreading the work over the machine* — `ceil(tokens / sm_count)` — which is exactly the
tile that makes `ntiles == sm_count` and leaves a grid-stride loop nothing to iterate over.
F-03 took that derived tile as given, because it belongs to the kernel it was modifying,
and thereby chose the one tile at which its own mechanism is a no-op.

**When a proposal changes the execution model, the incumbent's derived parameters are part
of what is being changed, not part of the environment.** Re-derive them for the new model
before measuring, and if the new mechanism's parameter is unreachable inside the old
predicate's range, say so — that is the result.
