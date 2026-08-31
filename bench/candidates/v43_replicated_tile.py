"""Candidate v43 -- the tile sweep is replicated and reduced by its floor before it acts.

Generation 43. Parent: `v42_hot_tuned_tile` (`5643e2c`). Branch: `cand/g43/stable-tiles`.
Builds the fix finding 53 pre-registered and deliberately did NOT bundle, so that it could
not ride into its own measurement -- and then corrects it, because the pre-registered form
was measured and is wrong. Read "THE FIX IS A FLOOR" below before anything else.

WHAT IT CHANGES -- ONE VALUE, IN ONE PLACE, AGAIN
--------------------------------------------------
`attn_tile_replicates = 2`, the second extension point v23 grew. `autotune_tile` runs its
existing sweep twice over the same probe tensor and the same reference, reduces the two
readings of each arm to their MINIMUM, and then runs v23's decision rule once on the
reduced table with nothing else changed. The kernel is untouched, the timer is v42's, no
tile is hardcoded, `DECISIVE` does not move, and no config id appears anywhere.

WHY THE PARENT NEEDED THIS
----------------------------
v42 replaced an instrument that could not resolve its arms with one that can. It bought
accuracy and it paid in variance, and finding 53 measured both halves:

    B=1, true margin 28%     flushed: the derived tile, 6/6   hot: (16,4,1), 7/7
    B=4, true margin ~2%     flushed: the derived tile, 6/6   hot: THREE tiles in 6

`DECISIVE = 0.10` was calibrated against the OLD instrument's noise distribution. The
flushed timer never had an instability problem because its tie table could not clear the
bar at all -- accidentally stable, and at B=1 stably wrong.

    "A blunt instrument is stable; sharpening it buys accuracy with variance, and you
     must check where you spent it."   -- finding 53's proposed L63

THE FIX IS A FLOOR, NOT A VOTE, AND THAT IS A CORRECTION TO THE PRE-REGISTRATION
---------------------------------------------------------------------------------
Finding 53 pre-registered: *"displace the derived tile only if both sweeps clear
`DECISIVE` and agree on the winner."* That was built first, exactly as written, and
measured. **It loses B=1 -- v42's entire win -- in 5 of 10 fresh processes**, and it
loses it deterministically: always the 5 in which this candidate primed second, after the
control arm was already built, primed and resident.

`bench/probes/g43_stable_tiles/sweep_grids.py` says why. Six back-to-back sweeps at B=4,
in the model's own regime:

    tile        sw1     sw2     sw3     sw4     sw5     sw6     floor
    (16,4,1)  2.743   4.002   2.547   4.029   2.741   2.574    2.547
    (64,4,1)  2.711   2.517   3.752   2.708   2.529   4.245    2.517   <- derived
    winner     (64)    (64)    (16)    (64)    (64)    (16)

The two sweeps that displace -- reported at 1.473x and 1.649x -- are **not a challenger
reading fast. They are the incumbent reading slow**: 3.752 and 4.245 against its own 2.517
floor, while the challenger barely moves. Contamination on this harness is ONE-SIDED (a
descheduled host thread, an unsettled graph and a co-resident allocation can only make a
reading slower), so it lands on whichever arm it lands on -- and a per-sweep `min()` then
hands the sweep to the arm that happened to be missed. A vote between two contaminated
rankings is decided by where the contamination fell, not by which tile is faster.

The floor is the estimator the mechanism demands, and the one CLAUDE.md already prescribes
for a card whose clocks will not lock: the mean is a statistic about how often the machine
misbehaved, the minimum is a statistic about the code. Reduced by the floor over adjacent
PAIRS of the sweeps above, every window agrees -- B=4 holds the derived tile (0.988x,
inside `DECISIVE`) and B=1 displaces it (1.277x / 1.284x / 1.284x), which is the margin
four independent measurements across three generations agree on.

So the pre-registration named the right defect, the right cost, and the wrong remedy. That
is recorded here rather than quietly corrected.

WHAT IS MEASURED
----------------
`bench/probes/g43_stable_tiles/prime_stability.py --mode fresh`, one process per
replicate, BOTH arms built and primed in it (which is `bench/abba.py`'s regime and
therefore the regime every ranking of these two is taken in), prime order alternated:

    cfg 3   v42: 3 distinct plans in 22 asks     v43: 1 in 22
    cfg 2   v42: 1 distinct plan  in 22 asks     v43: 1 in 22, and it is v42's tile

THE MEASUREMENT HAZARD THIS GENERATION FOUND, WHICH IS NOT THE CANDIDATE'S
---------------------------------------------------------------------------
Two probes and one test in this generation were wrong before they were right, both times
because of regime, and both are worth knowing about before measuring this tuner again:

  * **A one-arm-per-process probe sees no instability at all.** With a single arm resident,
    v42 and v43 both select one plan in 8 of 8 on both shapes. The whole effect appears
    only once a second model has been built and primed. A probe that had not modelled
    `abba.py` would have concluded "nothing to fix".

  * **`hot_time` silently degrades to `do_bench` outside `torch.inference_mode()`.** Once
    any model has been run under inference mode, `do_bench_cudagraph` raises `Inplace
    update to inference tensor outside InferenceMode`, and `hot_time`'s bare `except`
    returns the flushed number instead -- the entire grid comes back on the 1.024 us event
    tick, the exact instrument v42 removed, under v42's name. `_decide_attn` runs inside
    `inference_mode` at the real call site, so this is a probe hazard and not a shipping
    one; it is named in `hot_time`'s own docstring as a robustness note and had never been
    observed to fire.

WHAT IT DOES NOT CLAIM AND DOES NOT TOUCH
-------------------------------------------
Not a kernel change, not a timer change, not a predicate change, not a new form.
`attn_choice.autotune_looped` and `autotune_vendor` are untouched, so shapes that select
through the looped routine keep whatever instability the g41 audit recorded there. Finding
53 named those as "not this candidate's to fix" and they are not this one's either.

THE EXPECTED SCORE DELTA IS ~ZERO, AND THAT IS THE POINT
----------------------------------------------------------
B=1's win must survive unchanged; B=4 is past the 3.0 cap so its stabilisation scores
nothing in either direction. **This is a candidate whose value is in the VARIANCE and not
in the mean**, which is unusual here and is stated plainly rather than dressed up: a plan
that varies run to run adds its variance to every measurement taken of the candidate
(L29), and every measurement of every DESCENDANT. If it reads 1.000x on every config with
the plans fixed, it has done exactly what it was built to do.

COST
----
One extra pass of the grid at prime time. No extra compilation (the second sweep hits
Triton's JIT cache) and no extra allocation (the probe tensor and reference are made once
for all sweeps, deliberately, so the replicates differ only in when they ran). On the
announced shapes that is ~1 s against the frontier's existing 14-67 s of tuning, entirely
outside the timed region.
"""

from __future__ import annotations

from .v42_hot_tuned_tile import build as build_v42


def build(baseline_cls):
    v42_cls = build_v42(baseline_cls)

    class CandidateV43(v42_cls):
        # THE WHOLE DIFF. v23's `_decide_attn` passes this straight to `autotune_tile`;
        # `1` there means one sweep, which is what every ancestor runs.
        #
        # TWO, not three or five. Two is the smallest number that can have a floor at
        # all, it is what finding 53 pre-registered, and its cost is one extra sweep.
        # Measured, two is enough: the one-sided contamination hits an individual arm in
        # roughly a third of sweeps, so the chance of both readings of the SAME arm being
        # hit is small, and configs 2 and 3 each selected one plan in 22 of 22 asks. A
        # larger number would buy a tighter floor with more prime time and nothing
        # measured so far asks for it.
        attn_tile_replicates = 2

    return CandidateV43
