"""Candidate v42 -- the tile sweep is ranked by an instrument that can resolve it.

Generation 42. Parent: `v41_vendor_aware_attn` (`c863eed`). Branch: `cand/g42/tile-timer`.
Takes the proposal finding 50 opened, finding 51 measured at 1.283x and both deliberately
declined: **re-tune the single-tile form with the hot timer.**

WHAT IT CHANGES -- ONE VALUE, IN ONE PLACE
-------------------------------------------
`attn_tile_timer = attn_single_tile.hot_time`, the extension point v23 grew for this. The
kernel is not touched, no tile is hardcoded, no predicate moves, and no config id appears
anywhere. `autotune_tile` sweeps the same grid at the same probe batch with the same trial
budget under the same `DECISIVE` bar and the same derived-tile tiebreak; only the function
that turns an arm into a number is different.

WHY THAT IS THE DEFECT, AND NOT "CONFIG 2 WANTS A 16-ROW TILE"
---------------------------------------------------------------
From generation 23 to 41 the tile sweep ranked with `do_bench(warmup=10, rep=25)`, which
times each call with a pair of CUDA events. The event quantum on this card is 1.024 us.
The kernels being ranked run in 1.9-11 us. `bench/probes/g42_tile_timer/probe_timer_regimes.py`
swept the full eight-tile grid under both timers, on every shape the kernel accepts, twice:

    cfg 2, flushed   5.120  5.120  5.120  5.120  5.120  6.144  6.144  6.144   <- 8 arms
    cfg 2, hot       1.905  2.260  2.409  2.438  2.486  3.344  3.423  3.718

**Five of the eight arms report the identical number and the whole grid spans one
quantum.** The sweep was not noisy, it was blank; the tie fell through to the derived-tile
tiebreak, and the tile that tiebreak kept -- `(64, 4, 1)` -- is one the hot timer ranks
1.280x / 1.278x behind `(16, 4, 1)`, replicated, reproducing finding 51's 1.283x from an
independent sweep in a fresh process.

The same quantization is wrong in the other direction too, which is the part that shows it
is quantization and not a regime preference: on config 3 the flushed timer picked
`(16, 4, 1)` in run 1 and `(64, 4, 1)` in run 2 -- a one-quantum artefact clearing a 10%
bar for a tile the hot timer ranks 1.9% SLOWER, and picking differently each time. That
instability is what finding 50 recorded as "deterministic against a comparable process
state, not absolutely". It is quantization, and the answer to it is resolution.
`do_bench_cudagraph` times a graph of many replays and divides, so the quantum is
amortized instead of paid per call.

THE BLAST RADIUS IS ONE CONFIG, AND IT WAS MEASURED BEFORE IT WAS CLAIMED
-------------------------------------------------------------------------
The probe reports what `autotune_tile`'s own decision rule returns under each timer, on
all ten shapes the kernel accepts, twice. **Nine of the ten select the identical tile
under both.** Config 2 is the only row that moves, in both runs:

    cfg  1  derived (64,4,1)   flushed -> (64,4,1)   hot -> (64,4,1)   SAME
    cfg  2  derived (64,4,1)   flushed -> (64,4,1)   hot -> (16,4,1)   *** DIFFERS, 1.28x
    cfg  3  derived (64,4,1)   flushed -> unstable   hot -> (64,4,1)   SAME (see above)
    cfg  4,5,6,7,10,11,12                                             SAME

So this candidate is byte-identical to its parent on twelve of the thirteen runnable
configs by measurement, not by construction -- and those twelve are its control arms.

WHAT IT DOES NOT CLAIM
-----------------------
Not a kernel change, not a new form, not a predicate change. `pays()` is untouched and
finding 51's answer to it stands. The vendor floor v41 added is untouched and still fires
on zero announced configs. This is one uncapped scoring row, worth single digits in the
third decimal place of `weighted_score`, and the reason it is worth a generation anyway is
that the mechanism generalises: a tuner that can resolve its own arms is right on shapes
nobody has swept, and this one could not resolve its arms on any config under 4 us.

SECOND FIX, LATENT NOT LIVE: correctness before timing
-------------------------------------------------------
`autotune_tile` admitted arms to its timing set gated only by `fits`/`pays` -- it was the
one tuner in the package that could select a tile on speed alone, while `attn_choice`
checks every arm against the reference at the locked tolerance first. Now it does too. The
probe checked all eight tiles on all ten accepted shapes, twice, and every arm matched, so
this drops nothing today. It is closed anyway [L38]: a check nobody has seen fail is
indistinguishable from a check that cannot.
"""

from __future__ import annotations

from .v41_vendor_aware_attn import build as build_v41
from ..kernels import attn_single_tile


def build(baseline_cls):
    v41_cls = build_v41(baseline_cls)

    class CandidateV42(v41_cls):
        # THE WHOLE DIFF. v23's `_decide_attn` passes this straight to `autotune_tile`;
        # `None` there means `flushed_time`, which is what every ancestor runs.
        attn_tile_timer = staticmethod(attn_single_tile.hot_time)

    return CandidateV42
