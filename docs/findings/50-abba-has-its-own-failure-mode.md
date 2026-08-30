# 50 — The ABBA protocol fails on large configs, symmetrically to how the isolated one fails on small ones

**Date:** 2026-08-31. Found while finalising the submission measurement.

## The false regression

The final replicated run reported `v38_stream_fallback` at **0.8451x on config 13** — a 15%
regression against v26 — where the immediately preceding run, same protocol, had it at
**0.9989x**. 3953.66 us against 3307.52 us: **19.5% apart**, on a 3.3 ms config where L42
measured >1 ms rows reproducing within 0.6%.

Both numbers were min-of-5-rounds at 200 warmup. That is not within-run noise.

## It is not the candidate

Measured in isolation, one arm per process, three trials, 200 warmup:

    v38  trial0  min 3287.0 us   path=resident  basis=attempt
    v38  trial1  min 3289.1 us   path=resident  basis=attempt
    v38  trial2  min 3310.6 us   path=resident  basis=attempt
    v40  trial0  min 3313.6 us   path=resident
    v40  trial1  min 3299.3 us   path=resident
    v40  trial2  min 3293.2 us   path=resident

Stable to 0.7%, and `path=resident` every time — so finding 46's residency fix is working
and the streaming path was never taken. **The 3953 us came from the measurement, not the
model.**

## The mechanism, and why it is the mirror image of finding 45

`bench/abba.py` holds **every arm resident in one process** — that is the whole point, and
it is what makes it correct on sub-millisecond configs where the isolated protocol drifts
and misreports construction-time planners by 2-4x.

But config 13 is `B=64, S=1024`: 65536 tokens, and each arm carries its own CUDA graph and
static buffers. Three or four of those resident at once is exactly **finding 05's
co-residency hazard**, which inflated a config-6 baseline 4.1x through host-memory spill.
The protocol built to avoid one distortion walks into the other.

So the two protocols fail on disjoint config sets, for opposite reasons:

    ISOLATED     wrong on SMALL configs   drift between arms; construction-time work
                                          inside the window (findings 42, 45)
    ABBA         wrong on LARGE configs   co-residency pressure across arms (finding 05)

Neither is universally right, and this project has now been burned by both.

## What to use

  * **Sub-millisecond configs (1, 2, 3, 4, 7, 9, 10, 11, 12):** ABBA, all arms resident.
  * **Large configs (5, 6, 8, 13):** one arm per process, replicated, compared across runs.
  * **Either way, replicate**, and include a byte-identical control arm whose reading you
    already know.

The corrected reading of the final run is therefore that v38 and v40 are **flat on config
13** (3287-3313 us against v26's 3325), not regressed.

## L57 — A protocol built to avoid one distortion will find the other one

Every measurement protocol in this project was introduced to fix a specific, real defect in
its predecessor, and each introduced a new one on a disjoint part of the matrix:

    min(median,median) isolated  ->  fixed co-residency spill, broke small-config ranking
    ABBA all-resident            ->  fixed small-config ranking, broke large-config spill

That is not a sequence of mistakes; it is what happens when a single number is asked to
cover a 5000x range of problem sizes on a card with 16 GB and no clock lock. **The
resolution is not a better protocol but a per-regime one, chosen by measured shape — the
same dispatch discipline the candidates themselves are held to (rule 2).**

The tell, both times, was a byte-identical or known-stable arm reading differently between
runs. That is why every comparison in this project now carries a control it does not need.
