# 29 — v19 is flat, and both probes that said otherwise failed by documented hazards

**Date:** 2026-08-30. **Candidate:** `v19_norm_fused` (gen 19, parent v18).
**Idea:** proposal D-01. **Verdict:** not a frontier advance; kept as a measured node.

## Result

    cfg  6   64.944 -> 65.172 ms   +0.4%   FUSED    flat
    cfg 13    3.293 ->  3.030 ms   -8.0%   FUSED    real win
    cfg  7    0.115 ->  0.112 ms   -2.7%   FUSED
    cfg  9    0.254 ->  0.239 ms   -6.0%   fallback (noise: the code path is identical)

    geomean vs compiled   2.765x -> 2.788x   (+0.8%, inside the noise floor)
    total wall time       76.9 ms -> 76.9 ms  (identical)

Config 13's -8.0% is outside the noise floor and is a genuine win. Config 6 is 84% of
wall time and is flat, so the aggregate does not move. Correctness was clean everywhere:
0 failed elements at the locked tolerance on every config, and `max_abs` on config 6
actually IMPROVED (1.87e-3 -> 1.56e-3), as predicted — the residual never round-trips HBM.

## Three measurements of config 6, two of them wrong

    op-level probe        "3.84x faster on the replaced segment"
    model-level probe     "16.2% SLOWER"       (113.3 ms vs 97.5 ms)
    harness sweep         "+0.4%"              (65.2 ms vs 64.9 ms)   <- authoritative

**Both wrong numbers came from hazards already written down in this repository, by me,
within the last two days.**

### The op-level probe compared against a strawman (L33, sharpened)

My probe's "current" baseline was `F.layer_norm(...)` then `fused_ffn(...)` called
separately in eager. The real candidate does not do that: **Inductor fuses the residual add
and the LayerNorm into a single kernel**, and that fused kernel is what v19 must beat. I
measured against a decomposition the real system never executes.

This is a sharper form of L33 than v15's. There the isolated win was real and got diluted
by context. Here the isolated win was **never available**, because the baseline I compared
against does not exist at run time. Isolation does not merely shrink an effect; it can
invent one.

### The model-level probe recreated finding 05

To time the real candidate I built the baseline model and the candidate in one process and
held both. That is precisely finding 05 — a co-resident model inflated config 6's baseline
4.1x by forcing a host-memory spill — and it is why `run_matrix` times the arms in
isolation and why `bench/gpu_lock.py` exists at all. I wrote the lock hours earlier, then
wrote a probe that walks straight into the hazard the lock was built for, because the lock
guards *processes* and I put both models in *one*.

The +16.2% was memory pressure, not the kernel.

## L41 — A probe is a measurement instrument and needs the same protocol as the harness

`bench/run_matrix.py` embodies six rules learned the hard way: time the arms in isolation,
one config per subprocess, correctness before timing, min-of-N under unlockable clocks,
refuse a dirty tree, refuse a contended GPU. **Every ad-hoc probe silently opts out of all
six**, and this project has now produced three wrong numbers that way — finding 09's
scripts, v15's contended sweep, and both of v19's probes.

The rule going forward: **a probe may propose, it may never conclude.** A number that will
change a decision has to come through the harness. If a probe and the harness disagree, the
probe is wrong until proven otherwise — that is the correct prior, and it was correct all
three times.

Related: L9 said "ad-hoc scripts reintroduce the errors the harness prevents", and was
recorded on day one. This is its third recurrence. The difference now is that the failure
was fast and cheap: the harness contradicted the probe within four minutes, and the finding
cost nothing but the time to write it down.

## Disposition

v19 stays on `cand/g19/norm-fused` as a measured node, not merged. It is not worse
(+0.8% geomean, identical wall time) and it wins on config 13, but it adds a second kernel
and a second store for no aggregate gain, and the frontier should not carry complexity that
does not pay.

**What it does establish:** the pointwise/normalization bucket is NOT the opportunity the
profile suggested. Those nine kernels at 661-672 GB/s are at the roofline, but Inductor was
already fusing them well enough that deleting them changes nothing. The remaining time on
config 6 is attention, and that is where the next kernel should go.
