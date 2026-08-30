# 33 — The weights were already in L2. A-06 is closed, and the API works.

**Date:** 2026-08-30. **Proposal:** A-06, "pin the weight matrices in L2 with a
persisting-access window". **Parent it would have had:** `v26_causal_correct`.
**Branch:** `cand/g30/l2-persistence`. **Candidate produced: none.**
**Probes:** `bench/probes/l2_persistence/`.

## The claim, and the number that kills it

A-06 argued that config 6's activation stream — 327 MB fp16 per tensor against a 48 MiB
L2 — evicts the fp16 weight cache under LRU, so every CTA-row re-fetches its weight tile
from HBM. It priced the worst case at **~10 GB of weight re-reads ≈ 16 ms**, up to 23% of
config 6, and said explicitly that this was a *hypothesis, not an observation*.

It is not an observation, and it is false. Measured on the frontier's own FFN megakernel
at config 6's shape:

| arm | ms | achieved GB/s counting **activations only** |
|---|---|---|
| activation-only floor (same three streams, no arithmetic) | **2.647** | 619.0 |
| the real kernel — one 64 KiB weight pair, all 20000 programs | **2.702** | 606.3 |
| 1 MiB weight arena (16 copies) | 2.706 | 605.4 |
| **32 MiB weight arena (512 copies) — forced to miss** | **4.728** | 346.5 |
| the real kernel **+ a persisting window over the weights** | 2.709 | 604.8 |
| 32 MiB arena **+ a persisting window over the arena** | **2.707** | 605.2 |

The real kernel runs **2.1% above a pure-streaming floor measured on exactly its own
activation traffic**. That 2.1% is the entire budget for weight DRAM traffic, both
tensor-core GEMMs and the `erf`. Weight traffic is somewhere inside it, and it cannot be
more than 34 MB against 1.638 GB of activations.

Installing the window on the real access pattern moves it **−0.25%**. Nothing.

## Why the contrast is the measurement, and not the roofline

`ncu` is unavailable here: `ERR_NVGPUCTRPERM`, because WSL2 denies GPU performance
counters and the fix is a modprobe option on the *host* driver. So `dram__bytes_read.sum`,
which A-06 correctly named as the cheapest falsifier, cannot be read on this box.

The substitute does not need it. `_ffn_block_off` is the frontier's `_ffn_block` plus one
per-program offset into a weight arena — identical instruction stream, identical
arithmetic, identical activation traffic. The **only** difference between arms is how many
distinct weight copies the grid touches, so the delta between two arms *is* weight DRAM
traffic, with no roofline assumption anywhere in the argument:

    weight bytes per program : 2 * D * F * 2 = 64 KiB at D = F = 128
    programs                 : ceil(1_280_000 / 64) = 20000
    full-miss weight traffic : 20000 * 64 KiB = 1.311 GB = 2.136 ms at 613.7 GB/s

    measured cold - hot      = 4.728 - 2.702 = 2.026 ms

**The real kernel is already saving 94.8% of the theoretical worst case.** If it were
missing L2 the way A-06 supposed, there would have been nothing left for the 512-copy arm
to lose.

## Both positive controls fire — [L38]

A null from an instrument that cannot detect anything is not evidence, so neither null
was accepted until the corresponding control was shown to fire:

1. **Can the contrast see weight DRAM traffic at all?** Yes: 32 MiB of weights costs
   **+75%** (2.702 → 4.728 ms). The instrument works.
2. **Does the persisting-access window do anything on this card?** Yes, dramatically:
   the same 32 MiB arm goes **4.728 → 2.707 ms, +42.7%**, recovering essentially the
   whole penalty and landing back on the resident-weights time. The ctypes shim is
   correct, the driver accepts the window, and sm_89 honours it.

So the two facts stand together and neither is an artifact of a broken tool: **the
feature works, and there is nothing here for it to do.**

## The derivation that should have preceded the proposal — [L37]

A-06's ceiling assumed a 0% weight hit rate. The reuse distance says that cannot happen.
A weight line is re-referenced once per CTA, and one CTA touches `BM * (2D + 4D + 4D)` =
80 KB of activation at `BM = 64, D = 128`. Over one full sweep of the 48 MiB L2 the weight
tile is therefore referenced

    48 MiB / 80 KB ≈ 600 times

A line touched 600 times per cache-sweep is never the LRU victim. The general condition,
in the [L37] form that belongs in the proposal rather than in the post-mortem:

    weights survive L2 when   weight_arena_bytes * (reuse distance in CTAs)  <<  L2_bytes

The probe locates that crossover directly: resident at a 1 MiB arena (2.4 MB reuse
distance), evicted at a 32 MiB arena (75 MB reuse distance) — i.e. the boundary sits where
the reuse distance crosses the cache size, as it must. **At `d_model = 128` the whole
four-layer weight set is 768 KiB, 1.6% of L2**, on eleven of the fourteen announced rows.

## Config 8 is the one row the arithmetic does NOT close, so it was measured

Writing that condition as a test immediately failed on config 8, which is the finding this
note would not have contained if the claim had been left in prose ([L40]):

    config 8: d_model = ffn_dim = 1024, 4 layers
    per layer   (4 D^2 + 2 D F) * 2 B  =  12.00 MiB
    all layers                         =  48.00 MiB   ==  this card's L2, to the byte

Layer 1's weights genuinely can be evicted by layers 2–4 plus the activation stream. That
is exactly A-06's scenario, and unlike config 6 it is not disposed of by size. So it was
run: the four weight-bearing GEMMs of each layer over one contiguous 48 MiB arena, with and
without a window over the whole arena (set-aside 33 MiB, this card's maximum, `hitRatio`
33/48). Norms and attention are omitted, which *removes* activation traffic and therefore
biases toward finding the weights resident — the bias runs against the null, which is the
direction an honest null needs.

    no persistence      4.834 ms
    persisting window   4.864 ms     -0.62%

Null again, and the per-GEMM roofline says why config 8 was never going to care:

| GEMM | measured | activation floor | vs floor | achieved |
|---|---|---|---|---|
| qkv `[8192,1024]x[1024,3072]` | 593.0 µs | 109.4 µs | 542% | **86.9 TFLOP/s = 98.5% of measured peak** |
| out-proj `[8192,1024]x[1024,1024]` | 209.5 µs | 54.7 µs | 383% | 82.0 TFLOP/s = 93% of peak |
| ffn-in `[8192,1024]x[1024,1024]` | 209.1 µs | 54.7 µs | 382% | 82.0 TFLOP/s = 93% of peak |

**Config 8 is compute-bound at 93–98.5% of this card's measured 88.2 BF16-TFLOP/s.** It
runs at 3.8–5.4× its bandwidth floor, so a memory-system policy has no time to give back
however the cache behaves. Its compulsory weight floor is 48 MiB per forward = 0.082 ms,
**1.25% of its measured 6.549 ms**, and persistence cannot touch a compulsory miss.

Config 14 (`d_model` 1024, 2 layers, 24 MiB) shares the width and is priced by the same
argument; the harness cannot build its 12.21 GiB input anyway (finding 09).

## The compulsory-miss ceiling, which closes the region rather than this config

Even granting a hypothetical eviction, **L2 persistence can only convert capacity and
conflict misses into hits. The first read of a weight byte is a compulsory miss and is
unavoidable.** So the absolute ceiling on A-06 across the whole model is the weight
traffic that is *re-fetched*, and the floor that must be paid regardless is 768 KiB at
config 6 — 0.002% of what the config moves — and 48 MiB at config 8, 1.25% of its measured
time. There is no configuration of the announced matrix in which this proposal has a prize,
not merely no configuration in which it has been shown to.

## cuBLAS is in the same position

The other big weight consumer at config 6 is the QKV projection, which the frontier runs
through `F.linear`, not through our kernel:

    F.linear [1.28M,128] x [128,384]   2.182 ms
    activation-only floor              2.136 ms      (600.8 GB/s counting activations)
    worst-case weight re-read          +1.602 ms     (128-row tiles, 96 KiB each)

**+2.2% over its activation floor against a +75% worst case.** cuBLAS's weight reads are
as resident as ours. Nothing in config 6 is paying for weight traffic.

## It does not hurt either

The one way A-06 could have been worse than inert is the set-aside: reserving L2 for
persisting accesses takes it away from everything else, and this card allows up to
33 MiB — 69% of the cache. Measured on the real access pattern:

    set-aside 0 (frontier)     2.744 ms
    set-aside 1 MiB            2.742 ms   +0.07%
    set-aside 33 MiB (max)     2.742 ms   +0.07%

Flat, and for a reason worth writing down: **config 6's activation stream has zero reuse.
Every byte is read once.** Shrinking its share of L2 costs nothing because it was never
using L2 as a cache — only as a coalescing buffer. That also disposes of the adjacent
idea of marking the activations `cudaAccessPropertyStreaming` to make room: there is no
one to give the room to.

## Why this is not [L33] in disguise

The obvious objection is that this is a mechanism measured in isolation, and [L33] says
such a measurement measures the isolation. It does — but the direction matters. The FFN
megakernel is the **most favourable** site in the model for A-06: it has the highest
weight-bytes-per-activation-byte ratio of anything at config 6, and it is the kernel A-06
would have pinned first. Isolation inflates an effect; here the inflated effect is zero,
and diluting zero by the kernel's ~30% share of config 6 does not produce a number worth
chasing. **An inflated null is still a null**, which is the one case where an isolated
probe is allowed to be decisive, and it is why no end-to-end sweep was spent on this.

Per [L41] the probe still only proposes: what it proposes is *not building the thing*,
which costs no GPU time to act on.

## Verdict

**Region closed on every announced row. No candidate.** A-06's mechanism is real, its API
works on this card, and its premise is false: at config 6 by a factor of ~20 on the traffic
accounting, and at config 8 — the one row where the size argument does not apply — because
the config is compute-bound at 93–98.5% of measured tensor-core peak. The finding it retires is the recurring
intuition — the one that also produced v3's L2-sized chunking, killed by the g10 ablation
([L17]) — that config 6's 327 MB activation stream is thrashing something we care about.
It is not. It is streaming past a cache that has room to spare, at 99% of the device's
measured bandwidth, which is what [L41] already concluded about config 6 from the other
direction: what is left there is attention, not memory-system policy.

## What survives for reuse

`bench/probes/l2_persistence/l2_persist.py` is a working, verified ctypes binding to
`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)`, `cudaStreamSetAttribute`/
`GetAttribute(AccessPolicyWindow)` and the two device attributes, with the struct layout
confirmed by read-back. Measured on this card: **L2 48 MiB, max persisting set-aside
33 MiB, max access-policy window 128 MiB.** If a future candidate ever does build a
working set that thrashes L2 — a persistent megakernel holding a K/V region, say — the
tool is written and its 42.7% control run shows what it can do when there is something to
do.
