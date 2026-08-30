# 50 — The looped attention form was aimed at the wrong config twice, and the census is what found the right one

**Date:** 2026-08-30. **Generation:** 40. **Branch:** `cand/g40/attn-loop-census`.
**Parent:** `v38_stream_fallback` (`7cee27c`). **Candidate:** `v40_looped_attn`.
**Picks up:** the proposal finding 48 opened and declined to build.

**Numbering note:** taken as 50 on a candidate branch while `ben` is ahead. Renumber at
merge if it clashes. `00-learnings.md` and the findings README are deliberately NOT edited
here; the proposed lessons are stated at the bottom for the merge to pick up.

---

## THE CENSUS, AND THE DILUTED CEILING, STATED BEFORE A LINE OF KERNEL CODE

`bench/probes/g40_attn_loop/probe_census.py`, config 10 on `v38_stream_fallback` — the
candidate that actually ships — inside the replayed CUDA graph, 20 forwards profiled after
200 settling calls, correctness checked before anything was timed:

```
wall (median of 100, settled)              251.90 us
device time in the graph                   222.50 us   (88.3% of wall)

  _proj_gemm                          x12   69.92 us
  ampere_fp16_s1688gemm_..._relu_..   x 4   54.67        <- qkv, still cuBLAS
  _attn_single_tile                   x 4   44.14
  triton_per_fused_..._layer_norm_*   x 9   46.15
  Memcpy DtoD                         x 2    7.62

  bucket              calls   us/fwd   % device   % wall
  projection GEMM      16.0   124.59      56.0     49.5
  layernorm             9.0    46.15      20.7     18.3
  ATTENTION             4.0    44.14      19.8     17.5
  copy                  2.0     7.62       3.4      3.0
```

**Attention is 17.5% of config 10's wall**, which confirms the 17.6% finding 48 assumed
from "F-00 §1" but had never measured on this candidate. Per call: **11.04 µs**.

### The correction finding 48 did not make, and it is the reason this was worth re-doing

Finding 48 priced its proposal by multiplying that in-graph time by an op-level ratio
measured under `do_bench`. Its incumbent read **24.757 µs per call against the 11.04 µs the
model pays** — a **2.24x regime gap**. `do_bench` flushes L2 and pays a launch; the model
runs the kernel L2-hot (48 MB of L2 on this card) inside a captured graph. The looped
form's entire claimed mechanism is *latency hiding*, and latency is exactly what changes
between those two regimes. **An op-level ratio was not transferable evidence here, and
finding 48 transferred it in full** [L33].

So the ratio was re-measured in the model's own regime.
`bench/probes/g40_attn_loop/probe_regime.py` sweeps both forms over their complete legal
grid with `sdpa+repack` as a third arm, correctness-gated per arm, and times every arm
**twice — once L2-flushed, once L2-hot**. Two independent runs of the whole probe:

```
cfg 10  H=2 hd=64 S=128     incumbent    best looped     ratio (run 2 / run 1)
   flushed                  22.528 us      18.432       1.222x / 1.375x
   HOT                      11.189          9.109       1.228x / 1.226x
```

The hot ratio **replicates to 0.2%**, and the incumbent's hot time (11.189 µs)
independently reproduces the census's 11.04 µs/call. The flushed ratio moves 12% between
runs — which is its own small result: **the regime finding 48 measured in is the noisier
of the two as well as the wrong one.**

### THE DILUTED FIGURE, IN UNITS OF `weighted_score`

At the measured hot ratio, assuming it transfers into the graph in full:

| | |
|---|---|
| attention in the graph | 44.14 µs/fwd (17.5% of wall) |
| saved at 1.228x | 8.20 µs/fwd — **3.25% of config 10's wall** |
| config 10 | 2.33x → 2.408x |
| **Δ `weighted_score`** | **+0.0056** |

**Is that above the floor?** L29 puts the project's noise floor at ±7%, and finding 47
measured 7.1% between two runs of a byte-identical arm — under which 3.25% is
unresolvable, and finding 48 stopped there for exactly that reason. **But finding 49's
addendum measured the floor of the instrument this candidate is ranked with**: `bench/abba.py`
at `--warmup 200`, byte-identical arms read **52.22 µs against 52.22 µs and 6593.54 against
6593.54** — identical to the hundredth of a microsecond. Against *that* floor 3.25% is
roughly thirty times resolvable. The ±7% figure is a property of the old protocol, not of
the hardware, and treating it as a hardware constant is what closed this lever the first
time.

<!-- MEASUREMENT SECTION: filled in below from the ABBA run -->

## TWO RESULTS NOBODY ASKED FOR

### 1. Config 9 is closed, and finding 48's +0.0021 for it is WITHDRAWN

Finding 48 reported the looped form at **1.124x over SDPA on config 9** and priced it at
+0.0021. Re-measured here it is **0.955x flushed and 0.826x hot**, both replicated across
two runs. It loses, and it loses badly in the regime that counts.

The disagreement has a cause, and it is not noise. Finding 48's config-9 winner was
`BM=128`. At `B*H = 64` that is **64 CTAs on 66 SMs — one block per SM, one wave**, which
is precisely the configuration this candidate's predicate declines, because a loop with no
second wave to overlap has nothing to hide its memory latency behind. With that arm
excluded the best remaining looped tile is slower than the vendor. Finding 48 saw the same
grid arithmetic — it printed the wave table and wrote *"one wave, nothing to hide behind"* —
and then priced the arm anyway, because the arm was the fastest thing its sweep had found.

**The mechanism argument and the selected arm contradicted each other, and the sweep won.**

### 2. The incumbent single-tile kernel LOSES to the vendor on config 10

In the hot regime, `sdpa+repack` reads **9.987 µs against the incumbent's 11.189** — the
kernel this project ships on config 10 is ~12% slower than the vendor call it replaced.

This was pre-registered. `bench/kernels/attn_single_tile.py` carries an OPEN QUESTION
written into the source at generation 23:

> "The screen measured config 10 (head_dim 64) at -7.1% end to end — the marginal case,
> sitting at exactly MIN_RESIDENT_BLOCKS, one pass, inside the ±7% floor… It is
> deliberately NOT implemented until a full sweep confirms the regression is real."

This is that sweep, and **the pre-registered suspicion was right**. The fix is not to raise
`MIN_RESIDENT_BLOCKS` — that file's own comment explains why it cannot be (head_dim 32 is a
1.55x win at the same 4.9 blocks/SM). The fix is to stop assuming: `attn_choice` sweeps
`sdpa+repack` as a timed arm, so the shape now gets whichever of **three** options is
fastest instead of whichever of two.

## WHAT WAS BUILT

* `bench/kernels/attn_looped.py` — the looped kernel as a real source file, with the
  predicate expressed as `B * heads * cdiv(S, BM)` against the measured
  `multi_processor_count` (CLAUDE.md rule 2: shapes and measured device properties, never
  a config id), plus two **mechanism constraints on the sweep grid** stated with their
  pre-registered cost:
  * `block_n < block_m` — the loop must run more than one trip. This is **finding 47
    written down as a constraint**: F-03 was priced at +0.0138 on arms whose grid-stride
    loop ran exactly once, and nobody checked.
  * `num_stages >= 2` — pipelining is the whole mechanism; `num_stages=1` disables it.
* `bench/kernels/attn_choice.py` — the symmetric chooser. Both forms **and** `sdpa+repack`
  swept over their full legal grids, one timer, one repeat count, correctness gate per arm,
  spilling arms dropped before timing, and the incumbent holding the ground unless beaten
  by v23's existing `DECISIVE` 10% margin. The probe allocation is bounded by a fraction of
  measured `total_memory`, so seq_len 100000 (a 9.8 GiB probe tensor) declines the sweep
  instead of OOMing the model it is tuning.
* `bench/candidates/v40_looped_attn.py` — overrides `_decide_attn` and `_attention` only.
* **A refactor of three shipped ancestors.** The
  `if self.attn_used: single_tile_attention(...) else: <sdpa + repack>` block existed
  inline in four places — v23's `_core`, v34's `_core`, and both branches of v36's. It moved
  unchanged into `CandidateV23._attention` and the four sites became
  `self._attention(qkv, a, b, s)`. The alternative was copying three long `_core` bodies
  into the new candidate and keeping them in sync forever, which is the [L14] shape
  exactly. **The claim that the refactor is behaviour-preserving is load-bearing** — v38 is
  the ABBA control arm — so `tests/bench/test_attn_extension_point.py` asserts the method
  reproduces the inline expression on both branches for all four candidates, rather than
  assuming it.

## PROPOSED LESSONS

Not appended to `docs/findings/00-learnings.md` — `ben` is ahead and it would collide.

### L57 — An op-level ratio is evidence only in the cache regime you will spend it in

Finding 48 measured attention at 24.757 µs under `do_bench` and priced a saving against an
in-graph cost of 11.04 µs — a 2.24x regime gap, in a mechanism whose entire claim is
*latency hiding*, where the regime difference IS whether the latency exists. The ratio
happened to survive re-measurement here (1.200x flushed → 1.228x hot), and that is luck,
not method: on config 9 the two regimes disagree by 16% and change the verdict's sign
relative to what was published.

`do_bench` flushes L2 and pays a launch; `do_bench_cudagraph` does neither. CLAUDE.md
already forbids comparing one to the other. **The unstated corollary is that a ratio
computed inside one of them may not be multiplied by a time measured inside the other** —
which is what pricing a census with an op-level sweep does every time. Time both arms in
the regime the model runs, or say in the write-up that you did not.

### L58 — When the mechanism argument and the swept winner disagree, the sweep is the thing that is wrong

Finding 48 printed a wave table saying config 9 had "one wave, nothing to hide behind" and
then priced config 9 at +0.0021 on a `BM=128` arm — the arm that *creates* the one-wave
grid. The sweep found it because a sweep will always return its fastest row; the mechanism
said it could not be right; the sweep was believed.

A sweep has no theory and cannot decline. That is what makes it useful for picking a tile
and useless for deciding whether a lever exists. **Where the two disagree, the sweep is
reporting a measurement of something — usually noise, here a 2.1x-unstable baseline — and
the mechanism is the only thing in the room that can say which.** The cost of getting this
backwards was a published +0.0021 that measures −17% when re-run.
