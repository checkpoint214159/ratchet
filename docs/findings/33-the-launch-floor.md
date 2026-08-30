# 33 — Thirty-six kernels on every config, and what a kernel costs when it computes nothing

**Date:** 2026-08-30. **Candidate:** `v34_launch_bound` (gen 34, parent v26_causal_correct).
**Branch:** `cand/g34/launch-bound`. **Verdict:** mechanism proven; config 2 is
~25% faster on min-of-N with no overlap between the two distributions, but its MEDIAN is
now unstable because the config has crossed from GPU-bound to CPU-bound. The controller's
full sweep is the authoritative number.

## The census

Profiled the real frontier — built the way `run_matrix` builds it, warmed the way the
harness warms it, so this is the candidate and not a decomposition (L41). Kernels per
forward, counted from device events inside the replayed CUDA graph:

| config | wall | kernels/forward | device time |
|---|---|---|---|
| 2 (B=1) | 0.061 ms | **36** | 53.2 µs |
| 12 (B=64, S=32) | 0.103 ms | **36** | 95.2 µs |
| 8 (d_model 1024) | 6.549 ms | **36** | 6340.5 µs |

**Thirty-six on all three, with an identical decomposition** at L=4 layers:

```
16  GEMM         qkv, out_proj, ffn_in, ffn_out              4 per layer
 9  LayerNorm    norm1 x4, norm2 x4, final_norm              each with the residual add
                                                             and the fp16 downcast already
                                                             fused in by Inductor
 4  attention    _attn_single_tile (v23), or flash where it declines
 4  GELU         NOT fused into the ffn_in epilogue
 3  Memcpy DtoD  _static_x.copy_(x), _static_m.copy_(mask), _static_y.clone()
```

Two things in that table are worth stating on their own.

**The GELUs are unfused because of finding 22.** Inductor's 68-SM veto disables Triton
GEMM templates on this 66-SM card, so every GEMM here is cuBLAS/CUTLASS and takes no
epilogue at all. Finding 22 concluded the veto "buys nothing" because the pointwise fuser
was already collecting that work into the LayerNorm kernels. That was right about
LayerNorm and wrong about GELU: GELU has no LayerNorm to hide in and is its own kernel on
every layer of every config. It is not worth lifting the veto for — but the claim
"epilogue fusion was already happening elsewhere" was too broad.

**Nine LayerNorm kernels is already the fused count**, not the naive one. Inductor folds
the residual add and the fp16 downcast into each. There is nothing left to win there by
better fusing; only by making them not exist.

## What a kernel node costs

Calibrated on this card by capturing a graph of N identical trivial kernels and fitting
replay time against N over N = 1 … 256 (a device calibration, the same kind of thing
`oracle/device.py` does for eager launch overhead — not a candidate timing):

```
   nodes  replay us   us/node
       1      4.691     4.691
      16     12.785     0.799
      64     52.086     0.814
     256    209.009     0.816

   fit: replay(N) = 1.886 + 0.7984 * N  us
   device duration of one trivial node: 775 ns
```

**Every kernel node costs ~0.8 µs whatever it computes.** So 36 nodes is a 28.7 µs floor:
47% of config 2's entire wall, 28% of config 12's, and 0.45% of config 8's. This was
independently stated to the executor as 594 ns + 122 ns ≈ 0.71 µs / `replay(N) = 1.82 +
0.771·N`; the two agree on the slope to 3.5%, which is inside the run-to-run floor.

The lever on the launch-bound rows is not a faster kernel. It is fewer kernels.

## The trade inverts with size, and the existing predicate only knows one direction

`kernels.ffn_fused.amortizes` asks a **bandwidth** question: do enough tokens stream past
the hoisted weights to pay for hoisting them? At d_model = ffn_dim = 128 it wants ~32000
tokens, so it declines every launch-bound row. On its own terms it is correct, and
finding 29 confirmed it from the other side — v19 folded the norms into the megakernel and
measured **FLAT on config 6**, because config 6 is at 97% of the HBM roofline and fusion
there only moves bytes that were already moving at the achievable rate.

**Config 2 has almost no traffic and pays in launches instead.** The same fusion is worth
having there for the opposite reason. So there are two independent reasons to fuse, they
fire on disjoint shapes, and the codebase only had a predicate for one of them.

The new one is `one_wave`, and it is occupancy, not a fitted constant:

> When every thread block of the fused segment is resident on the device at once, the
> segment runs in ONE WAVE. Nothing in it is throughput-limited; its cost is one launch
> latency plus one block's serial chain. Splitting that same work across five launches
> multiplies the launch latency by five and buys nothing back, because there was never a
> second wave for the later launches to overlap with. Above one wave the reverse holds,
> and `amortizes` governs.

Inputs are `multi_processor_count` and `shared_memory_per_multiprocessor`, read off
`get_device_properties` at run time. No config id, no announced shape, no crossover fitted
to these fourteen rows. On the announced matrix it selects **2, 3, 4, 12**; `amortizes`
keeps **6, 7, 13**; the sets are disjoint and a test asserts it rather than assuming it.
Halving the SM count in the test drops config 12 out of it, which is the "another GPU can
evaluate this" check from L28.

## Result: 36 → 20 kernels, mechanism verified before anything was claimed (L36)

```
config 2   v26: 36 kernels, 53.2 us device
           v34: 20 kernels, 44.6 us device
```

Three deletions, in descending size:

1. **v19's norm-fused megakernel under the new predicate.** Per layer the attention
   residual add, norm2, ffn_in, GELU, ffn_out and the *next* layer's norm1 collapse into
   one kernel: five nodes become one, ×4 layers.
2. **The attention out-projection is handed over in fp16 and widened inside the kernel.**
   It is an fp16 GEMM over fp16 operands, so its result is already fp16 and the widening
   is bit-identical. This one is a lesson in itself: the first build of the candidate hit
   **24 kernels, not the 20 predicted**, because with the residual add moved inside the
   megakernel the `.float()` had no LayerNorm epilogue left to fuse into and became its
   own node per layer. **Fusion relocates work; it also strands work that was only ever
   free because it was riding along inside something else.**
3. **`_static_m.copy_(mask)` was dead.** `_nomask` implies `zero` is a Python `False` baked
   into the traced graph, so `mask` is unreachable on every path through `_core`.
   Captured as `None` instead — one memcpy node and its `cudaMemcpyAsync` gone on every
   call of every config.

`launch_tile` narrows the tile in this regime instead of widening it: there are not enough
rows to amortize a weight load, so spread blocks across SMs and let the 48 MB L2 serve the
identical weight reads. Probed at three token counts it picks the winner every time, and
v17's fixed `BLOCK_M=64` would have been ~2x worse:

```
tokens=128   best (16,8) 6.14 us   BM=64 -> 11.26 us   derived BM=16  ✓
tokens=512   best (16,8) 7.17 us   BM=64 -> 11.26 us   derived BM=16  ✓
tokens=2048  best (32,4) 11.26 us  BM=64 -> 13.31 us   derived BM=32  ✓
```

**Finding 16's Dynamo guard cost is still gone.** The CPU-side profile at config 2 shows
no `TorchDynamo Cache Lookup` row at all, against the 22.5 µs/call it cost before v12/v13.
Re-checked because a fix six generations old is exactly the kind that rots unobserved.

## The screen disagreed with itself, and that is the finding

Two screens, same commit, same clean tree:

```
             cfg 2      cfg 7     cfg 8     cfg 10    geomean vs parent
run 1      -23.3%       -3.5%     -0.1%      -0.8%          +8.1%
run 2       +1.7%       -3.5%     -0.1%      -0.8%          +0.7%
```

**Configs 7, 8 and 10 reproduce to four decimal places.** So the harness is not noisy here,
and config 2's 33% swing is a *discrete state difference*, not variance.

It is — but not the state first suspected. Repeated independent builds at config 2, each in
its own process, reporting the candidate's own introspection alongside both statistics:

```
                n    min ms, sorted (300 timed calls each)
  v26 (parent)  5    0.0604 0.0604 0.0604 0.0604 0.0614
  v34          10    0.0440 0.0450 0.0451 0.0451 0.0470
                     0.0471 0.0471 0.0471 0.0471 0.0471

                     median ms, sorted
  v26 (parent)  5    0.0604 0.0614 0.0614 0.0614 0.0676
  v34          10    0.0451 0.0451 0.0454 0.0471 0.0481
                     0.0481 0.0584 0.0612 0.0635 0.0666
```

**On min-of-N the two distributions do not overlap at all.** v34 is 0.0440–0.0471 in every
one of ten runs; v26 is 0.0604–0.0614 in every one of five. That is a **~25% improvement,
reproduced ten times out of ten** — and min-of-N is the statistic this project prescribes
for itself (`docs/00-mission.md`), precisely because the SM clocks are not lockable under
WSL2.

**On the median they overlap**, and only v34's median is unstable — it spans 0.0451–0.0666
against the parent's tight 0.0604–0.0676. `run_matrix` scores on a median, which is why one
screen read −23.3% and the other +1.7% from identical code at the same commit.

Two things are going on and it is worth separating them.

*The smaller one:* `attn_single_tile.autotune_tile` flips tiles run to run at config 2 —
v34's slow-median runs are the ones that kept the derived `(64,4,1)`. This is v23's
machinery, inherited unchanged, and v23's own docstring predicted it ("a candidate whose
own tile varies run to run adds that noise to every measurement taken of it") and set
`DECISIVE = 0.10` against it. At config 2 the tiles compared run in ~1 µs against a ~1 µs
event timer, so noise clears a 10% margin easily and the guard does not bind. But the
parent flips too (1 of 5 runs) and stays stable anyway, so this is an amplifier, not the
cause.

*The real one:* **v34 pushed config 2 across from GPU-bound to CPU-bound.** Config 2 was
already 232 µs CPU against 126 µs GPU when finding 16 profiled it. v34 removes 16 of 36
kernel nodes and 8.6 µs of device time, so the GPU side gets smaller while the CPU side —
`cudaGraphLaunch`, two memcpys, the output clone — does not. The minimum still measures the
GPU (the CPU occasionally runs ahead), but the median now samples the CPU's jitter. The
parent has enough GPU work to stay on the GPU's side of the race and is therefore stable.

That is a satisfying result and an uncomfortable one: **the optimization worked well enough
to invalidate the statistic that was measuring it.**

**Consequence, and it is not about this candidate:** config 2 is one of the four screen
configs, so it enters every screen verdict this project issues. Any candidate that makes
config 2's GPU side small enough will start reading noisy on the median, and the screen has
no way to tell that from a regression.

## What is claimed, and what is not

**Claimed, and verified:**
- 36 → 20 kernel nodes per forward at config 2; 16 × 0.798 µs = 12.8 µs of launch floor
  removed with them. Counted, not inferred.
- Device time 53.2 → 44.6 µs at config 2, reproducible from the profile.
- Config 2 min-of-N 0.0604 → ~0.0465 ms, **10 of 10 runs against 5 of 5, no overlap**. The
  12.8 µs of launch floor accounts for ~86% of the 14.9 µs saved; the rest is the fused
  kernel doing the same arithmetic in less device time.
- Configs 7, 8, 10 identical to four decimals across two screens — the predicate declines
  them and the decline costs nothing. Config 6 is untouched by construction.
- 0 failed elements at the locked 2e-3/2e-2 on every config tested, fused and fallback.

**NOT claimed:** the `+8.1%` screen geomean. That number is one draw from a median that has
become unstable for the reason above, and its twin draw said `+0.7%`. Neither is the
candidate's number.

**NOT claimed:** configs 3, 4 and 12. The predicate selects them and the same 16 nodes come
out, but the screen set does not include them and no sweep has been run.

**L33's diluted figure, stated before measuring and unchanged after:** 12.8 µs against each
wall is 21.0% on config 2, 12.4% on config 12, 11.5% on config 4, and configs 3 and 7 are
already past the 3.0 score cap. That is at most **+0.065 of 3.000** on the capped weighted
score. Configs 1, 8, 9 and 10 get exactly nothing, by construction.

## Two lessons

**L43 — Fusion strands the work that was riding along for free.** The predicted kernel
count was 20 and the first build measured 24. Moving the residual add into the megakernel
deleted the LayerNorm epilogue that the attention output's `.float()` had been living
inside, and the cast became four new kernels. Before claiming a fusion removes N kernels,
ask what was fused into the thing being removed. The count is only knowable by counting.

**L44 — When a screen disagrees with itself, the configs that AGREE tell you where to
look.** Configs 7, 8 and 10 matching to four decimals is what proved config 2's swing was
a discrete state and not noise; had everything been noisy the right conclusion would have
been "wider floor, take more samples". Reproducibility on the untouched arms is a free
control, and it converts an ambiguous result into a diagnosable one. Related to L38: the
useful check is the one capable of distinguishing the cases.
