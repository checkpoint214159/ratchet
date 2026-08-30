# 43 — cuBLAS picks an `m16n8k8` kernel at K=128, and the GELU has nowhere to hide

**Date:** 2026-08-30. **Candidate:** `v36_gemm_gelu`, branch `cand/g36/gemm-gelu`,
parent `v34_launch_bound`. **Proposal:** F-01.

**Numbering note.** This branch is cut from `cand/g34/launch-bound`, whose `docs/findings/`
stops at 33 and whose "launch floor" note is numbered 33 there and 39 on `ben`. This file
uses **`ben`'s numbering** throughout (39 = the launch floor, 42 = the harness/grader
disagreement), and is numbered 43 to sit after `ben`'s last. `00-learnings.md` and the
findings README are deliberately **not** edited here: this branch's copies stop at L44 and
row 33, so appending would conflict with `ben`'s L45–L52 at the same line. **L53 below is
stated here for the merge to pick up.**

## The census, on a config nobody had profiled

Every profile in this project's record was config 6 or config 2. Config 9 is the #1
headroom row and had never been looked at. Per-forward device time inside the replayed
graph on `v34_launch_bound`, config 9 — 220.0 µs total, 35 kernel nodes:

```
ampere_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_tn   x12   68.03 us  30.9%
ampere_fp16_s1688gemm_fp16_128x128_ldg8_relu_f2f_tn          x4    53.09 us  24.1%
pytorch_flash::flash_fwd_kernel<...>                         x4    35.72 us  16.2%
triton_per_fused__to_copy_add_native_layer_norm_*            x9    43.45 us  19.8%
triton_poi_fused_gelu_2                                      x4    12.14 us   5.5%
Memcpy DtoD                                                  x2     7.61 us   3.5%
```

**The sixteen projection GEMMs are 55.0% of device time, at 47.3 TFLOP/s — 53.6% of this
card's measured 88.2 BF16-TFLOP/s peak.** `s1688` names `mma.sync.m16n8k8`; sm_89 also
issues `m16n8k16`, twice the K-depth per instruction for fp16 operands, and that is what
`tl.dot` emits. cuBLAS's heuristic picks the k8 kernel on all twelve narrow-K calls.

It does **not** do this at d_model 1024. Config 8 gets
`cutlass_80_tensorop_f16_s16816gemm` and hits 100.4% of measured peak. The bad selection
is specific to narrow K, and that is the whole shape of the result below.

## What was measured

`bench/kernels/proj_gemm.py`: a Triton GEMM with an optional exact-`erf` GELU epilogue,
18 swept tiles, and a predicate that **times the vendor against the sweep on the real
operand shapes and keeps the vendor unless Triton wins by more than 10%**. Four call
sites decided independently. GPU lock held throughout.

Op-level ratios under `do_bench` (L2-flushed), vendor / best Triton tile, both arms timed
twice straddling the sweep — **indicative only, and see "the wrong timer" below**:

| shape | site | vendor | triton | ratio |
|---|---|---|---|---|
| `[8192,128]x[128,384]` | qkv | 20.48 µs | 19.46 | **1.05–1.18x** |
| `[8192,128]x[128,128]` | out, ffn_out | 12.29 | 10.24 | **1.20–1.33x** |
| `[8192,128]x[128,128]` +GELU | ffn_in | 16.38 | 11.26 | **1.46–1.60x** |
| `[16384,128]x[128,128]` +GELU | cfg 5 ffn_in | 25.60 | 16.38 | **1.56x** |
| `[2048,128]x[128,384]` | cfg 4/12 qkv | 10.24 | 7.17 | **1.43x** |
| `[65536,128]x[128,128]` +GELU | cfg 6/13 ffn_in | 111.62 | 41.98 | **2.66x** |
| `[8192,1024]x[1024,3072]` | cfg 8 qkv | 592.90 | 581.82 | 1.02x — **declined** |
| `[128,128]x[128,384]` | cfg 2 qkv | 5.12 | 5.12 | 1.00x — **declined** |

Every chosen tile is spill-free (`n_regs` 80–174, `n_spills` 0, read off the
`CompiledKernel` — ncu is unavailable under WSL2).

The GELU ratios are the interesting half and they are not an arithmetic win. At
`[65536,128]` the free-standing GELU reads and writes 33.6 MB, which at the measured
613.7 GB/s is ~55 µs — as much as the GEMM it follows. **Fusing it away deletes a
bandwidth-bound kernel, not a flop-bound one**, which is why the epilogue ratio (1.46x)
is so much larger than the GEMM-only ratio (1.20x).

## In the model, the GEMM speedup is 1.05x and the epilogue is the whole story

Device census, config 1, both models resident, 20 forwards each:

```
                        v34 (230.2 us / 35 nodes)      v36 (206.5 us / 31 nodes)
  projections x12    71.24 us  ampere_fp16_s1688      67.58 us  _proj_gemm
  qkv         x4     55.59     ampere_fp16_s1688      52.30     (vendor: declined)
  attention   x4     37.53     _attn_single_tile      35.18     _attn_single_tile
  GELU        x4     12.79     triton_poi_fused_gelu   ----     absorbed
  layer norms x9     45.44     triton_per_fused        44.10    triton_per_fused
```

**The hand-written GEMM beats cuBLAS by 1.05x in the model, not the 1.20–1.33x the
L2-flushed probe reported.** The proposal predicted exactly this and said to halve the
probe ratio, because in the model the weight arena is L2-resident (finding 33 measured it
at 94.8%) and the probe's flushed cache understates the vendor, whose narrow-K kernel
re-reads the same 32 KB matrix on every wave.

**Almost the entire measured win is the deleted GELU**: 12.79 µs of device time plus four
graph nodes at finding 39's 0.798 µs, against a 23.7 µs total saving. The `s1688` story is
real and it is worth 3.7 µs. The epilogue vacuum is worth 16 µs.

### The wrong timer, and the change it forced

`do_bench` zeroes an L2-sized buffer between iterations and pays a real launch per call.
The model does neither: it runs inside a replayed CUDA graph over L2-hot weights. So the
predicate was measuring a regime this call site is never in, in the direction that
flatters Triton. `plan` now times with **`do_bench_cudagraph`**, which replays a captured
graph and does not flush L2 (CLAUDE.md's own note on the two timers). The site selections
change: `qkv` at `[8192,128]x[128,384]` stops being taken on configs 1/9/10, and config 5
keeps only `ffn_in`.

## The kernel count falls by exactly four

Counted from device events, 10 forwards, v34 against v36:

```
cfg  9    35.0 -> 31.0     cfg 12    19.8 -> 19.8
cfg 10    34.9 -> 30.9     cfg  2    19.8 -> 19.8
cfg  1    34.9 -> 30.8
```

Four launches, one per layer: the GELU, absorbed. Configs 2 and 12 are on v34's
megakernel, which already owned the GELU, so nothing moves — and config 2 declines every
site outright.

## Finding 22's headline is narrower than it was stated

`v15_lifted_veto` lifted Inductor's `min_sms = 68` and so let it emit Triton GEMM
templates *with pointwise epilogues*. Finding 22 closed it on the geomean (−1.4%, later
−0.8%). The per-config rows, at two separate commits, say something the geomean hid:

```
              v9b      v15 (5cc0295a)   v15 (f7e70e9a)
  cfg  9    0.2488       0.2345 -5.8%     0.2355 -5.3%
  cfg 10    0.2529       0.2365 -6.5%     0.2365 -6.5%
  cfg 12    0.1485       0.1403 -5.5%     0.1413 -4.8%
  cfg  5    0.4383       0.4639 +5.8%     0.4669 +6.5%
```

Two commits, four configs, reproducing to 0.5%. Finding 22's null is an **aggregate null
over a matrix whose launch-bound rows had not yet been optimised**, and configs 9 and 10
are now the #1 and #3 headroom rows. The signal was in the ledger for eleven generations.

## Two measurement errors caught during construction

**The first `plan` call in a process read `F.linear([8192,128]x[128,384])` at 306.18 µs**,
where a clean process reads 21.50 µs six times running. cuBLASLt's first-use heuristic and
workspace setup landed inside the timing window. Taken at face value that is a 17.6x
"win" for Triton on a shape whose honest ratio is 1.24x — and the predicate would have
been deciding on garbage. The vendor arm is now timed **twice, straddling the sweep**, and
the min is kept. This is finding 42's lesson applied to our own tuner: when two arms are
timed in sequence, the order is part of the measurement.

**An earlier `_core` bailed to the parent whenever v23's attention kernel declined.**
Config 9 is `heads=1, head_dim=128`, exactly where v23 declines (finding 31). So the #1
headroom row ran four cuBLAS calls while `gemm_reason` reported "triton on
out+ffn_in+ffn_out", and every accuracy check passed. L36 in its purest form: the thing
that caught it was counting kernels, and that is what the test now asserts — the DROP,
not an absolute.

## The predicate declines, and it declines the right things

Evaluated on the announced matrix without ever being told a config id:

| config | sites taken |
|---|---|
| 11 | all four |
| 1, 9, 10 | out, ffn_in, ffn_out |
| 4, 12, 7 | qkv, out (a megakernel already owns the FFN) |
| 3 | qkv |
| 5 | ffn_in |
| 2 | **none** — every site ties at ~5 µs; the shape is launch-bound |
| 6, 13 | **none** — Triton loses at M ≥ 65536 on the sites left |
| 8 | **none** — the vendor is at 100.4% of peak |

Config 8 declining reproduces F-05's "config 8 is closed" from first principles, without
the predicate being able to see which config it is on.

## What it is worth, end to end

Parent against child, ABBA-interleaved, both models resident (the graded harness's own
condition), five rounds with the **cold round discarded**, min of the four remaining
medians. GPU lock held; one process per config.

| cfg | v34 | v36 | v34/v36 | sites |
|---|---|---|---|---|
| 1 | 236.5 µs | 225.3 | **1.0500** | out, ffn_in, ffn_out |
| 2 | 47.1 | 47.1 | 1.0000 | — |
| 3 | 55.3 | 52.2 | **1.0588** | qkv |
| 4 | 92.2 | 81.9 | **1.1250** | qkv, out |
| 5 | 454.7 | 433.2 | **1.0496** | ffn_in |
| 6 | 57.97 ms | 57.93 ms | 1.0006 | — *(timed solo; finding 05)* |
| 7 | 85.0 µs | 82.9 | **1.0247** | qkv, out |
| 8 | 6574.1 | 6571.0 | 1.0005 | — |
| 9 | 235.5 | 224.3 | **1.0502** | out, ffn_in, ffn_out |
| 10 | 242.7 | 231.4 | **1.0487** | out, ffn_in, ffn_out |
| 11 | 277.5 | 268.3 | **1.0344** | out, ffn_in, ffn_out |
| 12 | 87.0 | 74.8 | **1.1644** | qkv, out |
| 13 | 3294.2 | 3306.5 | 0.9963 | — |

**No regression anywhere.** Configs 8 and 13 run byte-identical code (`sites=()`), so
their 1.0005 and 0.9963 are this protocol's noise floor: **±0.4%**.

Configs 1, 9, 10 and 5 land on **+5.0%, +5.0%, +4.9%, +5.0%** — four rows agreeing to
0.15pp. That agreement is the check that matters: 1, 9 and 10 have *identical* GEMM
shapes and near-identical walls, so any protocol that gives them different answers is
measuring itself.

Applied to `v34_launch_bound`'s per-config speedups against the compiled baseline
(F-00's clean sweep), with the 3.0 cap:

```
  cfg  1   2.333 -> 2.450     cfg  9   1.816 -> 1.907
  cfg  4   2.382 -> 2.680     cfg 10   2.229 -> 2.337
  cfg  5   2.746 -> 2.882     cfg 12   2.421 -> 2.819
  cfg 2, 3, 6, 7, 8, 11, 13, 14: unchanged or already past the cap
```

**Δ weighted_score = +0.082 of 3.000** (2.489 -> 2.571). The proposal's stated realistic
band was +0.05 to +0.06 and its optimistic ceiling +0.12. This lands between them.

## The graded harness could not see this, and it is worth knowing why

`bench/end_to_end.py` runs the scored benchmark's own `main()`. Five runs per arm per
config gave:

```
  cfg  9   +9.6%      cfg  4  +12.5%       cfg  1   -2.5% and -7.5% (two runs)
  cfg 10   +3.8%      cfg 12  -35.1%       cfg  5   +5.3%
```

Configs 1, 9 and 10 again have identical GEMM shapes, and this protocol spreads them over
17 percentage points. **The parent's own arm moved 7.77x–10.32x across five identical
runs on config 1** — 33%, wider than every effect being looked for. Config 12's optimized
arm was bimodal, 9.5x–26.1x.

The cause is settling time, and it is measurable: in an interleaved run the FIRST round of
100 timed calls reads **932.9 µs for v34 and 888.8 µs for v36** on config 1, and rounds 2
and 3 read 250.9 and 238.6 µs — stable to 0.1 µs. It takes ~130 calls after the CUDA-graph
capture before the numbers mean anything, and the graded harness warms up 20. Both arms
pay it, so the *score* is not wrong; but a five-sample median of that quantity cannot rank
two candidates 5% apart on a 0.25 ms row.

This nearly produced a sixth wrong number. An early diagnostic script warmed 30 times and
read v36 at 861 µs against v34's 234 µs — **3.7x slower** — and the conclusion "the Triton
GEMM is catastrophic in-model" was one step away. What refuted it was running both arms in
one process and printing every round instead of the median: round 0 was ~900 µs for
*both*.

## L53 — A tuner that times two arms in sequence is a benchmark, and inherits every
## benchmark's bugs

Finding 42 established that `run_matrix` measured a different quantity than the grader.
This candidate then built a prime-time predicate that made the identical mistake one layer
down — arm A, a long stretch of compilation and autotuning, arm B — and a third one
underneath it, timing with a cache-flushing timer a call site that runs L2-hot inside a
graph. A dispatch predicate's error does not surface as a wrong number in a table. It
surfaces as the candidate quietly choosing the wrong kernel, forever, on a card where
nobody re-runs the sweep.

**Every timing comparison in this repository, including the ones inside candidates, has to
interleave, discard its cold round, and use a timer whose regime matches the call site's —
or say in the code why it does not.**
