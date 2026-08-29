# Finding 03 — Baseline and candidate measurements

Recorded 2026-08-29. Machine: NVIDIA RTX 4070 Ti SUPER (sm_89, 66 SMs, 99 KB shared
memory/block, 48 MB L2), WSL2, PyTorch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0, system
Python 3.10.12. **Clocks are not lockable under WSL** (`nvidia-smi -lgc` fails), so
timings use median-of-N with the two arms interleaved.

## The announced matrix, measured

All 14 configs, causal, CUDA events, median. **Rows 1-13 all run; only #14 fails.**
Rows in `bench/results.jsonl`; run `python3 bench/ledger.py` for the scoreboard.

| # | regime | baseline ms | v1 speedup | v1 max_abs | peak MB |
|---|---|---|---|---|---|
| 1 | mainstream | 1.673 | 1.72x | 1.49e-3 | 44 |
| 2 | launch-bound | 1.759 | **8.10x** | - | - |
| 3 | launch-bound | 1.755 | 7.36x | - | - |
| 4 | launch-bound | 1.664 | 4.44x | - | - |
| 5 | mainstream | 3.327 | 1.79x | - | - |
| 6 | throughput | **459.12** | 2.03x | - | 10400 |
| 7 | awkward head_dim | 1.687 | 2.70x | **1.88e-3** | - |
| 8 | wide model | 16.709 | **1.35x** | - | - |
| 9 | mainstream | 1.501 | 1.77x | - | - |
| 10 | mainstream | 1.730 | 1.94x | - | - |
| 11 | awkward head_dim | 7.334 | 4.42x | - | - |
| 12 | launch-bound | 1.692 | 4.72x | - | - |
| 13 | long context | **111.908** | 5.94x | - | - |
| 14 | extreme | **OOM** | - | - | - |

v1 aggregate: **3.11x geomean, 2.32x total wall**, all 13 passing with 0 failed elements.
v2 (see [04](04-the-flash-attention-that-never-was.md)) raises this to **5.64x geomean**.

**#6 and #13 are 93.4% of all baseline time in the matrix.** Optimizing anything else
first is optimizing the wrong thing.

**#7 has only a 6% correctness margin** (max_abs 1.88e-3 against the 2.0e-3 budget) --
the thinnest in the matrix, and the first thing that will break if precision is pushed
further.

### Launch gap per config -- hypothesis confirmed, but not everywhere

`(wall - kernel_busy) / wall` on the baseline, 132-156 kernel launches per forward:

| #2 | #3 | #12 | #4 | #7 | #9 | #10 | #1 | #11 | #5 | #8 | #13 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 86.6% | 85.5% | 82.6% | 75.9% | 54.7% | 50.3% | 47.4% | 31.8% | 6.4% | 5.4% | 4.8% | 4.5% |

The mechanism is a **hard floor, not a gradient**: configs 2, 3, 4, 7, 9, 10 and 12 all
land in 1.50-1.79 ms despite a **64x spread in token count** (128 -> 8192). They are not
computing; they are waiting for the CPU to issue work. CUDA graph capture collapses CPU
issue cost from 1.5-3.4 ms to 0.049-0.10 ms, a 30-60x reduction.

The hypothesis is **refuted for config 1** (31.8%, genuinely 1.17 ms GPU-busy) and for
5, 8, 11 and 13, which are GPU-bound. So "everything small is launch-bound" is too
coarse -- the boundary is measurable and belongs in the dispatch predicate.

### Baseline op attribution is not what a transformer profile usually looks like

On config 13: `copy_` 27.7% + `masked_fill_` 26.6% + `softmax` 13.3% + `mul` 13.0% =
**80.6% elementwise traffic over the score matrix**, against `bmm` 14.6% and `addmm`
2.4%. The reference materializes S x S scores and then walks over them repeatedly. That
is precisely the traffic a fused attention kernel deletes.

### Config 14 -- a capability result, not a speed result

The baseline **OOMs before `forward()` is ever called**, inside the benchmark's own input
generator: the 12.21 GiB fp32 input tensor is allocated twice. Materialized attention
would need 18.63 TB for one layer.

Measured, not estimated: **fp16 flash SDPA on `[1,16,100000,64]` causal runs in 245.5 ms
at 41.7 TFLOP/s (47% of this card's ceiling) using 0.77 GB.** A full 2-layer forward on
one sequence is 550.6 ms in 2.94 GB, extrapolating to ~17.6 s for all 32 sequences.

So the model is runnable and only the **harness's own fp32 input tensor** blocks it.
Flash already streams the KV axis; chunking over the batch axis suffices. If the grader
runs this config, "the reference cannot run it and our implementation can" is worth more
than any speedup in the matrix.

**WSL2 caveat, measured rather than assumed:** the Windows driver spills past the
15.99 GiB of VRAM into host memory -- allocations up to 20.5 GiB succeeded before OOM.
Any peak above ~16 GB is a spill, not a fit, and would fail on native Linux.

## Earlier: the reference benchmark's own default config

### Why the older numbers below are kept

The numbers in this lower section were measured against the **reference benchmark's own defaults**
(B=8, S=128, d_model=512, heads=8, ffn_dim=2048, **causal off**), which is **not a row of
the announced matrix**. Three structural differences make these numbers a methodology
result rather than a competition result:

- the announced matrix is **causal on every row**; this config is not;
- the announced matrix has **`ffn_dim == d_model`**, not the 4× expansion used here, so
  the feed-forward stage is over-weighted in this profile;
- **head_dim is 64 here**, while the matrix spans 8 → 256.

They are kept because they are the source of the precision findings in
[02](02-allowed-techniques.md) and of the ablation ladder, neither of which was re-run
against the matrix.

## Where the time goes (baseline, reference default config)

End-to-end median **2.3603 ms**; GPU kernel-busy **1.5024 ms**; **launch gap 0.8579 ms**
— **36.3% of wall clock with the GPU idle**, across **178 kernel launches per forward**.

| operation | µs | % of wall | launches |
|---|---|---|---|
| `addmm` (FFN + projections) | 1014.1 | 43.0% | 36 |
| **launch gap (GPU idle)** | **857.9** | **36.3%** | — |
| `copy_` (layout shuffling) | 127.6 | 5.4% | 43 |
| `bmm` (attention core) | 83.4 | 3.5% | 12 |
| `masked_fill_` | 68.1 | 2.9% | 19 |
| `native_layer_norm` | 64.6 | 2.7% | 13 |
| `gelu` | 42.6 | 1.8% | 6 |
| `add` (residuals) | 34.7 | 1.5% | 12 |
| `_softmax` | 27.0 | 1.1% | 6 |
| `mul` + `bitwise_not` | 40.3 | 1.7% | 25 |

The matrix multiplies already run at **89–92% of this card's measured TF32 ceiling**
(44.5 TFLOP/s), so there is no headroom in the arithmetic. The headroom is in the 36%
where the GPU does nothing.

## What each change bought

| variant | median | speedup | correctness |
|---|---|---|---|
| baseline, unmodified | 2.3859 ms | 1.000× | — |
| + `F.scaled_dot_product_attention` | 1.8268 ms | 1.306× | pass |
| + CUDA graph | 1.4848 ms | 1.607× | pass |
| + `torch.compile(max-autotune)` | 1.3240 ms | 1.802× | pass |
| fused QKV + fp16 GEMM + SDPA + graph | 1.2431 ms | **1.836×** | pass |

CUDA graph capture applied to the **completely unmodified** baseline was worth **1.533×**
on its own — pure launch-overhead removal, no kernel changed.

**The uncomfortable comparison:** `torch.compile(max-autotune)`, one line of stock
PyTorch with no custom kernel, reaches 1.802× of the best 1.836×. Any hand-written kernel
must beat the compiler, not the naive baseline — and cannot address the largest lever at
all.

## Precision, measured

GEMMs cast to low precision with fp32 accumulation, same config:

| precision | max_abs | failed elements | verdict |
|---|---|---|---|
| fp16 | 0.0011 | 0 / 524,288 | **pass** |
| bf16 | 0.0096 | 21,782 / 524,288 | fail |

This is the evidence behind [02](02-allowed-techniques.md)'s conclusion that the accuracy
gate mandates fp16 over bf16.

## Candidate across shapes (reference model, not the announced matrix)

| config | baseline | candidate | speedup |
|---|---|---|---|
| B=4, S=2048, causal | 86.65 ms | 19.97 ms | 4.34× |
| B=8, S=1024, causal | 49.02 ms | 14.95 ms | 3.28× |
| B=8, S=128, 50% padded | 2.88 ms | 1.24 ms | 2.33× |
| B=8, S=512 | 12.35 ms | 6.67 ms | 1.85× |
| B=8, S=128 (default) | 2.28 ms | 1.24 ms | 1.84× |
| B=32, S=128 | 6.06 ms | 4.93 ms | 1.23× |

The spread — 1.23× to 4.34× from identical code — is the argument for shape-aware
dispatch, and a warning against quoting any single number as "the" speedup.

## Known fragilities

- **The accuracy figures sit close to the boundary.** `max_abs` of 0.0011–0.0015 passes
  `atol = 0.002` but would fail `0.001`. The reference file's own docstring states
  `atol=0.001` while its CLI default is `0.002` — worth resolving with the organizers,
  because the fp16 lever depends on which is authoritative.
- **The candidate passes under OR semantics.** The reference gate accepts an element if
  it satisfies *either* the absolute or the relative bound. This project's earlier oracle
  required *both*. `max_rel` reaches 306 on near-zero outputs, so under the stricter
  reading this candidate fails.
- **CUDA-graph capture is the most likely source of a silent wrong answer.** The harness
  hands fresh tensors to the accuracy trials and one fixed tensor to the timing loop, so
  the graph needs static input buffers with copy-in and must return a *clone* of its
  static output. The current candidate does this and passes, but it passes for reasons
  that are easy to get wrong.
