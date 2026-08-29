# Finding 03 — Baseline and candidate measurements

Recorded 2026-08-29. Machine: NVIDIA RTX 4070 Ti SUPER (sm_89, 66 SMs, 99 KB shared
memory/block, 48 MB L2), WSL2, PyTorch 2.8.0+cu128, CUDA 12.8, Triton 3.4.0, system
Python 3.10.12. **Clocks are not lockable under WSL** (`nvidia-smi -lgc` fails), so
timings use median-of-N with the two arms interleaved.

## Status: measurements against the announced matrix are IN PROGRESS

Everything below was measured against the **reference benchmark's own defaults**
(B=8, S=128, d_model=512, heads=8, ffn_dim=2048, **causal off**), which is **not a row of
the announced matrix**. Three structural differences make these numbers a methodology
result rather than a competition result:

- the announced matrix is **causal on every row**; this config is not;
- the announced matrix has **`ffn_dim == d_model`**, not the 4× expansion used here, so
  the feed-forward stage is over-weighted in this profile;
- **head_dim is 64 here**, while the matrix spans 8 → 256.

Treat what follows as a calibrated demonstration that the measurement path works, and as
the source of the precision findings in [02](02-allowed-techniques.md). Per-config
results against `bench/matrix.py` land in `bench/results.jsonl`.

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
