# GB10 submission — shape-aware kernel dispatch

This is the germane result, built only from what the TikTok TechJam 2026 problem statement
rewards. Everything measured against the **announced matrix** (all causal, `ffn_dim ==
d_model`), correctness under the statement's bound (**rel < 0.02 OR abs < 0.002**, fp32
accumulate), speedup over the **`torch.compile`** baseline.

## What the submission is

A `select(shape) -> recipe` dispatcher (`ratchet/kernels/dispatch.py`) that picks a
per-shape implementation, driving this repo's own kernels — the statement is explicit that
this is a dispatch problem ("different implementations for different shapes by adding shape
checks").

- **Kernels** (all hand-written, fp16 compute, fp32 accumulate):
  `flash_attention` (causal, streams the KV axis, head_dim 8–256 — pads <16 to 16 for the
  tensor cores, tiles 256-wide heads to fit smem) and `linear_tf32` (fp16 operands → fp16
  tensor-core GEMM) for the fused QKV, output projection and FFN.
- **Dispatch is calibrated from device properties**, not hardcoded thresholds: flash tiles
  from the smem budget, and the CUDA-graph decision from the measured launch overhead vs a
  FLOP estimate (`launch_frac`), gated by a static-buffer memory check. The same shape is
  launch-bound on one GPU and compute-bound on another, so the decision reads
  `ledger/device.gb10.json`.
- **CUDA graph** collapses the many-kernel forward into one replay on launch-bound shapes;
  dispatch turns it off where it is neutral or its buffers are too large.

## Results — announced matrix, GB10, vs `torch.compile` (13 runnable configs)

All correct vs the fp32 baseline.

| cfg | shape | graph | vs compile | vs eager |
| --- | --- | --- | --- | --- |
| 1 | B64 d128 H4 S128 | on | 2.10x | 2.89x |
| 2 | B1 d128 H4 S128 (launch-bound) | on | 2.38x | 3.10x |
| 3 | B4 | on | 1.73x | 2.28x |
| 4 | B16 | on | 1.95x | 2.49x |
| 5 | B128 | on | 2.07x | 2.95x |
| 6 | B10000 (occupancy) | off | 1.81x | 2.68x |
| 7 | d32 hd8 | on | 1.68x | 3.36x |
| 8 | d1024 hd256 | off | 3.75x | 3.69x |
| 9 | H1 hd128 | on | 2.02x | 2.04x |
| 10 | H2 hd64 | on | 2.29x | 2.29x |
| 11 | H16 hd8 | on | 5.22x | 5.18x |
| 12 | S32 (short seq) | on | 2.32x | 2.29x |
| 13 | S1024 (attention-heavy) | off | **9.93x** | 10.34x |

**geomean vs EAGER (the official evaluator baseline; `--compile-baseline` is off) = 3.24x**,
and **vs `torch.compile` (a harder, self-imposed baseline) = 2.58–2.68x** across runs (the
spread is torch.compile baseline variance + GB10's unlockable clock). Every config correct.

The win tracks attention cost: dense configs 1.7–2.4x, head_dim-8 with 16 heads 5.2x,
seq=1024 **~9.8x** (flash streams the causal KV axis where the baseline materializes the
score matrix). Dispatch's graph on/off is validated — the graph-off compute-bound configs
(5/6/8/13) hold or improve their speed while avoiding large static capture buffers.

### It is a kernel win, not a dtype trick (`tests/manual/decompose.py`)

The attention kernel measured against the baseline's explicit attention at **matched fp32
precision** (no dtype advantage, no graph, no compiler):

    cfg13 seq=1024   7.17x   |  cfg11 16-head  4.86x  |  cfg5 B128  4.03x  |  cfg1  3.59x

Most configs are a genuine 2.2–7.2x pure-kernel win from the streaming online-softmax +
exact causal-skip; fp16 is an additional tensor-core factor on top. Two configs (cfg8
head_dim 256, cfg10 head_dim 64) are flat at fp32 and win via fp16 — stated plainly.

## Why each piece is germane (maps to the judging)

| Piece | Judging axis |
| --- | --- |
| flash + fp16 GEMM kernels, causal, head_dim 8–256 | Technical Execution (the kernels) |
| fp16 + fp32-accumulate inside rel<0.02/abs<0.002 | Correctness clause |
| shape-aware `select`, device-calibrated | Innovation / Impact — the statement's dispatch problem |
| honest matrix-vs-`torch.compile` measurement, one-run-at-a-time, correctness-gated | "Methodology is the scored artifact" |

## The search loop — the dispatch table is measured, not asserted

`tests/manual/search_loop.py` is a single-process port of the reference branch's parametric
loop: **propose → gate(correctness) → measure(vs torch.compile) → record → select**, over
the knobs the dispatch exposes (`use_graph` × compute `dtype`). It does not invent kernels;
it tunes knobs, which a classical search does better than an LLM.

Properties that make it a loop, not a sweep:
- **Failures are data.** bf16 fails the correctness gate on every config and is written to
  the ledger as a failing row, not skipped — that is how the loop *knows* fp16 is required.
- **Promotion needs a margin** (3% noise floor), so it cannot report its own run-to-run
  spread as progress.
- **Append-only ledger** (`ledger/bench_results.jsonl`) with git provenance (sha, branch,
  dirty) — one row per (config, point).

Screen over a representative subset, and its measured table **agrees with the calibrated
dispatch** — including the non-obvious call that config 13 (seq=1024) is *faster with the
CUDA graph off*:

    cfg2  launch-bound   -> fp16 + graph      2.26x   (graph 2.26x > no-graph 1.62x)
    cfg11 hd8/16-head    -> fp16 + graph      3.10x
    cfg13 attention      -> fp16 + NO graph   5.54x   (no-graph 5.54x > graph 5.37x)

So the dispatch decisions are not hand-asserted heuristics — they are recoverable from a
measured, auditable ledger.

## Honest boundaries

- Config 14 (seq=100000) is excluded because the fp32 *reference* OOMs (same as the
  reference branch); the flash kernel itself streams it.
- LayerNorm still runs in torch (captured in the graph, ~zero launch cost). A Triton
  LayerNorm kernel exists (`layernorm.py`) to make the layer 100% custom kernels if the
  rules require it.
- The earlier tf32/tf32x2/fused-FFN exploration (docs `03-results.md`, E2–E10) is **not**
  part of this submission: it was tuned to non-causal, 4x-FFN shapes vs an eager baseline —
  the wrong problem. fp16 on the real matrix supersedes it.
