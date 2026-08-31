# TikTok TechJam 2026 — GPU Kernel for a Transformer Layer (GB10 / sm_121)

This is the isolated, submittable artifact: only the parts germane to the problem
statement. It is measured on an NVIDIA GB10 (Grace Blackwell, `sm_121`), all shapes from
the announced matrix, correctness under the stated bound, speedup over the evaluator's
eager baseline (the scored metric) and over `torch.compile` (a harder target).

Runtime note: kernels need the system CUDA-13 ptxas —
`export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.

## What is submitted (file manifest)

| File | Role |
| --- | --- |
| `ratchet/kernels/flash_attention.py` | FlashAttention-2 forward (Triton): online-softmax streaming, **exact causal-skip**, fp32 accumulate, head_dim 8–256 |
| `ratchet/kernels/linear_tf32.py` | GEMM kernel (Triton); fp16 operands → fp16 tensor cores, fp32 accumulate; fused bias+GELU |
| `ratchet/kernels/layernorm.py` | LayerNorm forward (Triton) — for the 100%-custom-kernel variant |
| `ratchet/kernels/graphed.py` | The layer forward: the kernels above + fp16 stream + optional CUDA-graph capture |
| `ratchet/kernels/dispatch.py` | **Shape-aware dispatch** `select(shape, device) → recipe`, device-calibrated |
| `tests/manual/search_loop.py` | The search loop: propose→gate→measure→record→select; writes the ledger |
| `tests/manual/matrix_bench.py` | Announced-matrix harness (correctness + speedup vs torch.compile) |
| `tests/manual/decompose.py` | Per-config kernel-vs-dtype isolation |
| `tests/manual/gpu_guard.py` | **Measurement hygiene**: refuses to benchmark a GPU shared with another process |
| `ledger/bench_results.jsonl` | Append-only measured ledger (git provenance + GPU exclusivity per row) |

**Explicitly NOT part of the submission** (kept as honest research history, but off-target):
the tf32/tf32x2/fused-FFN exploration (`transformer_layer.py`, `explore.py`, `fused_ffn.py`,
`docs/hardware/gb10/03-results.md`, E2–E10). It was tuned to non-causal, 4×-FFN shapes vs an
*eager* baseline — the wrong problem. On the real matrix (causal, `ffn_dim==d_model`, vs
`torch.compile`), the fp16 path supersedes all of it.

## How it maps to the problem statement

| Clause | This submission |
| --- | --- |
| "Submit one or several GPU kernels that implement the layers" | flash attention + fp16 GEMM kernels; every heavy op is hand-written |
| "different implementations for different shapes by adding shape checks" | `dispatch.select` — the explicit dispatch layer |
| "rel error < 0.02, abs error < 0.002; fp32 accumulate; naive FP8 out" | fp16 compute, **fp32 accumulate**, every config passes the gate |
| every announced config is causal | flash kernel skips the causal upper triangle — exact, not approximate |
| "optimize & test on your own machine" (no fixed hardware) | measured on GB10 vs `torch.compile`; methodology + ledger are the portable artifact |
| judging: Innovation/Impact/Feasibility (55%) | shape-aware dispatch + search loop + auditable ledger — a system, not just fast kernels |

## Correctness

Every reported config passes `abs ≤ 2e-3 OR rel ≤ 2e-2` elementwise vs the fp32 baseline,
with fp32 accumulation in every kernel. bf16 was tried and **fails** the gate (recorded in
the ledger) — hence fp16.

## Results

One clean run of the announced matrix on GB10, **all 13 configs correct** (fp16 compute,
fp32 accumulate). Every measurement below was taken with the GPU verified exclusive to the
measuring process (`tests/manual/gpu_guard.py`); see *Measurement hygiene*.

Two baselines are reported because they answer different questions. **`vs eager` is the
scored metric** — the evaluator's default baseline is eager (`--compile-baseline` is off,
see `docs/PROBLEM-STATEMENT.md`). `vs compile` is the harder self-imposed target.
`flash fp32` is the attention kernel against the baseline's explicit attention at **matched
fp32 precision** — no dtype advantage — reported as the median of 3 trials with the
observed half-range.

| cfg | shape [B,S,d] | heads/head_dim | graph | vs eager | vs compile | flash fp32 (pure kernel) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [64,128,128] | 4/32 | on | 2.97x | 2.14x | 3.61x ±0.1% |
| 2 | [1,128,128] | 4/32 | on | 2.77x | 2.13x | 2.36x ±0.2% |
| 3 | [4,128,128] | 4/32 | on | 2.88x | 2.25x | 3.02x ±0.2% |
| 4 | [16,128,128] | 4/32 | on | 2.84x | 2.09x | 2.66x ±0.4% |
| 5 | [128,128,128] | 4/32 | on | 3.02x | 2.12x | 4.04x ±0.6% |
| 6 | [10000,128,128] | 4/32 | off | 2.66x | 1.77x | 4.10x ±0.5% |
| 7 | [64,128,32] | 4/8 | on | 3.35x | 1.65x | 3.54x ±0.1% |
| 8 | [64,128,1024] | 4/256 | off | 3.79x | 3.80x | 0.58x* ±0.3% |
| 9 | [64,128,128] | 1/128 | on | 2.06x | 2.06x | 1.15x ±0.0% |
| 10 | [64,128,128] | 2/64 | on | 2.30x | 2.30x | 0.82x* ±0.2% |
| 11 | [64,128,128] | 16/8 | on | 5.19x | 5.18x | 4.45x ±0.3% |
| 12 | [64,32,128] | 4/32 | on | 2.23x | 2.23x | 1.32x ±0.6% |
| 13 | [64,1024,128] | 4/32 | off | **9.80x** | 9.80x | **6.25x ±0.2%** |

| geomean over 13 configs | speedup |
| --- | --- |
| **vs eager (official / scored)** | **3.20x** |
| vs `torch.compile` (harder, self-imposed) | 2.62x |

Config 14 `[32,100000,1024]` is excluded: at `seq_len=100000` the fp32 reference would
materialize an `N×N` score matrix of roughly 1.3 TB per layer and OOMs, so no same-machine
reference exists to score against.

\* cfg8/cfg10 (head_dim 256 and 64): the flash *algorithm* alone is neutral-to-negative at
fp32, and their win comes from the fp16 tensor cores. Everywhere else the kernel wins at
matched precision (1.2–6.3x), with fp16 as an additional factor on top.

**bf16 fails the correctness gate on every config** (26 ledger rows) — the loop's own
evidence for why the submission is fp16.

### Where the win comes from

`torch.compile` and eager are **identical** on cfg8–cfg13 (e.g. cfg13: 259.84 ms vs
259.81 ms). Once attention dominates, the compiler extracts nothing from the baseline's
explicit `matmul → masked_fill → softmax → matmul`; it only helps at the small-`d`,
batch-heavy end (cfg7 2.0x, cfg6 1.5x, cfg1 1.4x). That is the whole gap between the two
geomeans, and it is also why the flash kernel's structural advantage — never materializing
the score matrix, and skipping the causal upper triangle instead of masking it — survives
against a compiled baseline.

### Measurement hygiene

Two measurement faults were found and fixed; both had already corrupted reported numbers,
so they are recorded here rather than quietly repaired.

**Contention.** The search loop is single-process and argued that this gives the same
guarantee a GPU lock would. It does — within a process. With several agent sessions open on
one box, a second session benchmarking concurrently is invisible to it. Comparing a
contended run against a clean one, the geomean barely moved (2.678x → 2.642x) while
individual configs were wrong by up to 25% (cfg3 +25.1%, cfg4 −18.0%, cfg1 −15.7%): **the
aggregate masked the contamination that the per-config table did not.** `gpu_guard.py` now
refuses to measure a shared device, and every ledger row records `exclusive`.

**Sample count.** `triton.testing.do_bench(warmup, rep)` takes *milliseconds*, not
iterations, and divides them by the estimated per-call time. At the defaults, cfg13's fp32
baseline — ~59 ms per call — received exactly **one unwarmed sample**. That is how a
non-reproducible 7.17x reading for cfg13 reached an earlier draft of this table; the
correct figure is 6.25x. Sample counts are now derived from a measured estimate and floored
at 10 warmup / 30 timed iterations, the statistic is a median rather than a mean, and each
figure carries the spread across repeated trials. Reported spreads are now ±0.0–0.8%.

### Predicted vs measured dispatch

`dispatch.select` predicts the CUDA-graph decision analytically from device properties:
`launch_frac = fixed launch time / (fixed + estimated compute)`, graph on above `0.10`. The
search loop measures both settings and keeps whichever won. Prediction and measurement
agree on 12 of 13 configs — but **which** config disagrees is not stable between runs, and
that is the honest finding:

| config | launch_frac | predicted | measured graph on | measured graph off | gap |
| --- | --- | --- | --- | --- | --- |
| cfg 5 | 0.18 | on | 5.52 ms · 2.05x | 5.66 ms · 2.00x | 2.5% (agrees) |
| cfg 1 | 0.31 | on | 3.21 ms · 1.82x | 2.95 ms · 1.98x | 8.8% (disagrees) |

On the earlier contended run the disagreement sat on cfg5 and cfg1 agreed; on the clean run
they swap. Neither config has a stable winner, because near the threshold the graph's saved
launches and the extra traffic from its static capture buffers are close to balanced. The
defensible conclusion is not "the model is wrong at cfg5" but "at `launch_frac ≈ 0.2–0.3`
the two recipes are a tie, and the ledger should be trusted over the prior only when the
margin exceeds the 3% `NOISE_FLOOR`" — which is exactly the promotion rule the loop already
applies.

## Reproduce

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# measured dispatch table + append-only ledger over the whole matrix:
python tests/manual/search_loop.py
# announced matrix vs BOTH baselines (eager = scored, and torch.compile):
python tests/manual/matrix_bench.py
# prove the win is a kernel (fp32-matched) not a dtype trick; median of N trials:
python tests/manual/decompose.py --repeats 3
```

## AI-in-the-loop (bonus)

The dispatch table is not hand-asserted: `search_loop.py` is an LLM-in-the-loop parametric
search (propose → gate on correctness → measure vs torch.compile → record to an append-only
ledger with git provenance → select on a margin). Failures (e.g. bf16) are recorded as
data, not skipped. The full development history — every candidate, including the discarded
tf32x2/fused-FFN lines and *why* they were discarded — is in the git log and
`docs/hardware/gb10/`.
