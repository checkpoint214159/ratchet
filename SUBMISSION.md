# TikTok TechJam 2026 — GPU Kernel for a Transformer Layer (GB10 / sm_121)

This is the isolated, submittable artifact: only the parts germane to the problem
statement. It is measured on an NVIDIA GB10 (Grace Blackwell, `sm_121`), all shapes from
the announced matrix, correctness under the stated bound, speedup over `torch.compile`.

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
| `ledger/bench_results.jsonl` | Append-only measured ledger (git provenance per row) |

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

Announced matrix, GB10, vs `torch.compile`, **all 13 configs correct** (fp16 compute, fp32
accumulate). `graph` is the **measured** CUDA-graph winner for that shape — the best correct
point the search loop recorded in `ledger/bench_results.jsonl`, not the analytic prediction of
`dispatch.select` (the two are reconciled below). `flash fp32` is the attention kernel measured
against the baseline's explicit attention at **matched fp32 precision** — the pure-kernel win,
no dtype advantage.

| cfg | shape [B,S,d] | heads/head_dim | graph | vs compile | flash fp32 (pure kernel) |
| --- | --- | --- | --- | --- | --- |
| 1 | [64,128,128] | 4/32 | on | 2.35x | 3.59x |
| 2 | [1,128,128] | 4/32 | on | 2.30x | 2.24x |
| 3 | [4,128,128] | 4/32 | on | 2.16x | 3.05x |
| 4 | [16,128,128] | 4/32 | on | 2.37x | 2.62x |
| 5 | [128,128,128] | 4/32 | off | 2.07x | 4.03x |
| 6 | [10000,128,128] | 4/32 | off | 1.90x | 3.96x |
| 7 | [64,128,32] | 4/8 | on | 1.64x | 3.58x |
| 8 | [64,128,1024] | 4/256 | off | 3.78x | 0.63x* |
| 9 | [64,128,128] | 1/128 | on | 2.05x | 1.32x |
| 10 | [64,128,128] | 2/64 | on | 2.25x | 0.92x* |
| 11 | [64,128,128] | 16/8 | on | 5.25x | 4.86x |
| 12 | [64,32,128] | 4/32 | on | 2.28x | 1.35x |
| 13 | [64,1024,128] | 4/32 | off | **9.81x** | **7.17x** |

**geomean vs `torch.compile` = 2.68x** · all correct · (config 14 [32,100000,1024] excluded —
fp32 reference OOMs).

\* cfg8/cfg10 (large head_dim): the flash *algorithm* alone is neutral at fp32; their win is
the fp16 tensor cores. Everywhere else the kernel wins at matched precision (2.2–7.2x),
with fp16 as an additional factor.

**bf16 fails the correctness gate on every config** (recorded in the ledger) — the loop's
own evidence for why the submission is fp16.

### Official baseline is eager (the scored metric)

The evaluator's default baseline is **eager**, not `torch.compile` (`--compile-baseline` is
off; see `docs/PROBLEM-STATEMENT.md`). The table above beats the *harder* compiled baseline;
against the official eager baseline the same clean run is stronger:

| geomean over 13 configs | speedup |
| --- | --- |
| **vs eager (official / scored)** | **3.24x** |
| vs `torch.compile` (harder, self-imposed) | 2.62x |

Per-config vs eager, all correct: cfg1 3.31 · cfg2 2.78 · cfg3 3.09 · cfg4 2.80 · cfg5 2.98 ·
cfg6 2.68 · cfg7 3.27 · cfg8 3.78 · cfg9 2.06 · cfg10 2.31 · cfg11 5.24 · cfg12 2.25 ·
cfg13 9.84 (`tests/manual/matrix_bench.py`, one run reporting both baselines).

### Predicted vs measured dispatch (12/13 agree)

`dispatch.select` predicts the CUDA-graph decision analytically from device properties:
`launch_frac = fixed launch time / (fixed + estimated compute)`, graph on above `0.10`.
The search loop then *measures* both settings and keeps whichever actually won. On 12 of 13
configs the prediction and the measurement agree. They disagree on **config 5**:

| cfg 5 [128,128,128] | launch_frac | predicted | measured graph on | measured graph off |
| --- | --- | --- | --- | --- |
| | 0.18 | graph **on** | 5.99 ms · 1.88x | **5.44 ms · 2.07x** |

The gap is 10.2% — well past the loop's 3% `NOISE_FLOOR`, so it is a real effect, not run
noise. Config 5 sits just above the `0.10` threshold, in the band where the analytic model is
weakest: it counts saved launches but not the extra traffic the graph's static capture buffers
add, and at `B=128` that traffic outweighs the ~29 launches the replay removes.

This is the search loop doing its job rather than failing at it. The analytic `select` is a
cheap prior for shapes that have never been measured; the ledger is ground truth for shapes
that have. Where they conflict, the measurement wins and the disagreement is kept on the
record — the same discipline that recorded the bf16 failures instead of quietly dropping them.

## Reproduce

```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
# measured dispatch table + append-only ledger over the whole matrix:
python tests/manual/search_loop.py
# clean speedup-vs-torch.compile table over the whole matrix:
python tests/manual/matrix_bench.py
# prove the win is a kernel (fp32-matched) not a dtype trick:
python tests/manual/decompose.py
```

## AI-in-the-loop (bonus)

The dispatch table is not hand-asserted: `search_loop.py` is an LLM-in-the-loop parametric
search (propose → gate on correctness → measure vs torch.compile → record to an append-only
ledger with git provenance → select on a margin). Failures (e.g. bf16) are recorded as
data, not skipped. The full development history — every candidate, including the discarded
tf32x2/fused-FFN lines and *why* they were discarded — is in the git log and
`docs/hardware/gb10/`.
