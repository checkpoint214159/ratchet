# GB10 (NVIDIA Grace Blackwell) — device dossier

This subtree collects everything specific to bringing Ratchet up on an **NVIDIA GB10**
(Grace Blackwell, `sm_121`) box for **TechJam Task 3 — Implement a GPU Kernel for a
Transformer Layer**.

It is a **bring-up dossier**, not a second catalogue. It documents and *points at*
canonical state; it never forks it:

- Real measurements live only in the append-only `ledger/` (this dossier links rows).
- Per-device calibration lives only in `ledger/device.json` (this dossier explains it).
- The correctness oracle (Zone A, `ratchet/oracle/`) and its tolerances are untouched.
- Kernel code lives in `ratchet/kernels/`, not in a device silo.

## Files

| File | Purpose |
| --- | --- |
| [`00-toolchain.md`](00-toolchain.md) | torch/triton build that targets Blackwell `sm_121`; install + compile proof |
| [`01-calibration.md`](01-calibration.md) | Measured per-GPU record (roofline, shared-mem budget, tensor rates) |
| [`02-qualification.md`](02-qualification.md) | CUDA transformer/attention qualification checklist before any empirical event |
| [`03-results.md`](03-results.md) | Pointers to ledger evidence rows (never copies) |
| [`decisions.md`](decisions.md) | GB10-specific ADRs |

## Task 3 framing

Task 3 optimizes a **full transformer layer**, not attention alone. In Ratchet terms:

- Authoritative evaluator: `benchmarks/reference/torch_transformer_benchmark.py` (Zone A,
  SHA-pinned). Its baseline, CLI, input generation, and correctness rule are protected.
- Optimized code plugs in only through the `UserOptimizedTransformer` seam.
- Attention is the dominant sub-cost and has a stricter, separate research oracle
  (`ratchet/oracle/`); passing it does **not** by itself decide authoritative acceptance.

## Live bring-up status

| Step | State | Evidence |
| --- | --- | --- |
| Device identified | done | `nvidia-smi`: GB10, `sm_121` (compute cap 12.1), driver 580.159.03, CUDA 13.0 toolkit |
| Toolchain qualified (torch+triton on `sm_121`) | **done** | `00-toolchain.md`: torch 2.9.1+cu130, triton 3.5.1, Triton add proof green (needs `TRITON_PTXAS_PATH`) |
| GB10 calibrated (`ledger/device.gb10.json`) | **done** (peak-TFLOP gap) | `01-calibration.md`: 48 SM, 99 KB smem, 246.8 GB/s, 2418 MHz; ridge point blocked on Zone-A peak entry |
| CUDA qualification gate | **9/9 checks pass** | `02-qualification.md`: golden re-pin fixed correctness; only non-blocking peak-TFLOP gap remains; awaiting human ratification |
| Transformer-layer kernels (E2–E6) | done | hand-written flash/TF32/QKV/full-layer all correct but 0.94–0.97x vs cuBLAS fp32. `03-results.md` |
| **Speedup found (E7)** | **done, positive** | TF32 tensor cores (baseline forgoes them) + SDPA: **1.16x–1.66x**, correct, on the authoritative evaluator; dispatch selects it |
| Trustworthy timing | **done** | drift-robust interleaved ratio harness (`tests/manual/timed_compare.py`); clocks unlockable without root |

## Promotion note — canonical `ledger/device.json`

GB10 calibration is written to `ledger/device.gb10.json`, not the canonical
`ledger/device.json`, which still holds the **historical RTX 4070 Ti SUPER** record
referenced by `docs/00-mission.md`. This box's real device is GB10, so promoting the GB10
profile to `ledger/device.json` is reasonable — but it overwrites a committed historical
artifact, so it is left as an explicit, separate decision rather than done silently here.

Update this table as each step lands. Nothing below "toolchain qualified" may claim an
empirical result until the gate in `02-qualification.md` passes.
