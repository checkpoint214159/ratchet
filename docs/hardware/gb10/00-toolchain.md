# GB10 toolchain

A measurement is only meaningful against a known toolchain, and the toolchain version is
stamped into every ledger row. GB10 is a **new** architecture, so this step is the first
real blocker and must be resolved before calibration or any kernel work.

## Observed environment (facts)

Captured on this box via `nvidia-smi` / `nvcc`:

```
GPU:                NVIDIA GB10 (Grace Blackwell)
Compute capability: sm_121  (nvidia-smi reports compute cap 12.1)
Driver:             580.159.03
CUDA toolkit:       13.0  (nvcc release 13.0, V13.0.88)
Memory (dedicated): N/A reported — unified LPDDR5X (Grace+Blackwell superchip)
```

Unified memory is a real difference from the historical `sm_89` discrete-GPU example in
`docs/00-mission.md`: there is no separate VRAM pool, so bandwidth/roofline and any
host↔device copy assumptions must be re-derived, not inherited.

## The problem with the pinned runtime

`pyproject.toml` `runtime` extra pins `torch==2.8.0` + `triton==3.4.0`. The historical
calibration used the `+cu128` wheel (CUDA 12.8). CUDA 12.8 predates `sm_121`; a `cu128`
torch will very likely fail to emit or run `sm_121` kernels on GB10, and Triton must be
able to target `sm_121` for any generated attention kernel to compile.

**Open decision (see `decisions.md`):** which torch/triton build actually targets
`sm_121` on CUDA 13.0. Candidates, cheapest-risk first:

1. A torch/triton wheel built for CUDA 13.0 (`cu130`) — likely a nightly at time of
   writing. Record exact version + index URL used.
2. Source build of Triton against the local CUDA 13.0 toolkit if no wheel targets
   `sm_121`.

If the qualifying build differs from the pinned `2.8.0 / 3.4.0`, that is a real toolchain
change: it must be recorded here and in `decisions.md`, and the ledger will stamp the
actual versions used — never the pinned aspiration.

## Compile proof (definition of done for this step)

This step is **done** only when all of the following are captured, on GB10:

1. `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"`
   reports a CUDA-13 build and `(12, 1)`.
2. `python -m ratchet.backends --backend cuda` reports the device available with correct
   capabilities.
3. A trivial Triton kernel (e.g. vector add) **compiles and runs** on GB10 and matches a
   torch reference — proving Triton can target `sm_121`, not just that torch imports.

## Qualifying toolchain (measured 2026-08-29)

| Item | Value |
| --- | --- |
| torch | `2.9.1+cu130` (linux aarch64, cp312), CUDA runtime `13.0` |
| triton | `3.5.1` (bundled with torch) |
| Python | uv-managed CPython `3.12.14` (python-build-standalone — bundles `Python.h`) |
| ptxas | **system CUDA 13.0** `/usr/local/cuda/bin/ptxas` (V13.0.88), via `TRITON_PTXAS_PATH` |
| numpy | installed (torch import warns without it) |

Divergence from the pinned `runtime` extra (`torch==2.8.0` / `triton==3.4.0`) is expected
and recorded in `decisions.md` ADR-GB10-001: those predate `sm_121` / CUDA 13.

### Three gotchas, each with its fix

1. **CPython dev headers.** System `/usr/bin/python3.12` has no `Python.h`; Triton
   JIT-builds a C driver shim at runtime and fails with `fatal error: Python.h`. Fix: build
   the venv on a **uv-managed** Python (`uv python install 3.12`), which bundles headers —
   no `sudo apt install python3.12-dev` needed.
2. **Bundled `ptxas` is too old.** Triton 3.5.1 ships a CUDA-12.8 `ptxas` that rejects
   `sm_121a` (`ptxas fatal: Value 'sm_121a' is not defined`). Fix: export
   `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (CUDA 13.0 supports `sm_121` and
   `sm_121a`). **This env var is required for every kernel run on GB10.**
3. **torch's own max-cap warning is cosmetic here.** torch 2.9.1 prints "Maximum cuda
   capability supported ... (12.0)" for GB10's `sm_121`; cuBLAS-backed ops and (with the
   ptxas override) Triton kernels still run. Treat as a warning, not a failure — but
   re-check any op that silently falls back before trusting a measurement.

### Proof output

```
# TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas python triton_add_proof.py
triton add max_err 0.0
compiled target: GPUTarget(backend='cuda', arch=121, warp_size=32)
TRITON sm_121 COMPILE+RUN OK

# python -m ratchet.backends --backend cuda
{"capabilities": {"availability": "available", "supported_dtypes": ["float32","bfloat16"],
 "supports_compilation": true, "supports_events": true, "supports_peak_memory": true,
 "validation": "unvalidated"},
 "identity": {"backend": "cuda", "device_name": "NVIDIA GB10",
 "framework_version": "2.9.1+cu130", "runtime_version": "13.0"}}
```

## Status

**Done.** Triton compiles and runs on `sm_121`; CUDA backend probes available. Note
`validation: unvalidated` — a positive probe is not qualification; the gate in
`02-qualification.md` (calibration, correctness matrix, timing) is still required before
any empirical event.
