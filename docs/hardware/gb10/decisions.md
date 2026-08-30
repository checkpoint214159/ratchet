# GB10 decisions (ADRs)

GB10-specific architecture/decision records. Repo-wide decisions stay in
`docs/decisions.md`; only device-scoped choices belong here.

## ADR-GB10-001 — Toolchain build targeting `sm_121` (ACCEPTED 2026-08-29)

**Context.** GB10 is `sm_121` on CUDA 13.0. The `pyproject.toml` `runtime` extra pins
`torch==2.8.0` / `triton==3.4.0`, historically the `+cu128` (CUDA 12.8) wheel, which
predates `sm_121`. A measurement is only valid against the toolchain actually used.

**Decision.** Install `torch==2.9.1+cu130` (linux aarch64, cp312) from
`https://download.pytorch.org/whl/cu130`, with its bundled `triton==3.5.1`. Build the
venv on a **uv-managed** CPython 3.12 (for `Python.h`), and point Triton at the **system
CUDA 13.0 `ptxas`** via `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`. Full rationale,
gotchas, and proof output in `00-toolchain.md`.

**Consequence.** This diverges from the pinned `2.8.0 / 3.4.0`; the divergence is
deliberate and the ledger must stamp the real versions used (`torch 2.9.1+cu130`,
`triton 3.5.1`), never the pin. `TRITON_PTXAS_PATH` is a required part of the runtime
environment for every GB10 kernel run — a run without it is not reproducible. Whether to
update the `pyproject.toml` pin (or add a `runtime-cu130` extra) is deferred: it is a
repo-wide toolchain-contract change, out of scope for this device dossier.

**Status.** Accepted. Compile proof green; Triton runs on `sm_121`. Does not by itself
qualify empirical measurement — see `02-qualification.md`.

---

## ADR-GB10-002 — Tile budget from measured shared memory (OPEN)

**Context.** Tile configs are device-specific; the `sm_89` example could not fit
`128×128, d=128, 3-stage`. GB10's shared-mem budget and MMA family (Blackwell) are not
yet measured.

**Decision.** _Open._ Solve tiles from GB10's measured shared-mem/block and tensor-core
capability (`01-calibration.md`), not by copying H100/Ada configs.

**Status.** Open — depends on ADR-GB10-001 and calibration.

---

## ADR-GB10-003 — Unified memory assumptions (OPEN)

**Context.** GB10 is a Grace+Blackwell superchip with unified LPDDR5X; `nvidia-smi`
reports no dedicated VRAM. Roofline and any H2D/D2H staging assumptions from discrete
GPUs may not hold.

**Decision.** _Open._ Re-derive bandwidth/roofline from measured unified-memory bandwidth;
audit the harness for discrete-VRAM assumptions before trusting steady-state numbers.

**Status.** Open — depends on calibration.
