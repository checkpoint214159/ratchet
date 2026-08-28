# ratchet

Agentic GPU kernel optimization harness. Built for the TikTok TechJam 2026 GPU kernel
track and intended to outlive it. A ratchet turns one way: a measurement, once taken,
is never discarded, and the best-known kernel for a shape never regresses.

## Setup on a fresh machine (teammates start here)

Requirements: Linux or WSL2, an NVIDIA GPU, Python 3.10+, CUDA-enabled PyTorch, Triton.

```bash
git clone https://github.com/checkpoint214159/ratchet.git && cd ratchet
pip install pytest pytest-testmon
python3 -m ratchet.oracle.device      # calibrate for YOUR gpu -> ledger/device.json
./scripts/check-oracle.sh             # oracle integrity gate (must be green)
./.beryl/scripts/check.sh             # deterministic checks + full test suite
```

Calibration is per-machine and per-toolchain: never quote a number measured on someone
else's calibration. If `device.py` has no peak-FLOPs table entry for your GPU, add one
citing the whitepaper (the file explains how) — the ridge point is meaningless without it.

Do **not** run `bootstrap.sh`. It is the record of how this repo was originally built
from the handoff package; it refuses to run inside the working repo.

## Orientation

Read in order: `HANDOFF.md` → `docs/00-mission.md` → `docs/04-failure-modes.md` →
`docs/01-architecture.md` → `docs/02-milestones.md` → `specs/`.

`CLAUDE.md` is the binding contract: three zones (immutable oracle / evolvable
workspace / append-only ledger) and the five hard rules. Know which zone you are
editing before you edit.
