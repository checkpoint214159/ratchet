# Durable decisions and handoff preservation audit

## HANDOFF.md retirement — 2026-08-29

The root `HANDOFF.md` was the entry point for an earlier NVIDIA/WSL competition
specification package. Its useful content is preserved as follows:

| Former content | Preserved location | Treatment |
| --- | --- | --- |
| One-way evidence, immutable oracle, locked tolerances, correctness-first, append-only history | `.beryl/agent/project-brief.md`, `.beryl/agent/architecture.md`, `.beryl/agent/testing-policy.md`, `docs/benchmarking.md`, `docs/experiments.md` | Current invariant |
| Read order, setup, hardware probe, and recovery guidance | `README.md`, `AGENTS.md`, `.beryl/agent/task-routing.md`, `docs/hardware-support.md` | Replaced with current Beryl/XPU workflow |
| Tier rationale, critic/scout caution, benchmark failure modes, and research reading | `docs/00-mission.md` through `docs/04-failure-modes.md`, `specs/`, `papers_read.md`, `papers_to_read.md` | Historical source material; not current operating instructions |
| Original NVIDIA RTX 4070 Ti hardware facts, CUDA oracle/ledger layout, bootstrap sequence, and claim that this is only a specification package | None | Intentionally not preserved as current fact; contradicts the implemented Intel-first no-run repository |

No unique active requirement remains only in `HANDOFF.md`. Its deletion is authorized by
the ratified initial-build hierarchy after all dependent validation gates pass.
