# Design Tree

## Current Design Concept

Ratchet is an evidence ratchet: protected evaluation defines acceptance, backend
adapters make measurement honest for each accelerator, an append-only catalogue retains
every observation, and all dispatch, search, visualization, and paper output is a
rebuildable projection. Vendor optimization is specialized; research memory is shared.

## Open Decisions

| Decision | Options | Current Lean | Why |
| --- | --- | --- | --- |
| Intel implementation after profiling | Native PyTorch/Inductor, IPEX, SYCL, Intel Triton-XPU | Native PyTorch/Inductor first | Lowest toolchain risk; lower-level work requires measured headroom |
| First bottleneck response | SDPA, whole-graph compile, norm/FFN fusion, custom kernel | Profile then test compiled SDPA | It is falsifiable and portable, but attention may be a small end-to-end share |
| Candidate sandbox | spawned clean worktree, container/VM | spawned clean worktree | Crash containment is required; hostile-code security is out of scope |
| PDF milestone archival | every run, decisive narrative changes | decisive changes | `latest.pdf` stays current without filling Git with redundant binaries |

## Settled Decisions

| Decision | Choice | Date | ADR |
| --- | --- | --- | --- |
| Authoritative acceptance | Supplied full-transformer evaluator, executable OR tolerance | 2026-08-29 | [ADR 0002](adr/0002-separate-evaluation-measurement-and-backends.md) |
| Intel evidence | Synchronized sidecar over identical workload/configuration | 2026-08-29 | [ADR 0002](adr/0002-separate-evaluation-measurement-and-backends.md) |
| First measured vendor | Intel Arc/XPU | 2026-08-29 | [ADR 0002](adr/0002-separate-evaluation-measurement-and-backends.md) |
| Cross-vendor shape | Capability adapters with explicit validation state | 2026-08-29 | [ADR 0002](adr/0002-separate-evaluation-measurement-and-backends.md) |
| Research history | Append-only events plus rebuildable projections | 2026-08-29 | [ADR 0002](adr/0002-separate-evaluation-measurement-and-backends.md) |

## Pressure Points

- The current shell exposes neither Intel Arc nor a compatible PyTorch installation.
- The supplied evaluator synchronizes only CUDA and cannot substantiate XPU latency.
- The legacy oracle is CUDA/Triton attention-specific and semantically stricter.
- The default transformer is likely dominated by projections and FFN rather than only
  the attention core, so profiling must control optimization direction.
- NVIDIA and AMD adapters can be implemented and contract-tested here but not honestly
  performance-validated without their hardware.

## Recording Rule (Design Tree vs ADR)

Update this file for evolving choices and observed pressure. Create an ADR when a choice
changes module ownership, persistence schema, adapter contracts, protected evaluation,
security boundaries, shared vocabulary, or test strategy.
