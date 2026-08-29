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
| Cross-vendor shape | Capability adapters with explicit availability and validation state | 2026-08-29 | [ADR 0002](adr/0002-separate-evaluation-measurement-and-backends.md) |
| Research history | Append-only events plus rebuildable projections | 2026-08-29 | [ADR 0002](adr/0002-separate-evaluation-measurement-and-backends.md) |
| No-accelerator branch | Stop kernel iteration; build infrastructure and cited literature survey | 2026-08-29 | [ADR 0003](adr/0003-no-accelerator-literature-only-run.md) |
| Baseline portfolio | Four definition-only Intel-XPU future full-workload baselines; substituted attention is semantically constrained and oneDNN Graph dispatch remains unverified | 2026-08-29 | — |
| Bottleneck evidence | Keep source observations separate from explicitly unmeasured project hypotheses until qualified profiling exists | 2026-08-29 | — |
| First future Intel protocol | Compare definition-only compiled and eager full-workload arms only after FG-01 qualification | 2026-08-29 | — |
| Current autoresearch controller | Validate the exact unavailable environment first and prepare one canonical no-run event without executing or mutating the archive | 2026-08-29 | — |

## Pressure Points

- The current shell exposes neither Intel Arc nor a compatible PyTorch installation.
- The supplied evaluator synchronizes only CUDA and cannot substantiate XPU latency.
- The legacy oracle is CUDA/Triton attention-specific and semantically stricter.
- Literature motivates attention data movement, work partitioning, graph compilation,
  configuration sensitivity, XMX, and Triton-XPU scheduling as future hypotheses; none is
  an observed Ratchet bottleneck, so qualified profiling must control optimization direction.
- NVIDIA and AMD adapters can be implemented and contract-tested here but not honestly
  performance-validated without their hardware.
- Synthetic objectives may test orchestration mechanics but may never be stored or
  reported as empirical measurements.

## Recording Rule (Design Tree vs ADR)

Update this file for evolving choices and observed pressure. Create an ADR when a choice
changes module ownership, persistence schema, adapter contracts, protected evaluation,
security boundaries, shared vocabulary, or test strategy.
