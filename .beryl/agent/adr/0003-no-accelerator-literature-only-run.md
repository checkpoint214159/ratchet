# ADR 0003: Use A Literature-Only Run When No Accelerator Is Accessible

## Status

Accepted by explicit user instruction on 2026-08-29.

## Context

The backend doctor reports PyTorch unavailable, and the workspace exposes no `xpu-smi`,
`sycl-ls`, `/dev/dri`, or WSL `/dev/dxg` interface. This establishes that no qualified
PyTorch XPU runtime is available; it does not claim to enumerate every accelerator API.
Cached NVIDIA calibration describes a different historical environment. The user
directed the build not to perform empirical kernel iteration when a GPU is inaccessible
and to present the literature survey instead.

## Decision

Continue hardware-independent infrastructure with fake runtimes and synthetic test
fixtures. Record the environment as a provisional observation, then validate and import
it into the append-only archive as no-run evidence. Produce a cited
literature survey, proposed experiment protocol, and paper without latency, speedup,
memory, correctness, profiling, or current-best performance claims.

Intel Arc remains the first future empirical target. This hierarchy remains
literature-only even if XPU appears mid-build; a qualified runtime starts a new ratified
empirical hierarchy through FG-01.

## Consequences

- **Benefit:** The repository and research workflow can be verified without inventing
  evidence.
- **Benefit:** Future hardware access has an explicit resumable gate and experiment plan.
- **Tradeoff:** The current paper cannot claim a measured optimization or current best
  beyond an untuned fallback.
- **Follow-up:** A future run appends new environment and experiment events; it never
  rewrites this no-run record.
