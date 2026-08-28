# ADR 0002: Separate Evaluation, Measurement, And Accelerator Backends

## Status

Accepted by plan ratification on 2026-08-29.

## Context

The supplied evaluator defines a full-transformer workload and an absolute-OR-relative
correctness rule. The existing checksummed research oracle targets CUDA attention and
requires both error bounds. The supplied evaluator also synchronizes only CUDA, so its
host timing cannot substantiate Intel XPU performance.

## Decision

Keep three boundaries:

1. Protected evaluation owns transformer behavior and acceptance.
2. Measurement reuses that workload but obtains synchronized timing and memory evidence
   through an `AcceleratorBackend`.
3. Vendor adapters hide XPU, CUDA, and HIP objects and return vendor-neutral records.

Only the evaluator's designated optimized-model seam may delegate to `ratchet.models`.
The legacy attention oracle remains checksummed and supplementary. Intel Arc is the first
measured backend. CUDA and ROCm paths remain explicitly unvalidated until their hardware
gates execute.

## Consequences

- **Benefit:** Correctness semantics cannot drift to fit an optimization.
- **Benefit:** Intel measurements synchronize honestly without rewriting benchmark logic.
- **Benefit:** Vendor specialization remains possible without contaminating the domain.
- **Tradeoff:** Compatibility output and scientific performance evidence use different
  runners and must be labeled clearly.
- **Tradeoff:** NVIDIA and AMD support can be implemented here but not empirically
  validated on unavailable hardware.
- **Follow-up:** Characterization tests must protect every evaluator behavior outside the
  designated candidate seam before it is integrated.
