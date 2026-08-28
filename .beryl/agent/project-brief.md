# Project Brief

## Product Goal

Build Ratchet for GPU-kernel researchers and coding agents so they can turn an
authoritative transformer workload into faster, correct, reproducible implementations
while retaining every experiment and continuously publishing the important conclusions.

Intel Arc is the first measured target. NVIDIA CUDA and AMD ROCm/HIP are first-class
adapter paths with functional PyTorch fallbacks, but they are not described as validated
until their hardware gates run.

## Authoritative Contract

- `torch_transformer_benchmark.py` is the authoritative evaluator supplied by the user.
- Its baseline, CLI meaning, input generation, and absolute-OR-relative correctness rule
  are protected. Its original SHA-256 is
  `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`.
- Relocation must be byte-for-byte before any integration change.
- Only the designated `UserOptimizedTransformer` seam may delegate to optimized code.
- Genuine Intel performance evidence comes from a synchronized XPU sidecar using the
  same models, weights, inputs, and configurations; the evaluator's current non-CUDA
  host timing is not performance evidence.
- The stricter CUDA attention research oracle remains separate and does not decide
  authoritative transformer acceptance.

## Primary Workflows

1. **Run research:** accept or generate a hypothesis, isolate it, build it, validate
   correctness, benchmark it, compare it, and append an auditable experiment record.
2. **Steer research:** let a human add literature, constraints, hypotheses, and
   priorities that remain visible to the controller.
3. **Understand research:** regenerate a concise LaTeX paper and PDF from the complete
   catalogue, with traceable conclusions, negative findings, and next directions.
4. **Deploy the best implementation:** select a candidate using measured workload and
   backend capabilities, with an explicit untuned fallback.

## Scope

- Inference for the supplied full-transformer workload on one accelerator.
- Runtime discovery and isolated adapters for Intel XPU, NVIDIA CUDA, AMD ROCm/HIP, and
  a CPU correctness-only reference path.
- Append-only experiment and measurement history, content-addressed artifacts,
  independent experiment branches/worktrees, human research input, and automated search.
- Literature-to-hypothesis traceability through exactly `papers_read.md` and
  `papers_to_read.md` at repository root plus a machine-readable bibliography.
- A concise catalogue-derived LaTeX paper and continuously regenerated `latest.pdf`.

## Non-Goals

- Training/backward, distributed, or multi-GPU execution.
- Equal performance or unearned validation claims across vendors.
- OS-grade containment of actively malicious candidate code.
- Benchmark detection, hardcoded benchmark outputs, weakened correctness, or a
  performance claim based on asynchronous host submission timing.
- Importing third-party kernels without compatible licensing and attribution.

## External Systems

| System | Why it exists | Interface owner | Failure fallback |
| --- | --- | --- | --- |
| Intel XPU runtime | First measured backend | `ratchet.backends` XPU adapter | Mark hardware nodes blocked; produce literature-only research output |
| CUDA / ROCm runtimes | Portable functional paths | matching backend adapter | Explicit unvalidated PyTorch fallback |
| Compiler stacks | `torch.compile`, optional SYCL/Triton candidates | backend adapter | Eager/vendor-library candidate |
| Git and GitHub | Provenance, experiment isolation, durable delivery | experiment workspace adapter | Retain local commits and report push failure |
| Tectonic | Reproducible LaTeX-to-PDF build | reporting adapter | Keep validated LaTeX and report missing PDF tool |
| Coding-agent/LLM runner | Architectural proposals and synthesis | optimization proposer adapter | Human/file-backed hypothesis queue |

## Definition Of Done

The initial build is complete only when every ratified hierarchy node passes, durable
context is promoted, the transient hierarchy is removed, `HANDOFF.md` is retired after
its useful content is preserved, and the completed commits are verified on GitHub.

If no accelerator is accessible, no empirical kernel-iteration claim may be made. The
system, literature survey, and paper may still be built, but hardware-dependent hierarchy
nodes remain explicitly blocked rather than being satisfied with fabricated or cached
measurements.
