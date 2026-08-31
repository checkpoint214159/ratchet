# Project Brief

## Product Goal

Build Ratchet for GPU-kernel researchers and coding agents so they can turn an
authoritative transformer workload into faster, correct, reproducible implementations
while retaining every experiment and continuously publishing the important conclusions.

Intel Arc is the first future measured target. The current workspace has no qualified
PyTorch XPU runtime, so the current delivery verifies hardware-independent infrastructure
and produces a cited literature survey without kernel iteration or performance claims.
NVIDIA CUDA and AMD ROCm/HIP remain first-class adapter contracts with defined fallbacks,
but runtime functionality is not described as validated until their hardware gates run.

## Authoritative Contract

- `docs/PROBLEM-STATEMENT.md` is the canonical repo-owned copy of the TikTok TechJam 2026
  problem #3 statement (last updated by the organizers 27 August 2026) and of the
  engineering contract derived from the evaluator. The organizer's source document is
  auth-gated, so this file is the only in-repo authority for the announced test shapes,
  the correctness bound, the judging weights, and the deliverables. Read it before any
  kernel, dispatch, or benchmark work.
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

Human steering is retained in a separate append-only planning queue. Each input remains
`planning_only` behind `FG-01`, is chained to its predecessor, and may create an idea or
attach a constraint, reviewed-literature reference, priority, or acyclic redirect. The
queue cannot create an optimization request, experiment event, candidate, measurement, or
backend action; a later controller may consume only its deterministic projection after
independently enforcing the active no-run gate.

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
| Intel XPU runtime | First future measured backend | `ratchet.backends` XPU adapter | Complete no-run path; defer qualification to FG-01 |
| CUDA / ROCm runtimes | Portable adapter contracts | matching backend adapter | Defined untuned fallback; runtime remains unverified |
| Compiler stacks | Future `torch.compile`, SYCL, or Triton protocols | backend adapter | Record the literature-backed protocol without execution |
| Git and GitHub | Provenance, experiment isolation, durable delivery | experiment workspace adapter | Retain local commits and report push failure |
| Tectonic | Reproducible LaTeX-to-PDF build | reporting adapter | Keep validated LaTeX and report missing PDF tool |
| Coding-agent/LLM runner | Architectural proposals and synthesis | optimization proposer adapter | Human/file-backed hypothesis queue |

## Definition Of Done

The initial build is complete only when every ratified hierarchy node passes, durable
context is promoted, the transient hierarchy is removed, `HANDOFF.md` is retired after
its useful content is preserved, and the completed commits are verified on GitHub.

Because no qualified PyTorch XPU runtime is available, no empirical kernel-iteration
claim may be made in this build. The system, literature survey, and paper are built
through a no-run gate: the environment observation and stop decision are facts, while
timing, correctness, memory, and speedup fields remain absent. Synthetic fixtures never
enter the production catalogue.

The first controller pass is recorded as `EXP-0001` / `EVT-000001`. It stopped at
`ENV-0001` because PyTorch XPU was unavailable, generated no candidate, and contains no
correctness, compilation, timing, memory, profile, trace, counter, comparison, speedup,
or current-best result. Its intended future protocol remains `PROTO-INTEL-0001` behind
FG-01.

## Current research interface

`./scripts/verify-autoresearch.sh` is the reproducible no-GPU entry point. It verifies the
archive, regenerates `research/paper/latest.pdf` in Tectonic cached/untrusted mode, and
runs Beryl's deterministic installed-project gate. Its successful output proves the
current state only: one verified no-run event and zero empirical events.

The regenerated paper records `EVT-000001` / `EXP-0001`, its XPU stop reason, the absence
of all result fields, three reviewed literature links, and the FG-01-gated
`PROTO-INTEL-0001` next hypothesis. The citation-aware scout may prepare only reviewed,
planning-only architectural intents; the candidate-held-out critic is explicitly dormant
until measured candidate evidence exists. Neither can produce a candidate, score,
measurement, or archive mutation in this build.
