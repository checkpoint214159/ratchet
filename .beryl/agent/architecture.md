# Architecture

## Organizing Rule

Acceptance semantics, accelerator execution, measurement orchestration, immutable facts,
optimization policy, and human-facing synthesis are separate responsibilities. A backend
may specialize execution but may return only vendor-neutral domain records.

## Bounded Contexts

| Context | Owns | Does Not Own | Public Entry Point |
| --- | --- | --- | --- |
| Evaluation | Transformer configuration, reference/baseline model, input and correctness semantics, protected benchmark seam | Timing, device APIs, search policy | `ratchet/evaluation/__init__.py` |
| Benchmarks | Definition-only eager, compiled, SDPA, and vendor-library baseline protocols and evaluator custody | Framework imports, model construction, timing, or backend calls | `ratchet/benchmarks/__init__.py` |
| Models | Candidate transformer implementations and weight-compatible public model factory | Evaluation criteria, timers, catalogue writes | `ratchet/models/__init__.py` |
| Backends | Accelerator discovery, synchronization, timing primitives, memory, compilation, capabilities | Workload correctness, promotion, narrative | `ratchet/backends/__init__.py` |
| Measurement | Current-build no-run gate and future correctness-first orchestration contract | Candidate generation, synthetic execution, device-specific objects, paper selection | `ratchet/measurement/__init__.py` |
| Experiments | Separate event/experiment/environment IDs, journaled append-only events, artifacts, exact evidence schemas, projections | GPU execution, hypothesis generation, LaTeX presentation | `ratchet/experiments/__init__.py` |
| Dispatch | Evidence- and capability-driven candidate selection | Timing, benchmark semantics, vendor SDK calls | `ratchet/dispatch/__init__.py` |
| Optimization | Human queue, proposer adapters, parametric/architectural search, accept/reject policy | Mutating evaluator, measurement facts, vendor clients | `ratchet/optimization/__init__.py` |
| Reporting | Statistics, importance selection, figures, tables, LaTeX/PDF generation | Editing catalogue facts, executing candidates | `ratchet/reporting/__init__.py` |
| Legacy attention oracle | Existing checksummed CUDA attention reference and diagnostics | Authoritative transformer acceptance or Intel timing | `ratchet/oracle/__init__.py` |

Baseline configurations are definitions, never executable adapters. Every definition keeps
the complete `reference_transformer` workload and authoritative evaluator custody. SDPA and
vendor-library variants may substitute only the attention core while preserving weight-copy
compatibility, valid-mask and causal semantics, and the output contract. The current Intel
vendor declaration names oneDNN Graph SDPA, remains unavailable and unvalidated, and requires
oneDNN Graph partition inspection before any future result can claim that dispatch.

Evaluation publishes a Torch-free structural contract for the evaluator's designated
`UserOptimizedTransformer` class, its unchanged forward signature, and causal/valid-mask/
output obligations. `BaselineTransformer` inheritance is recorded as the current observed
implementation, not a future requirement. The contract also records strict-by-default
baseline-to-candidate copying, the evaluator-supported `strict=False` path, and the
alternative to customize `copy_model_weights` when parameter names differ. It is not an
executable candidate factory; future work must retain the protected evaluator hash and
implement only after a separately ratified hardware-qualified hierarchy.

## AcceleratorBackend Public Contract

`ratchet.backends` exposes one vendor-neutral `AcceleratorBackend` protocol:

- `probe() -> BackendIdentity`
- `capabilities() -> BackendCapabilities`
- `synchronize() -> None`
- `time(callable, config) -> TimingEvidence`
- `reset_memory_stats() -> None`
- `memory_stats() -> MemoryEvidence`
- `compile_model(model, policy) -> CompiledModel`

Vendor objects and SDK types remain internal to `ratchet/backends/xpu/`,
`ratchet/backends/cuda/`, and `ratchet/backends/hip/`. Domain records contain ordinary
Python values and stable enums only. `BackendCapabilities` separates current runtime/device
availability from the evidence-backed validation state; an available adapter is not thereby
qualified for empirical claims.

## Boundary Rules

1. A context imports only another context's public entry point.
2. Evaluation never imports measurement or optimization.
3. Measurement may call evaluation, models, and backends, then append through the public
   experiments API; it may not choose future candidates.
4. Optimization may read experiment projections and request measurements; it may not
   edit protected evaluation or historical events.
5. Reporting is a pure reader of catalogue projections and literature records.
6. Dispatch compares evidence only within the same evaluation contract, backend,
   device, toolchain, dtype, and configuration.
7. CPU timing is diagnostic only; it is never accelerator-performance evidence.

## Forbidden Import Policy

- `ratchet/evaluation/** -> ratchet/measurement/**`
- `ratchet/evaluation/** -> ratchet/optimization/**`
- `ratchet/models/** -> ratchet/backends/*/internal/**`
- `ratchet/measurement/** -> ratchet/backends/{xpu,cuda,hip}/**`
- `ratchet/experiments/** -> torch.{xpu,cuda}/**`
- `ratchet/reporting/** -> ratchet/measurement/**`
- `ratchet/optimization/** -> ratchet/oracle/internal/**`

## Trust And Data Flow

```text
Human/literature -> Optimization -> Models -> Measurement
                                            |        |
                                    Evaluation    Backends
                                            \        /
                                             Evidence
                                                |
                                          Experiments
                                           /        \
                                      Dispatch    Reporting -> PDF
```

The experiment event log and measurement artifacts are facts. Rankings, dispatch tables,
critic predictions, charts, and paper prose are derived views and may be regenerated.
Archive writes use a process lock and a validated, fsync'd recovery journal spanning the
content-addressed target and manifest. Recovery accepts only the recorded before or after
state and verifies the completed archive before removing the journal.

## No-Accelerator Execution

This build is unconditionally literature-only after the recorded XPU availability
decision. If XPU becomes available, FG-01 redirects empirical work into a new ratified
hierarchy rather than changing this build in place. Once the archive exists, the
controller validates the provisional environment observation and appends
`NoRunEvidence` with environment identity, planned hypothesis, and stop reason. It then
stops before candidate generation, compilation, correctness, profiling, timing, or
memory collection. Timing, memory, correctness, speedup, profile, trace, counter, and
current-best fields must be absent, not zero or synthetic. The gate cannot be bypassed
by selecting XPU, CUDA, HIP, or a host timer in the current build.

Synthetic timing and objective fixtures live only under `tests/fixtures/`. They exercise
selection mechanics but are prohibited from the production experiment catalogue and
paper results. A literature claim carries a bibliography key; it is never phrased as a
project measurement.
