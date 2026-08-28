# Initial Build Hierarchy

## Build Contract

- Scope: Research-driven, append-only, multi-vendor transformer optimization system;
  Intel Arc is the first future measured target and the supplied evaluator controls
  acceptance. This build produces hardware-independent infrastructure and a cited
  literature survey because no qualified PyTorch XPU runtime is available in the
  current workspace.
- Non-goals: training/backward, distributed execution, cross-vendor performance parity,
  hostile-code sandboxing, benchmark manipulation, or unvalidated performance claims.
- Ratified on: 2026-08-29
- Last updated: 2026-08-29
- Completion rule: every node and required check passes, durable context is promoted

## Nodes

### IB-00 — Contract and hierarchy
- Parent: root
- Dependencies: none
- Deliverable: Ratified scope, canonical boundaries, and Git-tracked hierarchy.
- Acceptance checks:
  - `./.beryl/scripts/check-md.sh`
  - `./.beryl/scripts/check-initial-build-workflow.sh`
  - `git ls-files .beryl/agent/hierarchy.md`
- Status: complete
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
  - `.beryl/agent/design-tree.md`
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/ubiquitous-language.md`
  - `.beryl/agent/testing-policy.md`
  - `.beryl/agent/adr/0002-separate-evaluation-measurement-and-backends.md`
- Evidence: Ratified 2026-08-29; `check-md` and initial-build workflow passed;
  hierarchy is not ignored and enters CB-01.

### IB-01 — Beryl and test readiness
- Parent: IB-00
- Dependencies: IB-00
- Deliverable: Clean manifest, real Python checks, and documented installed-target Beryl state.
- Acceptance checks:
  - `./.beryl/scripts/check-tests-unchanged.sh`
  - `.venv/bin/ruff format --check . && .venv/bin/ruff check .`
  - `pytest -m 'not gpu'`
  - `./.beryl/scripts/check.sh`
- Status: complete
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
  - `.beryl/agent/agent-rules.md`
- Evidence: manifest excludes ignored bytecode; direct simulated-missing-torch collection,
  `pytest -m 'not gpu'` (2 passed), native Ruff format/lint, agent doctor, and
  `./.beryl/scripts/check.sh` passed on 2026-08-29. Generated `.testmondata*` caches
  are ignored; the recoverable tracked `.testmondata-shm` and `.testmondata-wal` files
  were intentionally retired from tracking on 2026-08-29.

### IB-02 — Benchmark custody
- Parent: IB-00
- Dependencies: IB-01
- Deliverable: Byte-preserved reference benchmark relocation and semantic characterization.
- Acceptance checks:
  - Original SHA-256 remains `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`
  - `pytest tests/evaluation -q`
- Status: complete
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
  - `.beryl/agent/design-tree.md`
- Evidence: Reference relocated byte-for-byte; SHA-256 matched exactly. CPU-only source
  characterization, including the no-`benchmarks.reference`-import guard (5 passed),
  Ruff format/lint, and `./.beryl/scripts/check.sh` (7 passed) succeeded on 2026-08-29.

### IB-03 — Domain boundaries
- Parent: IB-00
- Dependencies: IB-02
- Deliverable: Public contracts for evaluation, models, backends, measurement, experiments, dispatch, optimization, and reporting.
- Acceptance checks:
  - `pytest tests/contracts -q`
  - forbidden-import inspection
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/ubiquitous-language.md`
  - `.beryl/agent/adr/0002-separate-evaluation-measurement-and-backends.md`
- Evidence: Torch-free public records/protocols and AST boundary checks added; 26 contract
  tests, including nested cross-context rejection, Ruff format/lint, manifest check, and
  `./.beryl/scripts/check.sh` (33 passed) succeeded on 2026-08-29.

### IB-04 — Platform registry
- Parent: IB-03
- Dependencies: IB-03
- Deliverable: CPU, XPU, CUDA, and HIP adapters with explicit capabilities and validation states.
- Acceptance checks:
  - `pytest tests/backends -q`
  - unsupported capabilities fail clearly
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/design-tree.md`
- Evidence: Fake-runtime CPU tests cover CPU/XPU/CUDA/HIP event timing, compilation,
  explicit memory-reset lifecycle, read-only memory observation, unavailable runtime/device,
  and doctor output (16 passed); public contracts, including unavailable-capability
  contradictions, passed (31 total contract tests). Ruff format/lint, manifest,
  related-test command, full pytest (54 passed), and `./.beryl/scripts/check.sh` passed
  on 2026-08-29. No real accelerator was used; IB-05 is the XPU availability gate.

### IB-05 — XPU availability decision
- Parent: IB-04
- Dependencies: IB-04
- Deliverable: Recorded XPU availability decision and an enforced no-empirical-work gate.
- Acceptance checks:
  - `.venv/bin/python -m ratchet.backends --backend xpu`
  - unavailable runtime prevents kernel generation, profiling, and measurement
- Status: complete
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
  - `.beryl/agent/architecture.md`
- Evidence: Provisional observation `ENV-0001` records backend doctor exit 2 (`PyTorch is
  not installed`), no `xpu-smi`, no `sycl-ls`, no `/dev/dri`, and no `/dev/dxg` on
  2026-08-29. ADR 0003 activates the literature-only branch; no kernel candidate or
  empirical result was produced. IB-07 will validate and import this observation.

### IB-06 — Measurement orchestration contract
- Parent: IB-03
- Dependencies: IB-04
- Deliverable: Correctness-first subprocess harness contract for synchronized timing,
  memory, provenance, timeout, and crash containment, verified only with fake backends
  here; authoritative evaluator execution remains unvalidated.
- Acceptance checks:
  - `pytest tests/measurement -q`
  - incorrect candidates have no timing; crash and timeout remain recorded
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/testing-policy.md`
- Evidence: The public measurement surface is an unconditional no-run gate with no
  successful execution path. Correctness-first subprocess, timeout, crash, synchronized
  timing, and memory lifecycle mechanics are isolated under `tests/fixtures/` and always
  emit synthetic-classified evidence. Constructor invariants prevent no-run facts from
  representing anything except unavailable status. On 2026-08-29, 11 measurement tests,
  43 measurement plus contract tests, Ruff format/lint, manifest checks, and architecture
  and slice reviews passed. No PyTorch or accelerator execution occurred.

### IB-07 — Immutable experiment archive
- Parent: IB-03
- Dependencies: IB-03
- Deliverable: Append-only event catalogue, unique experiment IDs, artifacts, versioned
  empirical and no-run schemas, validation/storage of `ENV-0001` as immutable
  provenance, and deterministic projections.
- Acceptance checks:
  - `pytest tests/experiments -q`
  - projections rebuild byte-identically; duplicate IDs and mutation fail
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/ubiquitous-language.md`
- Evidence: Separate `EVT-NNNNNN`, `EXP-NNNN`, and `ENV-NNNN` identities, exact
  conditional success/failure/no-run schemas, protected evaluator provenance,
  content-addressed artifacts, comparison-regime checks, and deterministic projections
  are implemented. A process lock and validated fsync'd journal recover interrupted
  target-plus-manifest transactions and reject tampering. `ENV-0001` is immutable
  unavailable-XPU provenance only and forbids empirical append; IB-19 alone may append
  the no-run experiment event. Process races, crash recovery, finite/recomputed timing
  statistics, referential integrity, and no-run result-field exclusion are covered. On
  2026-08-29, 41 focused tests, Ruff, manifest and Beryl checks, plus architecture and
  slice reviews passed. No GPU work occurred.

### IB-08 — Literature and hypotheses
- Parent: IB-07
- Dependencies: IB-07
- Deliverable: Root literature trackers, bibliography, cited summaries, and auditable human hypothesis queue.
- Acceptance checks:
  - `pytest tests/literature -q`
  - bibliography keys resolve and read/to-read transitions preserve history
- Status: complete
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
- Evidence: `papers_read.md` records nine read primary sources with source URLs and
  DOIs where assigned; `papers_to_read.md` holds three explicit unread backlog items.
  BibTeX and chained immutable read-transition records align with the same keys, and
  hardware-gated `IDEA-0001` queues the follow-on question without creating an experiment
  protocol. The project retires IPEX as a future integration path without making a
  product-retirement claim. On 2026-08-29, `pytest tests/literature -q` (7 passed),
  repository-wide Ruff, manifest verification, and `./.beryl/scripts/check.sh` passed.
  No GPU or empirical work occurred.

### IB-09 — Paper pipeline
- Parent: IB-07
- Dependencies: IB-07, IB-08
- Deliverable: Concise catalogue-derived LaTeX, figures, tables, selection logic, and validated latest PDF.
- Acceptance checks:
  - `python -m ratchet.reporting build-paper`
  - deterministic regeneration and Tectonic PDF validation
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/testing-policy.md`
- Evidence: none

### IB-10 — Baseline portfolio
- Parent: IB-04
- Dependencies: IB-02, IB-03, IB-04
- Deliverable: Reproducible eager, compiled, SDPA, and vendor baseline definitions with
  compilation separated from steady state; no baseline is executed in this build.
- Acceptance checks:
  - `pytest tests/benchmarks -q`
  - configuration and provenance contract for every baseline definition
- Status: complete
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: Four vendor-neutral, definition-only Intel-XPU future baseline
  configurations (eager, compiled, SDPA, and vendor-library) retain the complete
  reference-transformer workload and authoritative evaluator SHA-256. Compilation,
  first-run, and steady-state reporting are separated and synchronized; the compiled
  definition fixes Inductor backend/mode/fullgraph/dynamic policy. SDPA and oneDNN Graph
  SDPA substitutions are attention-core-only and preserve weight-copy, valid-mask, causal,
  and output semantics. The oneDNN Graph partition-dispatch check remains required and
  unverified; all configurations remain unavailable and unvalidated. On 2026-08-29, 11
  CPU-only portfolio tests, Ruff format/lint, manifest verification, and
  `./.beryl/scripts/check.sh` passed. No PyTorch, candidate, kernel, or accelerator
  execution occurred.

### IB-11 — Candidate seam
- Parent: IB-02
- Dependencies: IB-02, IB-03, IB-10
- Deliverable: The evaluator's designated customization seam is structurally
  characterized without importing PyTorch, implementing, or timing an optimized kernel.
- Acceptance checks:
  - protected-region AST/hash guard
  - source-level weight-copy, mask, causal, and output contract characterization
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
- Evidence: none

### IB-12 — Baseline profiling
- Parent: IB-10
- Dependencies: IB-08, IB-10
- Deliverable: Cited literature survey of likely Arc transformer bottlenecks, clearly
  distinguished from a project profile.
- Acceptance checks:
  - every bottleneck claim traces to a reviewed primary source
  - all unmeasured project-specific claims are labelled as hypotheses
- Status: pending
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: No XPU profiler trace is permitted in the current environment.

### IB-13 — First Intel hypothesis
- Parent: IB-11
- Dependencies: IB-08, IB-11, IB-12
- Deliverable: Literature-backed future Intel experiment protocol queued as
  `not_run_hardware_unavailable`; no candidate implementation is generated.
- Acceptance checks:
  - explicit hypothesis, shapes, correctness tolerances, timing method, and stop criteria
  - no empirical result fields are populated
- Status: pending
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: If no GPU is accessible, no kernel iteration is performed.

### IB-14 — Multi-vendor dispatch
- Parent: IB-04
- Dependencies: IB-04, IB-07, IB-10
- Deliverable: Capability- and evidence-driven selection with explicit untuned fallbacks;
  every unavailable or unmeasured backend remains untuned.
- Acceptance checks:
  - `pytest tests/dispatch -q`
  - decisions respond to profile perturbations; no evaluator detection
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/ubiquitous-language.md`
- Evidence: none

### IB-15 — Human research input
- Parent: IB-08
- Dependencies: IB-08
- Deliverable: Auditable intake for ideas, constraints, literature, priorities, and redirects.
- Acceptance checks:
  - `pytest tests/optimization/test_human_queue.py -q`
- Status: pending
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
- Evidence: none

### IB-16 — Independent experiment branches
- Parent: IB-07
- Dependencies: IB-01, IB-07
- Deliverable: Safe branch/worktree lifecycle bound to experiment provenance.
- Acceptance checks:
  - two isolated dry-run worktrees and conflict-safe consolidation
- Status: pending
- Canonical context targets:
  - `.beryl/agent/agent-rules.md`
  - `.beryl/agent/architecture.md`
- Evidence: none

### IB-17 — Autoresearch controller
- Parent: IB-14
- Dependencies: IB-05, IB-07, IB-14, IB-15, IB-16
- Deliverable: Idea-to-synthesis controller with correctness gating, comparison,
  recording, and bounded continuation; unavailable hardware stops before candidate
  generation, compilation, correctness execution, or timing.
- Acceptance checks:
  - `pytest tests/optimization/test_controller.py -q`
  - end-to-end no-run path records the unavailable reason without empirical, profile,
    trace, or counter fields
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/design-tree.md`
- Evidence: none

### IB-18 — Search strategies
- Parent: IB-17
- Dependencies: IB-17
- Deliverable: Parametric and architectural search mechanics, caching, infeasibility
  recording, and random-search ablation, exercised with test-only synthetic fixtures.
- Acceptance checks:
  - `pytest tests/optimization/test_search.py -q`
- Status: pending
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: none

### IB-19 — First autoresearch run
- Parent: IB-17
- Dependencies: IB-05, IB-07, IB-13, IB-17
- Deliverable: Immutable no-run research event proving the controller stopped because
  hardware was unavailable.
- Acceptance checks:
  - complete immutable event with environment, intended protocol, and unavailable reason
  - timing, memory, correctness, speedup, profile, trace, counter, and current-best fields
    are absent
- Status: pending
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
  - `.beryl/agent/design-tree.md`
- Evidence: No synthetic or historical measurement may substitute for the missing run.

### IB-20 — Research synthesis
- Parent: IB-09
- Dependencies: IB-08, IB-09, IB-19
- Deliverable: Concise cited literature survey, source-derived visualization, explicit
  no-run disclosure, and an evidence-backed next hypothesis.
- Acceptance checks:
  - every paper claim traces to experiment IDs or cited literature
  - `research/paper/latest.pdf` regenerates from repository data
- Status: pending
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
  - `.beryl/agent/testing-policy.md`
- Evidence: Without GPU, the paper is a literature survey and makes no empirical claims.

### IB-21 — Adversarial pool
- Parent: IB-17
- Dependencies: IB-02, IB-06, IB-07
- Deliverable: Test-only synthetic numerical near-miss fixtures without tolerance changes;
  fixtures never enter the experiment catalogue or paper evidence.
- Acceptance checks:
  - `pytest tests/evaluation/test_adversarial_pool.py -q`
- Status: pending
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
- Evidence: none

### IB-22 — Critic and scout
- Parent: IB-17
- Dependencies: IB-08, IB-18, IB-21
- Deliverable: Citation-aware scout and candidate-held-out epoch-frozen critic; the scout
  remains active for literature while the performance critic remains dormant without
  measurements.
- Acceptance checks:
  - `pytest tests/optimization/test_critic.py tests/optimization/test_scout.py -q`
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/project-brief.md`
- Evidence: none

### IB-23 — NVIDIA support path
- Parent: IB-04
- Dependencies: IB-04, IB-06, IB-14
- Deliverable: CUDA adapter contract, defined untuned fallback, timing contract, and
  hardware-gated validation command; runtime functionality remains unverified here.
- Acceptance checks:
  - backend contract tests
  - documented CUDA integration gate
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
- Evidence: Remains unvalidated without NVIDIA hardware.

### IB-24 — AMD support path
- Parent: IB-04
- Dependencies: IB-04, IB-06, IB-14
- Deliverable: ROCm/HIP adapter contract, defined untuned fallback, timing contract, and
  hardware-gated validation command; runtime functionality remains unverified here.
- Acceptance checks:
  - backend contract tests
  - documented ROCm integration gate
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
- Evidence: Remains unvalidated without AMD hardware.

### IB-25 — Durable documentation
- Parent: IB-00
- Dependencies: IB-20, IB-21, IB-22, IB-23, IB-24
- Deliverable: README, agent contract, required docs, current best, failures, questions, reproduction, Beryl guide, and demo.
- Acceptance checks:
  - clean-checkout documentation walkthrough
  - Markdown and link checks
- Status: pending
- Canonical context targets:
  - all relevant `.beryl/agent/*.md`
- Evidence: none

### IB-26 — Handoff retirement
- Parent: IB-25
- Dependencies: IB-25
- Deliverable: Remove stale `HANDOFF.md` after promoting its useful information.
- Acceptance checks:
  - preservation audit
  - `git ls-files HANDOFF.md` returns empty
- Status: pending
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
  - `.beryl/agent/architecture.md`
- Evidence: User ratified preservation/loss map and deletion on 2026-08-29.

### IB-27 — Completion and push
- Parent: root
- Dependencies: IB-00, IB-01, IB-02, IB-03, IB-04, IB-05, IB-06, IB-07, IB-08, IB-09, IB-10, IB-11, IB-12, IB-13, IB-14, IB-15, IB-16, IB-17, IB-18, IB-19, IB-20, IB-21, IB-22, IB-23, IB-24, IB-25, IB-26
- Deliverable: Full review, all available gates, durable context, hierarchy deletion, final commits, and verified GitHub push.
- Acceptance checks:
  - formatter, narrow suites, `./.beryl/scripts/check.sh`
  - recorded no-run gate and zero blocked active-scope nodes
  - clean worktree and remote commit verification
- Status: pending
- Canonical context targets:
  - all relevant `.beryl/agent/*.md`
- Evidence: none

## Deferred Hardware Qualification

The following future gate is deliberately outside this build's active completion graph:

- **FG-01 — Intel Arc qualification:** identify the device and validate XPU allocation,
  SDPA, compilation, synchronization, event timing, memory observation, and supported
  dtypes on an accessible Intel runtime. Successful qualification starts a new ratified
  empirical hierarchy; it does not rewrite this build's no-run evidence.
