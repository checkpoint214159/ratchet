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
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/testing-policy.md`
- Evidence: `.venv/bin/python -m ratchet.reporting build-paper` deterministically regenerates
  marked catalogue/literature/no-run inputs plus a local evidence-boundary figure, and writes a Tectonic-validated
  `research/paper/latest.pdf`. Selection resolves exactly the nine reviewed bibliography
  keys and the verified immutable archive projection; its nine reviewed sources and zero
  experiment events lead only to literature synthesis and no empirical conclusions. The
  empty catalogue accepts empirical/comparative tokens only in exact controlled generated
  disclosures and rejects them in every hand-authored included TeX source, including negated
  and spelled-out-ratio wording. TeX and bibliography dependencies use fixed local allowlists,
  and Tectonic is untrusted and cached-only; two cached builds produce byte-identical PDFs.
  On 2026-08-29, `pytest tests/reporting tests/literature -q` (30 passed), repository-wide
  Ruff, manifest verification, `pytest -m 'not gpu' -q` (160 passed), paper build, and
  `./.beryl/scripts/check.sh` passed. No GPU,
  PyTorch, candidate, hypothesis protocol, or empirical work occurred.

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
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
- Evidence: Torch-free evaluation records define the protected evaluator path/hash,
  designated `UserOptimizedTransformer` class and unchanged forward signature, current
  observed `BaselineTransformer` inheritance, strict-by-default weight copy with the
  evaluator-supported `strict=False` path, and the supported `copy_model_weights`
  customization path for parameter-name differences. Valid-mask/causal/output obligations
  remain fixed. Source-only AST tests prove these facts against the byte-preserved evaluator
  and reject executable integration state; no candidate factory, framework import, optimized
  candidate, or kernel exists. On 2026-08-29, 11 evaluation tests, 42 evaluation-plus-
  contract tests, Ruff format/lint, manifest verification, and `./.beryl/scripts/check.sh`
  passed. No PyTorch or accelerator execution occurred.

### IB-12 — Baseline profiling
- Parent: IB-10
- Dependencies: IB-08, IB-10
- Deliverable: Cited literature survey of likely Arc transformer bottlenecks, clearly
  distinguished from a project profile.
- Acceptance checks:
  - every bottleneck claim traces to a reviewed primary source
  - all unmeasured project-specific claims are labelled as hypotheses
- Status: complete
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: `LIT-SURVEY-0001` records six primary-source observations and six explicitly
  unmeasured, hardware-gated project hypotheses in aligned JSON and Markdown. Exact-schema,
  unique-ID, first-column citation resolution, per-hypothesis label/gate, and no-present-result
  checks passed with `pytest tests/literature -q` (10 passed), plus repository Ruff and
  `git diff --check` on 2026-08-29. Design and slice reviews approved the separation of
  literature fact from Ratchet protocol choice. No XPU profile, timing, candidate, kernel,
  or empirical result was produced.

### IB-13 — First Intel hypothesis
- Parent: IB-11
- Dependencies: IB-08, IB-11, IB-12
- Deliverable: Literature-backed future Intel experiment protocol queued as
  `not_run_hardware_unavailable`; no candidate implementation is generated.
- Acceptance checks:
  - explicit hypothesis, shapes, correctness tolerances, timing method, and stop criteria
  - no empirical result fields are populated
- Status: complete
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: `PROTO-INTEL-0001` defines the hardware-gated eager-versus-compiled Intel
  full-workload hypothesis with four exact float32 cases, conditional BF16/FP16 expansion,
  executable evaluator tolerances, synchronized block timing, stop ordering, and promotion
  thresholds. Exact-schema and recursive forbidden-field tests prove it contains no event,
  candidate, result, comparison, artifact, or decision payload. `pytest tests/protocols -q`
  passed 7 tests with Ruff, manifest, and diff checks on 2026-08-29; architecture and slice
  reviews approved. Status remains `not_run_hardware_unavailable`, and no production module,
  archive event, accelerator call, candidate, kernel, PyTorch execution, or empirical result
  was created.

### IB-14 — Multi-vendor dispatch
- Parent: IB-04
- Dependencies: IB-04, IB-07, IB-10
- Deliverable: Capability- and evidence-driven selection with explicit untuned fallbacks;
  every unavailable or unmeasured backend remains untuned.
- Acceptance checks:
  - `pytest tests/dispatch -q`
  - decisions respond to profile perturbations; no evaluator detection
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/ubiquitous-language.md`
- Evidence: The pure dispatch policy binds candidate evidence to immutable event membership
  in one verified projection and exact evaluator/configuration/profile/backend identity.
  CPU, zero-event, unavailable, unqualified, unsupported-dtype, missing-event/memory,
  compilerless, mismatched, or sub-threshold cases select immutable vendor-specific eager
  fallbacks. Only fully qualified evidence can produce a provenance-bound tuned decision,
  ranked deterministically by paired lower-bound speedup then implementation/event identity.
  `pytest tests/dispatch tests/experiments tests/contracts tests/reporting -q` passed 117
  tests with Ruff and `./.beryl/scripts/check.sh` on 2026-08-29; architecture and slice
  reviews approved. Current repository evidence selects no tuned implementation, and no
  GPU, PyTorch, candidate, kernel, synthetic archive record, or empirical result was used.

### IB-15 — Human research input
- Parent: IB-08
- Dependencies: IB-08
- Deliverable: Auditable intake for ideas, constraints, literature, priorities, and redirects.
- Acceptance checks:
  - `pytest tests/optimization/test_human_queue.py -q`
- Status: complete
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
- Evidence: A frozen public human-input contract and locked file-backed queue retain
  contiguous `HRI-NNNNNN` records in a canonical SHA-256 chain. `HRI-000001` preserves
  immutable `IDEA-0001` semantics and custody; subsequent records enforce exact kind rules,
  reviewed-literature resolution, planning-only FG-01 scope, priority ordering, constraint
  accumulation, first-occurrence literature deduplication, and acyclic redirects. Atomic
  writes, shared/exclusive process locks, symlink/partial/tamper/deletion/rewrite rejection,
  and concurrent append/read behavior are covered. `pytest tests/optimization/
  test_human_queue.py tests/contracts -q` passed 48 tests with Ruff, manifest, and diff
  checks on 2026-08-29; architecture and slice reviews approved. The queue exposes no
  controller conversion, archive write, backend call, candidate, or empirical field.

### IB-16 — Independent experiment branches
- Parent: IB-07
- Dependencies: IB-01, IB-07
- Deliverable: Safe branch/worktree lifecycle bound to experiment provenance.
- Acceptance checks:
  - two isolated dry-run worktrees and conflict-safe consolidation
- Status: complete
- Canonical context targets:
  - `.beryl/agent/agent-rules.md`
  - `.beryl/agent/architecture.md`
- Evidence: The project-owned workspace manager derives exact experiment/protocol/lane
  branches and external direct-child paths, binds protocol bytes at an immutable base commit,
  and finalizes clean descendant provenance with actual changed paths. Local no-remote tests
  create two isolated worktrees, consolidate disjoint commits deterministically through an
  atomic generated integration ref, retain sources, and prove sorted conflict refusal. Safety
  regressions cover malformed identity, path/ref/root races, symlinks, false provenance,
  stale or forged integration, dirty/unconsolidated cleanup, command allowlisting, and no
  force/network/destructive Git operations. `pytest tests/experiments/test_workspaces.py -q`
  passed 32 tests with Ruff, manifest, and `./.beryl/scripts/check.sh` on 2026-08-29;
  architecture and slice reviews approved. No archive event, remote mutation, candidate,
  accelerator, framework, or empirical work was performed.

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
- Status: complete
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
- Evidence: Six frozen `SYN-NUM-*` scalar fixtures pin exact/just-over absolute and
  relative thresholds, negative-reference magnitude handling, and an additive-tolerance
  trap against the evaluator's executable `abs_ok | rel_ok` rule. AST and repository scans
  prove no transformer configuration overlap, production/test import, archive reference,
  benchmark reference, or paper source/bibliography reference; the authoritative benchmark
  hash remains unchanged. `pytest tests/evaluation/test_adversarial_pool.py -q` passed 7
  tests with Ruff, manifest, and diff checks on 2026-08-29; architecture and slice reviews
  approved. No production API, tolerance change, candidate, GPU, PyTorch, archive event, or
  paper evidence was created.

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
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
- Evidence: The lazy CUDA adapter rejects HIP-built PyTorch, reports float32 by default and
  BF16 only after a positive device probe, and requires both events and synchronization for
  timing plus reset/allocated APIs for peak-memory support. Fake-runtime tests cover missing
  and optional capability surfaces, compilation, lifecycle, doctor output, source isolation,
  and the vendor-specific untuned fallback. The integration guide names the exact future
  hardware gate and makes no qualification claim. On 2026-08-29, the combined vendor-focused
  suite passed 16 tests with Ruff and diff checks; architecture and slice reviews approved.
  No NVIDIA hardware, PyTorch runtime, candidate, kernel, timing, or empirical result was used.

### IB-24 — AMD support path
- Parent: IB-04
- Dependencies: IB-04, IB-06, IB-14
- Deliverable: ROCm/HIP adapter contract, defined untuned fallback, timing contract, and
  hardware-gated validation command; runtime functionality remains unverified here.
- Acceptance checks:
  - backend contract tests
  - documented ROCm integration gate
- Status: complete
- Canonical context targets:
  - `.beryl/agent/architecture.md`
- Evidence: The lazy HIP adapter requires a HIP-built PyTorch runtime, keeps PyTorch's internal
  CUDA-compatible namespace behind the adapter, and exposes only HIP public identity. It
  reports float32 by default and BF16 only after a positive device probe; event, compilation,
  and allocated-peak capability failures are explicit while reserved peak memory is optional.
  Fake-runtime tests cover lifecycle, doctor output, source isolation, CUDA/HIP rejection,
  and the vendor-specific untuned fallback. The integration guide names the exact future
  hardware gate and makes no qualification claim. On 2026-08-29, the combined vendor-focused
  suite passed 16 tests with Ruff and diff checks; architecture and slice reviews approved.
  No AMD hardware, PyTorch runtime, candidate, kernel, timing, or empirical result was used.

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
