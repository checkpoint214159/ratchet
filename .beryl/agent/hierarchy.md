# Initial Build Hierarchy

## Build Contract

- Scope: Research-driven, append-only, multi-vendor transformer optimization system;
  Intel Arc is the first measured target and the supplied evaluator controls acceptance.
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
  - `ruff format --check . && ruff check .`
  - `pytest -m 'not gpu'`
  - `./.beryl/scripts/check.sh`
- Status: pending
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
  - `.beryl/agent/agent-rules.md`
- Evidence: none

### IB-02 — Benchmark custody
- Parent: IB-00
- Dependencies: IB-01
- Deliverable: Byte-preserved reference benchmark relocation and semantic characterization.
- Acceptance checks:
  - Original SHA-256 remains `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`
  - `pytest tests/evaluation -q`
- Status: pending
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
  - `.beryl/agent/design-tree.md`
- Evidence: none

### IB-03 — Domain boundaries
- Parent: IB-00
- Dependencies: IB-02
- Deliverable: Public contracts for evaluation, models, backends, measurement, experiments, dispatch, optimization, and reporting.
- Acceptance checks:
  - `pytest tests/contracts -q`
  - forbidden-import inspection
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/ubiquitous-language.md`
  - `.beryl/agent/adr/0002-separate-evaluation-measurement-and-backends.md`
- Evidence: none

### IB-04 — Platform registry
- Parent: IB-03
- Dependencies: IB-03
- Deliverable: CPU, XPU, CUDA, and HIP adapters with explicit capabilities and validation states.
- Acceptance checks:
  - `pytest tests/backends -q`
  - unsupported capabilities fail clearly
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/design-tree.md`
- Evidence: none

### IB-05 — Intel qualification
- Parent: IB-04
- Dependencies: IB-04
- Deliverable: Actual Arc identity and qualified XPU runtime/compiler/timer/memory capabilities.
- Acceptance checks:
  - `python -m ratchet.backends doctor --backend xpu`
  - XPU allocation, SDPA, compile, synchronize, event, memory, and dtype probes
- Status: pending
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
  - `.beryl/agent/architecture.md`
- Evidence: Current shell has no visible Arc or PyTorch; empirical work stops if unchanged.

### IB-06 — Authoritative measurement harness
- Parent: IB-03
- Dependencies: IB-04, IB-05
- Deliverable: Correctness-first subprocess harness with synchronized timing, memory, provenance, timeout, and crash containment.
- Acceptance checks:
  - `pytest tests/measurement -q`
  - incorrect candidates have no timing; crash and timeout remain recorded
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/testing-policy.md`
- Evidence: none

### IB-07 — Immutable experiment archive
- Parent: IB-03
- Dependencies: IB-03, IB-06
- Deliverable: Append-only event catalogue, unique experiment IDs, artifacts, schemas, and deterministic projections.
- Acceptance checks:
  - `pytest tests/experiments -q`
  - projections rebuild byte-identically; duplicate IDs and mutation fail
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/ubiquitous-language.md`
- Evidence: none

### IB-08 — Literature and hypotheses
- Parent: IB-07
- Dependencies: IB-07
- Deliverable: Root literature trackers, bibliography, cited summaries, and auditable human hypothesis queue.
- Acceptance checks:
  - `pytest tests/literature -q`
  - bibliography keys resolve and read/to-read transitions preserve history
- Status: pending
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
- Evidence: none

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
- Dependencies: IB-05, IB-06
- Deliverable: Fair eager, compiled, SDPA, and available vendor baselines with compilation separated from steady state.
- Acceptance checks:
  - `pytest tests/benchmarks -q`
  - correctness and provenance for every baseline
- Status: pending
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: none

### IB-11 — Candidate seam
- Parent: IB-02
- Dependencies: IB-02, IB-03, IB-10
- Deliverable: Optimized model delegates through the evaluator's designated seam only.
- Acceptance checks:
  - protected-region AST/hash guard
  - weight-copy, mask, causal, and output contract tests
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
- Evidence: none

### IB-12 — Baseline profiling
- Parent: IB-10
- Dependencies: IB-10
- Deliverable: Arc operator profile and evidence-backed end-to-end bottleneck.
- Acceptance checks:
  - XPU profiler trace and operator breakdown
- Status: pending
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: If no GPU is accessible, publish literature survey and leave this node blocked.

### IB-13 — First Intel hypothesis
- Parent: IB-11
- Dependencies: IB-11, IB-12
- Deliverable: Genuine profiling-justified candidate, initially compiled SDPA unless evidence redirects it.
- Acceptance checks:
  - authoritative correctness matrix
  - paired synchronized XPU timing and peak memory
- Status: pending
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: If no GPU is accessible, no kernel iteration is performed.

### IB-14 — Multi-vendor dispatch
- Parent: IB-04
- Dependencies: IB-07, IB-10, IB-13
- Deliverable: Capability- and evidence-driven selection with explicit untuned fallbacks.
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
- Dependencies: IB-07, IB-14, IB-15, IB-16
- Deliverable: Idea-to-synthesis controller with correctness gating, comparison, recording, and bounded continuation.
- Acceptance checks:
  - `pytest tests/optimization/test_controller.py -q`
  - end-to-end dry-run records compile and correctness failures
- Status: pending
- Canonical context targets:
  - `.beryl/agent/architecture.md`
  - `.beryl/agent/design-tree.md`
- Evidence: none

### IB-18 — Search strategies
- Parent: IB-17
- Dependencies: IB-17
- Deliverable: Parametric and architectural search, caching, infeasibility recording, and random-search ablation.
- Acceptance checks:
  - `pytest tests/optimization/test_search.py -q`
- Status: pending
- Canonical context targets:
  - `.beryl/agent/design-tree.md`
- Evidence: none

### IB-19 — First autoresearch run
- Parent: IB-18
- Dependencies: IB-09, IB-13, IB-17, IB-18
- Deliverable: Real Intel baseline and at least one complete optimization experiment.
- Acceptance checks:
  - complete immutable `EXP-NNNN` with correctness, timing, comparisons, and decision
- Status: pending
- Canonical context targets:
  - `.beryl/agent/project-brief.md`
  - `.beryl/agent/design-tree.md`
- Evidence: If no GPU is accessible, this node is blocked and no empirical substitute is created.

### IB-20 — Research synthesis
- Parent: IB-09
- Dependencies: IB-19
- Deliverable: Statistics, meaningful visualization, concise narrative, negative evidence, and next hypothesis.
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
- Dependencies: IB-06, IB-07, IB-19
- Deliverable: Permanent numerical near-miss pool without tolerance changes.
- Acceptance checks:
  - `pytest tests/evaluation/test_adversarial_pool.py -q`
- Status: pending
- Canonical context targets:
  - `.beryl/agent/testing-policy.md`
- Evidence: none

### IB-22 — Critic and scout
- Parent: IB-17
- Dependencies: IB-08, IB-18, IB-21
- Deliverable: Citation-aware scout and candidate-held-out epoch-frozen critic.
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
- Deliverable: CUDA adapter, functional fallback, timing contract, and hardware-gated validation command.
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
- Deliverable: ROCm/HIP adapter, functional fallback, timing contract, and hardware-gated validation command.
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
  - required hardware gates or explicit blocked status
  - clean worktree and remote commit verification
- Status: pending
- Canonical context targets:
  - all relevant `.beryl/agent/*.md`
- Evidence: none
