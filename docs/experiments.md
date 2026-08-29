# Experiments and provenance

The experiment archive is authoritative and append-only in spirit. Events use separate
`EVT-NNNNNN` identifiers, experiments use `EXP-NNNN`, and environment observations use
`ENV-NNNN`. Each append is schema-validated, content-addressed where appropriate, locked,
fsync-backed, and projected deterministically. Duplicate IDs, changed event bytes, broken
references, and interrupted transactions are rejected or recovered before a projection is
returned.

`FileExperimentArchive.verify()` is the required integrity check. The paper reads only its
verified public projection and records its projection identifier in generated data. Historic
evidence is never edited to change a conclusion; corrections or new outcomes are later
events.

`EXP-0001` is the model for a valid no-run event: it identifies the blocked XPU
environment, evaluator, plan, provenance, motivation, and intended protocol while omitting
all empirical result fields. A future empirical event must retain candidate provenance,
correctness and measurement evidence, comparisons, decision rationale, and artifacts.

Experiment worktrees are created only by `ExperimentWorkspaceManager`; they bind an exact
base commit and protocol and preserve source branches during consolidation. They never
push, delete a branch, execute a candidate, or append evidence by themselves.
