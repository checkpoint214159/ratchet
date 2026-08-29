# Architecture

The durable, current architecture of Ratchet lives in Beryl's canonical project context:

- [`.beryl/agent/architecture.md`](../.beryl/agent/architecture.md) — bounded contexts,
  their public entry points, and the invariants that hold across them.
- [`.beryl/agent/design-tree.md`](../.beryl/agent/design-tree.md) — the module map.
- [`.beryl/agent/ubiquitous-language.md`](../.beryl/agent/ubiquitous-language.md) — the
  vocabulary every context shares.

This file is a stable pointer so a reader who looks for `docs/architecture.md` (the layout
in the master research brief) is routed to the authoritative source rather than a
duplicate that can drift.

## One-paragraph orientation

Ratchet is organized as separated bounded contexts under `ratchet/`: `backends` (vendor
adapters behind a uniform probe/capability contract), `oracle` (the immutable correctness
and measurement core), `measurement`, `evaluation`, `experiments` (the append-only
archive and worktree lifecycle), `optimization` (human queue, search, scout, critic, and
the fail-closed controller), `dispatch` (evidence-driven backend selection), and
`reporting` (the deterministic paper pipeline). Cross-context imports are restricted to
public entry points and enforced by `tests/contracts/`. Vendor SDK objects never cross a
context boundary, and no empirical result is admitted until a backend passes its ratified
qualification hierarchy.

The historical single-GPU design notes (`docs/00-mission.md` through
`docs/04-failure-modes.md`, `specs/`, `prompts/`) are retained as provenance only; they
carry a historical banner and are not current instructions.
