# Ratchet — handoff package

A specification package for an agentic GPU kernel optimization harness, to be handed to a
Claude Code instance running in WSL2 with an NVIDIA GPU.

**Start at `HANDOFF.md`.**

```
HANDOFF.md              entry point: read order, first actions, the five rules
CLAUDE.md               durable agent rules — this is what stays in context
bootstrap.sh            exact setup sequence, incl. Beryl (read its header first)

docs/
  00-mission.md         what & why, competition constraints, tiering, timeline
  01-architecture.md    the three-zone design and the invariants
  02-milestones.md      M0–M14 with acceptance gates
  03-research-dossier.md papers + reference implementations + API facts that churn
  04-failure-modes.md   how kernel benchmarks lie — read before writing measurement code

specs/
  01-measurement-core.md  the oracle
  02-ledger.md            append-only schema + derived views
  03-search-loop.md       two-level search, algorithm selection, objective
  04-dispatch.md          self-calibrating shape × device dispatch
  05-critic.md            Tier 2 co-evolving critic
  06-scout.md             the research subagent

seed/
  ratchet/oracle/       WORKING CODE — device, inputs, reference, correctness, timing
  ratchet/ledger.py     WORKING CODE — append-only ledger + derived views
  prompts/              proposer and scout role prompts
```

The `seed/` code is real and tested where it can be tested without a GPU. Copy it in
rather than rewriting it; the comments explain why each line is there.
