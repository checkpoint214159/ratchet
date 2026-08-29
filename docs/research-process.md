# Research process

Ratchet separates immutable evidence from planning and presentation. `research/archive/`
contains append-only event facts; `research/ideas/intake/` contains an append-only human
planning queue; `research/paper/` is a reproducible, selective projection.

## Current state

`ENV-0001` establishes that PyTorch XPU is unavailable. `EVT-000001` / `EXP-0001` is a
verified no-run event that binds that environment, the authoritative evaluator digest,
`IDEA-0001`, and `PROTO-INTEL-0001`. It has no candidate, correctness, compile, timing,
memory, profile, trace, counter, comparison, speedup, or current-best field. It is not an
empirical failure and cannot support a performance claim.

## Steering and search

Humans add ideas, constraints, priorities, reviewed literature references, and redirects
through `ratchet.optimization.FileHumanResearchQueue`. The queue is chained and
planning-only. Search plans are finite canonical parametric or architectural spaces;
their caches retain considered and infeasible points but never scores. A citation-aware
scout creates only FG-01-gated architectural intents. The critic is epoch-frozen and
explicitly dormant until empirical candidate evidence exists.

## Qualified future loop

After user ratification of FG-01: probe the selected vendor, define a worktree-bound
candidate, run authoritative correctness before any timing, use synchronized alternating
steady-state measurements, append immutable evidence, and regenerate the paper. Failed,
inconclusive, and successful events remain in the archive. The paper may summarize them,
but must remain traceable and must not erase adverse evidence.

## Literature

Only reviewed sources in `papers_read.md` may be cited. Future reading belongs first in
`papers_to_read.md`; a chained transition in `research/literature/history/` preserves the
move. Literature motivates hypotheses but never becomes a Ratchet result.
