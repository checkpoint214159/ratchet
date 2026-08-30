# The loop: roles, and why it is shaped this way

One human, one orchestrator, N subagents, **one GPU**. Every structural decision below
falls out of that last fact.

```
                          ┌──────────────────────────┐
        human steering ──▶│      ORCHESTRATOR        │◀── the ledger, the findings,
        (redirects,       │  owns the GPU lock,      │    the frontier
         priorities)      │  the ledger, the merges  │
                          └───┬──────────┬───────┬───┘
                              │          │       │
              ┌───────────────┘          │       └───────────────┐
              ▼                          ▼                       ▼
      ┌───────────────┐        ┌──────────────────┐     ┌────────────────┐
      │  RESEARCHER   │  ...   │    EXPANDER      │ ... │    VERIFIER    │
      │  (exploration)│ 2-4 of │  (exploitation)  │ 2-5 │  (adversarial) │
      │  reads, cites │        │  writes kernels  │     │  breaks claims │
      │  NEVER measures│       │  in own worktree │     │  builds checks │
      └───────────────┘        └──────────────────┘     └────────────────┘
              │                          │                       │
              └──── proposals ───────────┴──── candidates ───────┘
                          all flow back to the orchestrator,
                          which is the ONLY thing that measures
```

## Why the orchestrator is the only thing that measures

Not a style preference — `[A2]`. Two processes on one GPU produce two wrong numbers, and
the corruption is invisible after the fact. A sweep that overlapped a research agent's
Triton benchmarking is not re-derivable; the row is simply suspect forever `[F26, F27]`.

So: **measurement is serialized through one holder of `bench/gpu_lock.py`.** Subagents
either take the lock or do not touch the GPU. A subagent's throwaway probe knows nothing
about the lock file, which is exactly why the lock is mandatory rather than advisory in
practice.

The second reason is `[A3]`: every ad-hoc probe opts out of all six harness guarantees at
once. Centralizing measurement is how you stop N agents each inventing their own
half-correct timing loop.

## Why the queue, not ideation, is the bottleneck

Five expanders produce candidates faster than one GPU can confirm them. A confident
verdict costs 15–20 minutes of GPU under a ±7% noise floor. So the pipeline is two-stage
`[D7]`:

| stage | tool | cost | writes to |
|---|---|---|---|
| screen | `bench/screen.py` | ~30 s, 4 configs | `bench/screen_log.jsonl` (advisory) |
| confirm | `bench/run_matrix.py` | ~112 s, all 14 | `bench/results.jsonl` (the ledger) |

Most ideas are wrong. They should die for 30 seconds, not 20 minutes.

## The four roles

### Orchestrator — owns the GPU, the ledger, and the tree

Holds: the lock, `main`/`ben`, every merge, every ledger write, the running learnings.
Does *not* write kernels and does *not* do literature search. Its job is scheduling under
a scarce resource and refusing bad evidence.

Per turn it: draws a parent by clade `[D2]`, draws an idea from the scored distribution
`[D5]`, pairs them, dispatches an expander, screens what comes back, confirms survivors,
writes the finding, merges or closes, and appends a learning if the turn changed what the
loop should do next.

### Researcher — exploration, and never a number

2–4 running concurrently, each given a **disjoint territory** so their convergence means
something. When two independent researchers with disjoint territories found the same
mechanism, the corroboration was real evidence about the *reading*, not about the value
`[L34, F22]`.

Reads papers, kernel repos, issue trackers, vendor changelogs. Emits scored proposals
against the rubric in `specs/07-proposal-rubric.md`, each with a resolving citation and a
regime predicate. **Never runs on the GPU.** A researcher that measures is a researcher
that corrupts someone else's sweep.

### Expander — exploitation, one candidate, one worktree

2–5 running concurrently. Each gets one drawn (parent, idea) pair and an isolated git
worktree, branches `cand/<generation>/<slug>` **cut from its parent's commit, not from the
trunk** `[D3, C5]`, writes the candidate, registers it in `bench/candidates/__init__.py`
with its declared parent, writes its unit tests, and hands back a branch. It does not
measure and it does not merge.

### Verifier — adversarial, and the highest-value role we added late

Its entire job is section **C** of the method. Given a claim, it asks: can this check
fail? was the subject ever built? is there a positive control? is this structural claim
enforced by anything executable? Four of our most expensive errors were caught by a human
looking rather than by the system noticing — this role is the attempt to make that
systematic rather than lucky.

## Concurrency rules

1. **One measurement at a time, through the lock.** `[A2]`
2. **A researcher never touches the GPU.**
3. **An expander works in its own worktree.** Shared checkouts mean one agent's
   in-progress file lands in another's commit — `git add -A` inside the loop did exactly
   this `[L12 notes]`. Stage explicit paths, never `-A`.
4. **A run that was blocked by the lock is BLOCKED, not a rejected candidate.** Recording
   a guard refusal as a verdict on the candidate is how you lose good ideas to scheduling.
5. **Diversity is enforced at the queue, not at the scorer** `[D11]` — no more than 2 of
   any 5 queued ideas may share a mechanism class. A scorer asked to be diverse will
   simply claim diversity.

## Territory assignment, concretely

Give each researcher a named vein and the list of what has already been mined
(`docs/findings/` + `bench/proposals/`). Overlapping territories waste the corroboration
signal; unassigned territories are how a whole regime goes uninvestigated for twenty
generations — `head_dim = 8` was labelled "NEVER INVESTIGATED" in the rubric for exactly
this reason.
