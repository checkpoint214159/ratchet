# bench/ — the empirical lane

Measurements against the announced competition matrix, on real hardware.

## Why this is separate from `research/`

`research/` is the fail-closed evidence archive: every empirical claim must pass a
ratified per-vendor qualification hierarchy (`FG-01`), and today no code path can mark a
backend `QUALIFIED`, so its `EvidenceClassification` admits only `NO_RUN` and
`SYNTHETIC`. That gate exists because it was authored on a machine with no GPU, and it
is the right default for a build that must never fabricate a number.

This directory is the other half: one team machine **does** have a CUDA device
(RTX 4070 Ti SUPER), and the competition submits on 2026-09-01. So measurements happen
here, plainly labelled as what they are — working measurements from an unqualified
backend, not ratified evidence.

**Nothing here is promoted into `research/archive/` implicitly.** Promotion requires the
NVIDIA qualification hierarchy to be written and ratified, an environment record that can
express an available CUDA device, and an executing measurement harness. Until then these
two lanes stay adjacent and honest about which is which.

## Git is the evolutionary tree

A candidate is a **commit**. Lineage is **git ancestry**. There is no second tree store.

```
                    ┌─ cand/g1/split-k ────────── 3f2a1c  score 2.41
  master ── a91f3d ─┤
   (baseline)       └─ cand/g1/fused-qkv ── 8c04be ─┬─ cand/g2/graph  d18e77  score 2.88
                                            score 1.84│
                                                      └─ cand/g2/fp16   b7c290  score 2.10
```

Four things this buys that a bespoke tree would have to reimplement:

- **Reproducibility is `git checkout <sha>`.** The exact source behind any score is
  recoverable, not described.
- **Merges are expressible.** Combining two promising candidates is a real merge commit
  with two real parents. A single-parent tree cannot represent that at all, and
  recombination is where a lot of evolutionary value lives.
- **Distribution is free.** Teammates on other machines push branches into the same tree
  with no coordination protocol.
- **Clade metaproductivity is reachability.** Pooling outcomes over a node's entire
  descendant subtree — the Huxley-Gödel correction to scoring nodes by their own
  performance — is just "which commits are reachable forward from here", computed once
  from `git rev-list --parents`.

### The rules this imposes

1. **Never record a measurement from a dirty tree without marking it.** A sha that
   doesn't describe the code that ran is a false provenance claim. Dirty rows are still
   recorded — they are evidence — but excluded from clade statistics and promotion.
2. **Never rebase, squash, amend or force-push a candidate branch.** Rewriting history
   silently reparents the tree and invalidates every statistic derived from it. Same rule
   the research archive states for its own events, for the same reason.
3. **A score is per (commit, config).** One commit yields up to 14 rows. Aggregates are
   derived views, never stored fields.

### Branch naming

```
cand/<generation>/<slug>        a candidate to be measured
```

Generation is a counter, not a guarantee of depth — a `g3` candidate may branch from a
`g1` commit if Thompson sampling picks that parent, and that is the point.

## The database

`bench/results.jsonl` — append-only, one JSON object per line, fsync'd per write. Facts
only. Never edited, sorted in place, or pruned.

Chosen over SQLite deliberately: JSONL merges cleanly when two teammates measure in
parallel, diffs readably in review, and survives a mid-write crash with the loss of one
line. Queries that want a relational view build one in memory from the rows — the rows
stay the source of truth.

Row shape:

```jsonc
{
  "schema": 1,
  "ts": "2026-08-29T…",
  "commit_sha": "8c04be…",     // WHAT CODE RAN — the primary key, with config_id
  "branch": "cand/g1/fused-qkv",
  "dirty": false,              // if true: excluded from clade stats
  "candidate": "fused-qkv-fp16-graph",
  "config_id": 1,              // row of the announced matrix
  "config": { … },             // denormalized so a row is readable standalone
  "status": "ok" | "incorrect" | "oom" | "compile_error" | "timeout" | "crash",
  "correctness": {"passed": true, "max_abs": …, "max_rel": …, "failed_elements": 0},
  "timing":  {"baseline_ms": …, "candidate_ms": …, "speedup": …, "method": …, "samples": …},
  "memory":  {"peak_alloc_bytes": …},
  "env":     {"device": …, "cc": "sm_89", "torch": …, "triton": …, "clocks_locked": false},
  "notes": ""
}
```

## Usage

```bash
python3 bench/matrix.py          # the 14 configs with derived feasibility figures
python3 bench/ledger.py          # scoreboard (score → sha) + clade metaproductivity
```

```python
from bench.ledger import BenchLedger, scoreboard, clade_stats, sample_parent

led = BenchLedger()
led.record(config_id=1, status="ok", candidate="fused-qkv-fp16-graph",
           timing={"baseline_ms": 2.28, "candidate_ms": 1.24, "speedup": 1.84,
                   "method": "cuda_event", "samples": 300, "reduction": "median"},
           correctness={"passed": True, "max_abs": 0.0011, "max_rel": 306.0,
                        "failed_elements": 0})

scoreboard(led)        # per-commit aggregate, sorted by weighted score
clade_stats(led)       # sha -> (successes, failures) pooled over descendants
sample_parent(led)     # Thompson-sampled commit to branch from next
```

`weighted_score` gives every config equal weight (the problem statement provides no
weighting and every row is a test case) and **clips speedups at 3×**, so one spectacular
regime cannot carry a submission that is mediocre everywhere else. A config with no
measurement scores 1.0 rather than being skipped — skipping would reward not measuring.
