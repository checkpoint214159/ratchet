# Ratchet

Ratchet is a research-driven transformer optimization environment. It preserves every
validated experiment in an append-only catalogue and turns the important evidence into a
concise LaTeX research paper.

## Fast, reproducible autoresearch setup

Requirements: Linux or WSL2, Python 3.10+, Git, [uv](https://docs.astral.sh/uv/), and
[Tectonic](https://tectonic-typesetting.github.io/) with its local bundle available.

```bash
git clone https://github.com/checkpoint214159/ratchet.git && cd ratchet
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
chmod +x scripts/verify-autoresearch.sh
./scripts/verify-autoresearch.sh
```

The default install is **CPU-only**: it carries no `torch`/`triton`, so the verification
path stays dependency-light. Install the vendor framework with the separate `runtime`
extra only on a machine that intends to qualify a backend:

```bash
uv pip install --python .venv/bin/python -e '.[dev,runtime]'
```

That single verification command is the supported low-friction entry point. It verifies
the append-only archive, rebuilds `research/paper/latest.pdf` using cached/untrusted
Tectonic, and runs every deterministic Beryl check. It fails fast with a useful error if
the virtual environment or Tectonic prerequisites are missing. It never generates a
candidate, invokes a backend, or manufactures a benchmark result.

Open `research/paper/latest.pdf` after a successful run for the current research state:
the reviewed literature, `EVT-000001` no-run evidence, its traceable future hypothesis,
and the boundary against empirical claims.

## Research workflow

1. Add a durable human idea, constraint, priority, redirect, or reviewed literature input
   through the append-only planning queue—not by editing historic events.
2. Define a finite, canonical, scoreless parametric or architectural search plan.
3. On qualified hardware, isolate a candidate worktree, validate correctness first, then
   measure synchronized steady-state execution and append immutable evidence.
4. Regenerate the paper. The catalogue contains every result; the paper selectively
   reports traceable conclusions and important negative findings.

## Running research with yourself in the loop

You are not a spectator of the optimizer. The paper is the read side of the loop and the
planning queue is the write side. A full turn of the loop is: read the PDF, append your
own input, let the input become a plan, run it on qualified hardware, then read the
regenerated PDF.

### Step 1 — read the current state

```bash
./scripts/verify-autoresearch.sh
xdg-open research/paper/latest.pdf
```

The PDF is the only artefact you need to read to know what has been tried, what the
evidence boundary is, and what the next hypothesis is. Do not read raw logs to steer.

### Step 2 — append your idea, constraint, priority, literature, or redirect

Human input enters through the append-only planning queue. It is chained and
hash-linked: you add records, you never edit or delete them.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from ratchet.optimization import (
    FileHumanResearchQueue, HumanInputKind, HumanInputSubmission,
)

queue = FileHumanResearchQueue(Path.cwd())

# 1. A new research question of your own.
queue.append(HumanInputSubmission(
    recorded_at="2026-08-29T12:00:00Z",          # UTC, your own timestamp
    actor="your-name",
    kind=HumanInputKind.IDEA,
    idea_id="IDEA-0002",                          # next free IDEA-NNNN
    statement="Is attention or the MLP block the dominant steady-state cost?",
    literature_keys=("dao2022flashattention",),   # optional, must already be read
))

# 2. A constraint the optimizer is not allowed to ignore.
queue.append(HumanInputSubmission(
    recorded_at="2026-08-29T12:05:00Z", actor="your-name",
    kind=HumanInputKind.CONSTRAINT, idea_id="IDEA-0002",
    statement="No result may be accepted from a single sequence length.",
))

# 3. A strategic priority. Higher wins in the projection ordering.
queue.append(HumanInputSubmission(
    recorded_at="2026-08-29T12:06:00Z", actor="your-name",
    kind=HumanInputKind.PRIORITY, idea_id="IDEA-0002",
    statement="Investigate before any low-level kernel work.", priority=10,
))
PY
```

The five kinds are the five ways you can steer:

| Kind           | Use it to                              | Extra field         |
| -------------- | -------------------------------------- | ------------------- |
| `IDEA`       | open a new research question           | —                  |
| `CONSTRAINT` | forbid an outcome you consider invalid | —                  |
| `PRIORITY`   | reorder what is investigated first     | `priority`        |
| `LITERATURE` | attach reviewed sources to an idea     | `literature_keys` |
| `REDIRECT`   | retire an idea in favour of another    | `redirect_to`     |

Rules the queue enforces for you, so a mistake fails loudly instead of quietly:

- `CONSTRAINT`, `PRIORITY`, `LITERATURE`, and `REDIRECT` must reference an `idea_id` that
  an earlier `IDEA` record already created.
- Every `literature_keys` entry must appear in `papers_read.md` **and** in
  `research/paper/bibliography.bib`. Read the paper first, add both entries, then cite it.
  Backlog entries in `papers_to_read.md` alone are rejected.
- Records are planning-only and gated behind `FG-01`. Appending an idea never runs a
  benchmark and never produces a performance claim.

### Step 3 — see what the optimizer will act on

```bash
.venv/bin/python -c '
from pathlib import Path
from ratchet.optimization import FileHumanResearchQueue
projection = FileHumanResearchQueue(Path.cwd()).projection()
print("projection", projection.projection_id[:16])
for item in projection.items:
    print(f"  {item.idea_id} priority={item.priority} "
          f"constraints={len(item.constraints)} :: {item.statement}")
'
```

Redirected ideas drop out of the projection but stay in the record. If your input is not
in the projection, it will not influence the next experiment — fix that before running.

### Step 4 — turn the queued idea into a plan, then run it

1. Write a finite, canonical, **scoreless** search plan with
   `ratchet.optimization.plan_search`, or a protocol under `research/protocols/`. Plans
   enumerate what to try; they never store a result.
2. Probe the vendor runtime (`python -m ratchet.backends --backend xpu|cuda|hip`). A
   positive probe only means a device exists.
3. Complete the vendor qualification hierarchy for that backend. Until it passes, the
   controller is fail-closed and records a no-run event instead of a number.
4. On qualified hardware: create an isolated experiment worktree, run the authoritative
   correctness matrix **before** any timing, take synchronized alternating steady-state
   measurements, then append the immutable event.
5. Regenerate the paper and go back to step 1:

```bash
.venv/bin/python -m ratchet.reporting build-paper
```

Accepted, rejected, and inconclusive events all stay in the archive. The paper may
summarize selectively, but `FileExperimentArchive.verify()` keeps every summary traceable
to the underlying event, so a negative result cannot be dropped to improve the story.

### What to do when you disagree with the optimizer

Do not hand-edit an event, a figure, or a generated `.tex` file. Append a `REDIRECT` or a
`CONSTRAINT` naming the idea you want abandoned and the reason. That keeps your judgement
in the permanent record and stops the search converging on a local maximum you have
already rejected.

## Collaborating on separate branches

Two people can chase different hypotheses at once without corrupting each other's
evidence. Isolation is per-experiment worktree, and integration is an explicit,
deterministic consolidation step — not an ad-hoc merge.

### Branch layout

`ExperimentWorkspaceManager` owns experiment branch names. It creates:

```text
ratchet/experiments/<exp-nnnn>/proto-<protocol-slug>/<lane>
```

and consolidates into:

```text
ratchet/experiments/integration/proto-<protocol-slug>/<16-hex-digest>
```

The **lane** is the collaboration unit: pick one lane name per person or per agent
(`alice`, `bob`, `fusion`, `tiling`) so two researchers can attack the same experiment and
protocol independently. Lanes must match `^[a-z][a-z0-9-]{0,31}$`.

### Start your own lane

```bash
mkdir -p ../ratchet-worktrees
.venv/bin/python - <<'PY'
from pathlib import Path
from ratchet.experiments.workspaces import (
    ExperimentWorkspaceManager, ExperimentWorkspaceSpec,
)

manager = ExperimentWorkspaceManager(
    repository=Path.cwd().resolve(),
    worktree_root=Path("../ratchet-worktrees").resolve(),   # must be outside the repo
)
workspace = manager.create(ExperimentWorkspaceSpec(
    experiment_id="EXP-0002",
    protocol_id="PROTO-INTEL-0001",
    protocol_digest="<sha256 of the protocol file>",
    lane="alice",
    base_commit="<full 40-char base commit>",
))
print(workspace.branch, workspace.path)
PY
```

The manager binds an exact base commit and protocol digest to the branch, so two lanes are
always comparable. It refuses to reuse an existing branch or path, and it never pushes,
never deletes a branch, never executes a candidate, and never appends evidence by itself.

### Work, then finalize

Commit inside your worktree as normal. When the lane is clean:

```python
provenance = manager.finalize(workspace)   # records branch, base, head, changed paths
```

`finalize` requires a clean worktree and produces a `WorkspaceProvenance` digest. That
digest is what makes your lane mergeable and auditable; keep it with the experiment
record.

### Merge two lanes

```python
result = manager.consolidate([alice_provenance, bob_provenance])
print(result.status)              # "consolidated" | "conflict" | "already_exists"
print(result.integration_branch)  # deterministic ratchet/experiments/integration/... ref
print(result.conflicts)           # populated only when status == "conflict"
```

Consolidation is deterministic: the same set of lanes always yields the same integration
commit and branch name. On conflict it **reports** the conflicting paths and creates
nothing — it will not force a resolution. Resolve the conflict in a lane, re-`finalize`,
and consolidate again.

After consolidation, `manager.cleanup(workspace, provenance, result)` removes the worktree
but retains the source branch, so a merged lane remains independently reproducible.

### Rules for shared work

- Never rebase, squash, force-push, or revert anything under `research/archive/`. It is
  append-only; a correction is a *new* event that references the old one.
- Never rewrite `research/ideas/intake/`. It is hash-chained; a rewrite breaks
  verification for everyone.
- Run `./scripts/verify-autoresearch.sh` in your lane before asking for a merge, and open
  a pull request rather than pushing to `master`.
- Regenerate the paper on the integration branch, not in a lane. Two lanes each editing
  `research/paper/generated/` will conflict on derived files that nobody should hand-edit.
- Compare lanes by evidence, not by narrative: identical base commit, identical protocol
  digest, and the authoritative correctness matrix passing in both.

Read [`docs/research-process.md`](docs/research-process.md) for the full process,
[`docs/benchmarking.md`](docs/benchmarking.md) for the measurement contract,
[`docs/hardware-support.md`](docs/hardware-support.md) for vendor gates, and
[`docs/experiments.md`](docs/experiments.md) for provenance and recovery rules.
See [`docs/architecture.md`](docs/architecture.md) for the bounded-context map and
[`docs/optimization-principles.md`](docs/optimization-principles.md) for the standing,
hardware-agnostic optimization contract.

## Agent and Beryl orientation

`AGENTS.md` routes implementation work through the Beryl initial-build or feature
workflow. The durable current architecture, vocabulary, test policy, and decisions live
under [`.beryl/agent/`](.beryl/agent/). Run `./.beryl/scripts/check.sh` after any change;
the installed-project command intentionally does not use `--development`.

`HANDOFF.md` has been retired: its useful safety principles are preserved in this README,
Beryl's canonical project context, and the documents above. Do not run `bootstrap.sh`; it
is historical setup material and refuses to run inside the working repository.
