# One turn of the loop

Exact commands. Run everything from the repo root. `python3` is system Python 3.10 on the
CUDA machine (no venv); the CPU-only research lane uses `.venv` — see `README.md`.

## 0. Before anything

```bash
./scripts/check-oracle.sh            # Zone A integrity — the immutable oracle
git status --short                    # a dirty tree bars every measurement from stats [B2]
python3 bench/ledger.py               # current frontier + clade metaproductivity
tail -60 docs/findings/00-learnings.md   # what the last turn changed about how we work
```

If `check-oracle.sh` fails, stop. A kernel that only passes after an oracle change is a
wrong kernel.

## 1. Draw a parent and an idea

```python
from bench.ledger import BenchLedger, scoreboard, clade_stats_by_candidate, sample_candidate
led = BenchLedger()
parent = sample_candidate(led)        # WHICH node to expand — CMP over declared lineage
```

Then draw an idea from `bench/proposals/` by its Beta posterior (`specs/07`). Redraw if
the idea's preconditions are incompatible with the parent, and **log the rejection** — a
proposal that keeps being rejected is mis-specified and you want to see that.

## 2. Dispatch an expander

```bash
git worktree add ../wt-<slug> -b cand/<generation>/<slug> <parent-sha>
```

`<parent-sha>` is the parent's commit, **not** the tip of the working branch. Cutting from
the trunk is what silently flattened eighteen generations into one chain `[D3, C5]`.

Hand the expander `docs/loop/roles/expander.md`, the drawn idea, the parent's source, and
the device table from `docs/00-mission.md`.

## 3. Screen — 30 s, verdict only

```bash
python3 bench/screen.py --candidate v<N>_<slug> --parent <parent-name>
```

Four configs across four regimes (launch-bound, `head_dim 8`, wide model, mainstream).
Writes `bench/screen_log.jsonl`, which is **advisory and never feeds sampling**.

A screen kills what is clearly bad. It does not rank things that are statistically tied,
and it cannot see a robustness fix at all `[D8]` — those need a bespoke falsifier.

## 4. Confirm — 112 s, recorded

```bash
python3 bench/run_matrix.py --candidate v<N>_<slug>              # all configs
python3 bench/run_matrix.py --candidate v<N>_<slug> --ids 6 13   # a subset
python3 bench/run_matrix.py --candidate v<N>_<slug> --dry-run    # no ledger write
```

Correctness runs before timing, in the same process; a failing candidate is never timed.
One config per subprocess, so an OOM or a hang costs one row and not the run. An OOM is a
**result**, recorded with `status="oom"`.

The tree must be clean and the GPU lock free. `--allow-contended` exists for capability
probes only; a row taken with it is not comparable to one without.

## 5. Test

```bash
python3 -m pytest tests/bench/ -q       # lineage, ledger, screen, per-candidate contracts
./.beryl/scripts/check.sh               # the full deterministic gate
```

Every candidate lands with its own `tests/bench/test_v<N>_*.py`. If the candidate's value
is invisible to the sweep `[D8]`, that test **is** the evidence — it must pin the parent's
degradation so the fix cannot silently rot.

## 6. Write it down, then commit

Nothing important may live only in a session's context.

- **`docs/findings/NN-slug.md`** — what was measured and how, on which machine, with the
  method metadata. Inferences labelled as inferences.
- **`docs/findings/00-learnings.md`** — append an `## Lnn` entry **only if the turn should
  change what the loop does next**. Not a log of what happened.
- **`docs/findings/README.md`** — one row in the index table.
- **`bench/candidates/__init__.py`** — the registry entry, with the declared `parent`.

```bash
git add <explicit paths>          # NEVER `git add -A` — it sweeps a subagent's WIP
git commit -m "v<N>: <what it now does differently>"
```

Commit boundaries that have worked: one for the candidate, one for its finding, one for
any harness change. Never rebase, squash, amend or force-push a candidate branch `[B3]`.

## 7. Merge or close

```bash
git checkout ben && git merge --no-ff cand/<generation>/<slug>
git worktree remove ../wt-<slug>
```

Merge a candidate that beat its parent by more than the noise floor `[A6]`, **or** one
whose value is a proven robustness fix `[D8]`, **or** one whose failure closed a region —
a negative result is kept, never deleted `[B5]`.

If it neither won nor taught anything, say so in the finding and leave the branch
unmerged. The branch is the record; deleting it destroys the lineage that made its
successor findable.

---

## Periodic turns that are not candidate turns

| every | do | why |
|---|---|---|
| ~5 generations | **ablate the frontier** — fork it into one-mechanism-removed siblings | loops add; only ablation subtracts `[D9]` |
| ~5 generations | **an audit turn**: *what does this depend on that we never varied?* Vary the harness's own knobs — `--padding 0.5`, `--input-scale 0.01`, `--dtype float16` — and its **defaults** `[E2]` | 7 for 7; the search loop found none of them `[E1]` |
| whenever the rubric changes | re-run `python3 bench/proposals/backtest.py` | it found three defects before spending a GPU minute `[D12]` |
| whenever you add a guard | build a condition that makes it fire | a guard that has never fired is not evidence `[C1]` |
