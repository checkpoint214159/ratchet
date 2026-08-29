"""Measurement ledger, keyed to git commits.

APPEND-ONLY. A measurement is a fact about hardware at a moment in time; it is never
edited, deleted, sorted in place, or pruned. Rankings and clade statistics are OPINIONS
-- derived views, rebuilt by pure functions from the rows, deletable at will.

WHY GIT IS THE EVOLUTIONARY TREE
--------------------------------
A candidate is a commit. Parent-child lineage is git ancestry. That is the whole design,
and it buys four things a bespoke tree store would have to reimplement badly:

  * The exact source that produced any score is recoverable by checkout. Reproducibility
    is not a promise, it is `git checkout <sha>`.
  * Lineage is already durable, distributed and mergeable. Two good candidates can be
    combined with an actual merge, and the merge commit is a real child of both parents
    -- which a single-parent tree cannot express at all.
  * Teammates on different machines contribute to one tree without a coordination
    protocol.
  * Clade metaproductivity (Huxley-Godel Machine) needs "pooled outcomes over a node's
    entire descendant subtree". Under git that is reachability, which we compute once
    from `rev-list --parents` rather than maintaining a parent pointer per row.

THE RULES THIS IMPOSES, which are not optional:

  1. Never record a measurement from a dirty tree without marking it. A sha that does not
     describe the code that ran is worse than no sha: it is a false provenance claim.
     `dirty=True` rows are recorded (they are still evidence) but excluded from clade
     statistics and from promotion.
  2. Never rebase, squash, amend or force-push a candidate branch. Rewriting history
     silently reparents the tree and invalidates every clade statistic derived from it.
     This is the same rule the research archive states for its own events.
  3. A score is per (commit, config). One commit yields up to 14 rows -- one per row of
     the announced matrix -- and an aggregate is a derived view over them, never a
     stored field.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Optional

SCHEMA = 1
DEFAULT_PATH = "bench/results.jsonl"

_REQUIRED = ("ts", "commit_sha", "config_id", "status")


# ======================================================================================
# Git
# ======================================================================================

def _git(*args: str, cwd: Optional[str] = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
    ).stdout.strip()


def head_sha(cwd: Optional[str] = None) -> str:
    return _git("rev-parse", "HEAD", cwd=cwd)


def current_branch(cwd: Optional[str] = None) -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)


def is_dirty(cwd: Optional[str] = None) -> bool:
    """Any staged or unstaged change to tracked files.

    Untracked files are deliberately ignored: scratch output next to the repo does not
    change what the measured code was.
    """
    return bool(_git("status", "--porcelain", "--untracked-files=no", cwd=cwd))


def provenance(cwd: Optional[str] = None) -> dict:
    """The fields every row needs to be traceable back to real source."""
    return {
        "commit_sha": head_sha(cwd),
        "branch": current_branch(cwd),
        "dirty": is_dirty(cwd),
    }


def _ancestry(cwd: Optional[str] = None) -> dict[str, list[str]]:
    """child_sha -> [parent_sha, ...] over every reachable commit, in ONE call.

    Doing this per-pair with `merge-base --is-ancestor` is O(n) subprocesses and gets
    slow the moment the tree is interesting.
    """
    out = _git("rev-list", "--all", "--parents", cwd=cwd)
    tree: dict[str, list[str]] = {}
    for line in out.splitlines():
        parts = line.split()
        if parts:
            tree[parts[0]] = parts[1:]
    return tree


def descendants(sha: str, cwd: Optional[str] = None,
                ancestry: Optional[dict[str, list[str]]] = None) -> set[str]:
    """Every commit reachable FROM sha going forward in time, including sha itself.

    Git stores child -> parents, so we invert once and walk. A merge commit is correctly
    counted as a descendant of both its parents, which is exactly what clade
    metaproductivity wants: a candidate that contributed to a successful merge gets
    credit for it.
    """
    ancestry = ancestry if ancestry is not None else _ancestry(cwd)
    children: dict[str, list[str]] = {}
    for child, parents in ancestry.items():
        for p in parents:
            children.setdefault(p, []).append(child)

    seen = {sha}
    queue = deque([sha])
    while queue:
        node = queue.popleft()
        for child in children.get(node, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


# ======================================================================================
# The ledger
# ======================================================================================

class BenchLedger:
    def __init__(self, path: str | os.PathLike = DEFAULT_PATH,
                 repo: Optional[str] = None):
        self.path = Path(path)
        self.repo = repo
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- writing ------------------------------------------------------------------

    def append(self, row: dict) -> dict:
        """Append one measurement. Flushed and fsync'd so a crash costs at most one row.

        There is deliberately no update(), no delete(), and no open mode other than "a".
        """
        row.setdefault("schema", SCHEMA)
        for field in _REQUIRED:
            if field not in row:
                raise ValueError(f"ledger row missing required field {field!r}")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return row

    def record(self, *, config_id: int, status: str, candidate: str,
               timing: Optional[dict] = None, correctness: Optional[dict] = None,
               memory: Optional[dict] = None, env: Optional[dict] = None,
               config: Optional[dict] = None, notes: str = "",
               ts: Optional[str] = None) -> dict:
        """Build a row with git provenance filled in, then append it.

        `ts` is a parameter rather than a call to now() so that a caller replaying a
        batch of measurements can stamp them with the time they were TAKEN.
        """
        if ts is None:
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).isoformat()
        row = {
            "ts": ts,
            "config_id": config_id,
            "config": config,
            "candidate": candidate,
            "status": status,
            "timing": timing,
            "correctness": correctness,
            "memory": memory,
            "env": env,
            "notes": notes,
            **provenance(self.repo),
        }
        return self.append(row)

    # -- reading ------------------------------------------------------------------

    def rows(self) -> Iterator[dict]:
        """Yield every row. A truncated final line (crash mid-write) is skipped and
        counted rather than raising -- the ledger must always be readable."""
        if not self.path.exists():
            return
        bad = 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
        if bad:
            print(f"[bench-ledger] skipped {bad} unparseable line(s) -- crash mid-write?")

    def clean_rows(self) -> Iterator[dict]:
        """Rows whose commit sha actually describes the code that ran."""
        for r in self.rows():
            if not r.get("dirty", False):
                yield r

    def passing(self) -> Iterator[dict]:
        for r in self.clean_rows():
            if r.get("status") == "ok" and (r.get("correctness") or {}).get("passed"):
                yield r


# ======================================================================================
# Derived views. Pure functions. Delete and rebuild at will.
# ======================================================================================

def best_per_config(ledger: BenchLedger) -> dict[int, dict]:
    """Fastest passing candidate per config id."""
    best: dict[int, dict] = {}
    for r in ledger.passing():
        speedup = (r.get("timing") or {}).get("speedup")
        if speedup is None:
            continue
        cur = best.get(r["config_id"])
        if cur is None or speedup > cur["timing"]["speedup"]:
            best[r["config_id"]] = r
    return best


def scoreboard(ledger: BenchLedger) -> list[dict]:
    """Per-commit aggregate: how many configs measured, how many pass, mean speedup.

    This is the table you sort to answer "which commit is winning", and it is why the
    ledger exists in this shape -- score maps to sha, and sha maps to source.
    """
    from collections import defaultdict
    per: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"measured": 0, "passed": 0, "speedups": {}, "failures": []}
    )
    for r in ledger.clean_rows():
        key = (r["commit_sha"], r.get("candidate") or "")
        agg = per[key]
        agg["measured"] += 1
        ok = r.get("status") == "ok" and (r.get("correctness") or {}).get("passed")
        if ok:
            agg["passed"] += 1
            sp = (r.get("timing") or {}).get("speedup")
            if sp is not None:
                agg["speedups"][r["config_id"]] = sp
        else:
            agg["failures"].append({"config_id": r["config_id"], "status": r.get("status")})

    from bench.matrix import weighted_score
    out = []
    for (sha, cand), agg in per.items():
        out.append({
            "commit_sha": sha,
            "short_sha": sha[:8],
            "candidate": cand,
            "configs_measured": agg["measured"],
            "configs_passed": agg["passed"],
            "weighted_score": weighted_score(agg["speedups"]),
            "speedups": agg["speedups"],
            "failures": agg["failures"],
        })
    out.sort(key=lambda d: d["weighted_score"], reverse=True)
    return out


def clade_stats(ledger: BenchLedger, repo: Optional[str] = None
                ) -> dict[str, tuple[int, int]]:
    """Pooled (successes, failures) over each commit's entire descendant subtree.

    The Huxley-Godel Machine's correction to the Darwin Godel Machine: a node's OWN score
    is a biased estimator of its value as an ancestor. A mediocre candidate that spawns
    good children is a good parent, and ranking by own-performance systematically
    discards those stepping stones.

    A "success" is a row that passed correctness AND beat baseline. Merely compiling is
    not success -- otherwise the loop drifts toward safe, slow, correct code, which is a
    measured failure mode of refinement loops.
    """
    own: dict[str, tuple[int, int]] = {}
    for r in ledger.clean_rows():
        sha = r["commit_sha"]
        s, f = own.get(sha, (0, 0))
        ok = (r.get("status") == "ok"
              and (r.get("correctness") or {}).get("passed")
              and ((r.get("timing") or {}).get("speedup") or 0) > 1.0)
        own[sha] = (s + 1, f) if ok else (s, f + 1)

    if not own:
        return {}

    ancestry = _ancestry(repo)
    pooled: dict[str, tuple[int, int]] = {}
    for sha in own:
        if sha not in ancestry:
            # Measured against a commit this repo no longer knows -- history was
            # rewritten, or the row came from a fork. Report it alone rather than
            # silently attributing it to nothing.
            pooled[sha] = own[sha]
            continue
        s = f = 0
        for d in descendants(sha, repo, ancestry):
            ds, df = own.get(d, (0, 0))
            s, f = s + ds, f + df
        pooled[sha] = (s, f)
    return pooled


def sample_parent(ledger: BenchLedger, repo: Optional[str] = None,
                  rng: Optional[random.Random] = None) -> Optional[str]:
    """Thompson sampling over clade metaproductivity -- returns a commit to branch from.

    Draw from Beta(1+successes, 1+failures) per commit and take the argmax. This explores
    commits with little evidence and exploits commits whose descendants do well, with no
    temperature parameter to tune.
    """
    rng = rng or random.Random()
    stats = clade_stats(ledger, repo)
    if not stats:
        return None
    best_sha, best_draw = None, -1.0
    for sha, (s, f) in stats.items():
        draw = rng.betavariate(1 + s, 1 + f)
        if draw > best_draw:
            best_sha, best_draw = sha, draw
    return best_sha


def failure_corpus(ledger: BenchLedger) -> list[dict]:
    """Every failure. The most valuable training signal after the wins, and the reason
    failures are recorded rather than skipped."""
    return [r for r in ledger.rows()
            if r.get("status") in ("incorrect", "oom", "compile_error", "timeout", "crash")]


if __name__ == "__main__":
    import sys

    led = BenchLedger()
    rows = list(led.rows())
    print(f"{len(rows)} row(s) in {led.path}")
    if not rows:
        print("no measurements recorded yet")
        sys.exit(0)

    print("\n=== scoreboard (score -> commit) ===")
    print(f"{'sha':<10} {'candidate':<28} {'cfgs':>5} {'pass':>5} {'score':>7}")
    for e in scoreboard(led):
        print(f"{e['short_sha']:<10} {e['candidate'][:28]:<28} "
              f"{e['configs_measured']:>5} {e['configs_passed']:>5} "
              f"{e['weighted_score']:>7.3f}")

    print("\n=== clade metaproductivity (pooled over descendants) ===")
    for sha, (s, f) in sorted(clade_stats(led).items(), key=lambda kv: -kv[1][0]):
        print(f"{sha[:10]}  successes={s:<4} failures={f}")
