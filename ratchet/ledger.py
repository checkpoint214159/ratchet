"""The ledger. Append-only.

ZONE C. Measurements are facts and are never deleted, edited, sorted in place or pruned.
Rankings, best-known tables and critic scores are OPINIONS: derived views, rebuilt by
pure functions, deletable at will.

That distinction is the design's load-bearing departure from the Red Queen Godel Machine.
RQGM erases utility records when it swaps an evaluator, which costs an LLM call. Here it
would cost a GPU benchmark run -- the dominant cost of the whole system -- so instead the
erasable evaluator is the CRITIC'S PREDICTIONS, and hardware measurements survive every
epoch transition untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

SCHEMA = 1
DEFAULT_PATH = "ledger/measurements.jsonl"


def candidate_id(source: str) -> str:
    """Content address a candidate by its normalized source.

    Normalization is whitespace-only on purpose: two kernels that differ by a comment are
    the same kernel, but two that differ by a constant are not.
    """
    norm = "\n".join(line.rstrip() for line in source.strip().splitlines())
    return hashlib.sha256(norm.encode()).hexdigest()[:8]


class Ledger:
    def __init__(self, path: str | os.PathLike = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- writing -----------------------------------------------------------------------

    def append(self, row: dict) -> None:
        """Append one measurement. Flushed and fsync'd so a crash costs at most one row.

        There is deliberately no update(), no delete(), and no open mode other than "a".
        scripts/check-oracle.sh greps for violations of this.
        """
        row.setdefault("schema", SCHEMA)
        for required in ("ts", "candidate_id", "status"):
            if required not in row:
                raise ValueError(f"ledger row missing required field {required!r}")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # -- reading -----------------------------------------------------------------------

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
            print(f"[ledger] skipped {bad} unparseable line(s) -- likely a crash mid-write")

    def ok_rows(self) -> Iterator[dict]:
        for r in self.rows():
            if r.get("status") == "ok" and r.get("correctness", {}).get("passed"):
                yield r


# ======================================================================================
# Derived views. Pure functions of the ledger. Delete and rebuild at will.
# ======================================================================================

def _shape_key(row: dict) -> str:
    s = row.get("shape", {})
    return "|".join(f"{k}={s[k]}" for k in sorted(s))


def _env_key(row: dict) -> str:
    e = row.get("env", {})
    return "|".join(str(e.get(k, "")) for k in ("device_name", "torch", "triton"))


def best_known(ledger: Ledger) -> dict[tuple[str, str], dict]:
    """Fastest passing candidate per (shape, environment)."""
    best: dict[tuple[str, str], dict] = {}
    for r in ledger.ok_rows():
        key = (_shape_key(r), _env_key(r))
        cur = best.get(key)
        t = r.get("timing", {}).get("mean_ns")
        if t is None:
            continue
        if cur is None or t < cur["timing"]["mean_ns"]:
            best[key] = r
    return best


def _ci_overlap(a: dict, b: dict, k: float = 2.0) -> bool:
    ta, tb = a.get("timing", {}), b.get("timing", {})
    ma, sa = ta.get("mean_ns"), ta.get("sem_ns", 0.0)
    mb, sb = tb.get("mean_ns"), tb.get("sem_ns", 0.0)
    if ma is None or mb is None:
        return True
    return not ((ma + k * sa) < (mb - k * sb) or (mb + k * sb) < (ma - k * sa))


def promotion_candidates(ledger: Ledger,
                         deployed: Optional[dict[tuple[str, str], str]] = None,
                         k_sigma: float = 2.0) -> list[dict]:
    """Which shapes have a challenger that beats what is currently deployed, decisively?

    `deployed` maps (shape_key, env_key) -> candidate_id of whatever the dispatch table
    currently selects. Pass the real table. If omitted, the runner-up is treated as the
    incumbent, which answers the weaker question "is the best-known meaningfully ahead of
    the next best, or is the ranking noise?"

    A challenger is returned only when it is faster AND its confidence interval does not
    overlap the incumbent's. A 3% win with overlapping error bars is noise, and promoting
    on it is how a search loop convinces itself it is improving while random-walking.
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in ledger.ok_rows():
        if r.get("timing", {}).get("mean_ns") is None:
            continue
        by_key.setdefault((_shape_key(r), _env_key(r)), []).append(r)

    out: list[dict] = []
    for key, rows in by_key.items():
        rows.sort(key=lambda r: r["timing"]["mean_ns"])
        best = rows[0]

        if deployed is not None:
            inc_cid = deployed.get(key)
            incumbent = next((r for r in rows if r["candidate_id"] == inc_cid), None)
        else:
            incumbent = next((r for r in rows[1:]
                              if r["candidate_id"] != best["candidate_id"]), None)

        if incumbent is None or incumbent["candidate_id"] == best["candidate_id"]:
            out.append({"shape_key": key[0], "env_key": key[1], "challenger": best,
                        "incumbent": None, "decisive": True,
                        "reason": "no incumbent to beat"})
            continue

        decisive = (best["timing"]["mean_ns"] < incumbent["timing"]["mean_ns"]
                    and not _ci_overlap(best, incumbent, k_sigma))
        out.append({
            "shape_key": key[0], "env_key": key[1],
            "challenger": best, "incumbent": incumbent, "decisive": decisive,
            "speedup": incumbent["timing"]["mean_ns"] / best["timing"]["mean_ns"],
            "reason": "non-overlapping CI" if decisive else "CIs overlap -- do not promote",
        })
    return out


def clade_stats(ledger: Ledger) -> dict[str, tuple[int, int]]:
    """Pooled (successes, failures) over each candidate's entire descendant subtree.

    Huxley-Godel Machine's correction to Darwin Godel Machine: a node's OWN score is a
    biased estimator of its value as an ancestor. A mediocre kernel that spawns good
    children is a good parent, and scoring by own-performance systematically discards
    those stepping stones.
    """
    children: dict[str, list[str]] = {}
    own: dict[str, tuple[int, int]] = {}

    for r in ledger.rows():
        cid = r["candidate_id"]
        pid = r.get("parent_id")
        if pid:
            children.setdefault(pid, []).append(cid)
        s, f = own.get(cid, (0, 0))
        if r.get("status") == "ok" and r.get("correctness", {}).get("passed"):
            own[cid] = (s + 1, f)
        else:
            own[cid] = (s, f + 1)

    memo: dict[str, tuple[int, int]] = {}

    def walk(cid: str, seen: frozenset) -> tuple[int, int]:
        if cid in memo:
            return memo[cid]
        if cid in seen:            # cycles should be impossible, but do not hang on one
            return (0, 0)
        s, f = own.get(cid, (0, 0))
        for ch in children.get(cid, []):
            cs, cf = walk(ch, seen | {cid})
            s, f = s + cs, f + cf
        memo[cid] = (s, f)
        return memo[cid]

    return {cid: walk(cid, frozenset()) for cid in own}


def sample_parent(ledger: Ledger, rng: Optional[random.Random] = None) -> Optional[str]:
    """Thompson sampling over clade metaproductivity.

    Draw from Beta(1+successes, 1+failures) per candidate and take the argmax. This
    explores nodes with little evidence and exploits nodes whose descendants do well,
    without ever needing a temperature parameter.
    """
    rng = rng or random.Random()
    stats = clade_stats(ledger)
    if not stats:
        return None
    best_cid, best_draw = None, -1.0
    for cid, (s, f) in stats.items():
        draw = rng.betavariate(1 + s, 1 + f)
        if draw > best_draw:
            best_cid, best_draw = cid, draw
    return best_cid


def failure_corpus(ledger: Ledger) -> list[dict]:
    """Every compile error and correctness failure.

    The critic's most valuable training signal, and the reason failures are recorded
    rather than skipped: in comparable tuning spaces 68-78% of configurations fail to
    compile, so the failures ARE the dataset.
    """
    return [r for r in ledger.rows()
            if r.get("status") in ("compile_error", "incorrect", "timeout", "crash", "oom")]


def critic_training_split(ledger: Ledger, holdout_frac: float = 0.25,
                          seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Split BY CANDIDATE, never by row.

    A candidate's measurements across shapes are not independent samples. Splitting by row
    leaks a candidate's own outcomes into both sides and makes the critic look far better
    than it is -- which then makes the promotion rule meaningless, which removes the only
    protection against a lenient critic. Splitting on the candidate is the whole point.
    """
    rows = list(ledger.rows())
    cids = sorted({r["candidate_id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(cids)
    n_hold = max(1, int(len(cids) * holdout_frac))
    held = set(cids[:n_hold])
    return ([r for r in rows if r["candidate_id"] not in held],
            [r for r in rows if r["candidate_id"] in held])


def beta_lower_bound(successes: int, failures: int, eps: float = 0.1) -> float:
    """eps-quantile of Beta(1+s, 1+f): a conservative lower bound on an accuracy.

    Used for critic promotion. A challenger must beat the incumbent's lower bound, so it
    has to win with statistical margin rather than by lucky sampling. Ties favour the
    incumbent, which damps churn.

    Implemented by bisection on the regularized incomplete beta to avoid a scipy
    dependency in the oracle path; swap for scipy.stats.beta.ppf if it is already present.
    """
    a, b = 1.0 + successes, 1.0 + failures

    def betacdf(x: float) -> float:
        if x <= 0.0:
            return 0.0
        if x >= 1.0:
            return 1.0
        # numerical integration is plenty for a promotion gate
        n = 2000
        h = x / n
        total = 0.0
        for i in range(n + 1):
            t = min(max(i * h, 1e-12), 1 - 1e-12)
            w = 1.0 if 0 < i < n else 0.5
            total += w * math.exp((a - 1) * math.log(t) + (b - 1) * math.log(1 - t))
        total *= h
        norm = math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
        return min(1.0, total / norm)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if betacdf(mid) < eps:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
