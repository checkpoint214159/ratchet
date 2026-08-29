"""Stage 1 of a two-stage evaluation: kill a candidate cheaply, or promote it.

WHY THIS EXISTS
---------------
One GPU, and a +/-7% noise floor (L29). Resolving a real 5% win needs repeats, so a
confident verdict costs 15-20 minutes of GPU. Five expander agents produce candidates
faster than that, and the queue -- not ideation -- becomes the bottleneck. Most ideas are
wrong; they should die for 20 seconds, not 20 minutes.

    screen  ->  bench/screen.py    20s   4 configs, one pass, verdict only
    confirm ->  bench/run_matrix.py 112s  all 13, recorded to the ledger

THE SCREEN SET WAS DERIVED, NOT CHOSEN
--------------------------------------
Measured from the 411 rows already in the ledger, costing no GPU time (L32: measure the
choice). Per-config wall time came from consecutive ledger timestamps within a sweep:

    cfg 6 alone is 48.5s of the 112s full sweep, so ANY subset containing it is ~50% of
    a full sweep and is not a screen at all.

Every subset tested retains the true top-1 candidate. None retains the true top-3 -- but
the top five candidates span 2.712x to 2.605x, a 4% spread INSIDE the noise floor, so
nothing separates them and a screen should not pretend to. The screen's job is to kill
what is clearly bad, not to rank what is statistically tied.

SCREEN_IDS covers four distinct regimes so a candidate that wins one and destroys another
is caught:

    2   launch-bound   B=1; CPU dispatch dominates (232us CPU vs 126us GPU)
    7   head_dim = 8   where vendor fast paths may refuse -- our identified blind spot
    8   wide model     d_model 1024, head_dim 256
    10  mainstream     the middle of the grid

SCREEN RESULTS DO NOT ENTER THE LEDGER. They are a partial sweep, and letting partial
sweeps into clade statistics would swamp the full ones. They go to bench/screen_log.jsonl,
which is advisory and never feeds sampling.

    python3 bench/screen.py --candidate v15_something --parent v9b_reduce_overhead
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.ledger import (BenchLedger, BASELINE, BASELINE_COMPILED,   # noqa: E402
                          compiled_baseline_ms, provenance)

SCREEN_IDS = (2, 7, 8, 10)
SCREEN_LOG = REPO / "bench" / "screen_log.jsonl"

# A screen is ONE pass, so it cannot resolve anything inside the noise floor. Promote on
# "not clearly worse than the parent" rather than on "better" -- a screen that demands an
# improvement it lacks the resolution to see would reject good candidates at random.
NOISE = 0.07


def parent_screen_geomean(ledger: BenchLedger, parent: str) -> float | None:
    """The parent's speedup over the compiled baseline, restricted to the screen configs."""
    compiled = compiled_baseline_ms(ledger)
    best: dict[int, float] = {}
    for r in ledger.clean_rows():
        if r.get("candidate") != parent or r.get("status") != "ok":
            continue
        if "padding_ratio=0.0" not in (r.get("notes") or ""):
            continue
        ms = (r.get("timing") or {}).get("candidate_ms")
        if ms and r["config_id"] in SCREEN_IDS:
            cid = r["config_id"]
            if cid not in best or ms < best[cid]:
                best[cid] = ms
    vals = [compiled[i] / best[i] for i in SCREEN_IDS if i in best and i in compiled]
    if len(vals) != len(SCREEN_IDS):
        return None
    return math.exp(sum(map(math.log, vals)) / len(vals))


def run_screen(candidate: str, allow_dirty: bool = False) -> dict:
    cmd = [sys.executable, str(REPO / "bench" / "run_matrix.py"),
           "--candidate", candidate, "--ids", *map(str, SCREEN_IDS),
           "--dry-run", "--json-out"]
    if allow_dirty:
        cmd.append("--allow-dirty")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), timeout=1800)

    rows, speedups, failures = [], {}, []
    for line in proc.stdout.splitlines():
        if not line.startswith("__ROW__"):
            continue
        r = json.loads(line[len("__ROW__"):])
        rows.append(r)
        ms = (r.get("timing") or {}).get("candidate_ms")
        correct = (r.get("correctness") or {}).get("passed")
        if r.get("status") == "ok" and correct and ms:
            speedups[r["config_id"]] = ms
        else:
            failures.append({"config_id": r.get("config_id"),
                             "status": r.get("status"),
                             "correct": correct})
    return {"rows": rows, "ms": speedups, "failures": failures,
            "stderr": proc.stderr[-2000:] if proc.returncode else ""}


def decide(ms: dict[int, float], failures: list, compiled: dict[int, float],
           parent_geo: float | None) -> tuple[str, float | None, str]:
    """(verdict, screen_geomean, detail). Pure, so the policy is testable without a GPU."""
    # CORRECTNESS IS NOT A TIEBREAK. Any failure on any screen config is a hard reject,
    # before a single timing number is looked at (CLAUDE.md rule 3).
    if failures:
        return "REJECT", None, f"correctness/run failures: {failures}"
    vals = [compiled[i] / ms[i] for i in SCREEN_IDS if i in ms and i in compiled]
    if len(vals) != len(SCREEN_IDS):
        return "REJECT", None, "incomplete screen"
    geo = math.exp(sum(map(math.log, vals)) / len(vals))
    if parent_geo is None:
        return "PROMOTE", geo, f"screen geomean {geo:.3f}x vs compiled; no parent baseline"
    delta = geo / parent_geo - 1
    if geo >= parent_geo * (1 - NOISE):
        return "PROMOTE", geo, (f"screen {geo:.3f}x vs parent {parent_geo:.3f}x "
                                f"({delta:+.1%}) -- within or above noise")
    return "REJECT", geo, (f"screen {geo:.3f}x vs parent {parent_geo:.3f}x "
                           f"({delta:+.1%}) -- clearly worse than the parent")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--parent", help="promote only if not clearly worse than this")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    led = BenchLedger()
    compiled = compiled_baseline_ms(led)
    res = run_screen(args.candidate, args.allow_dirty)

    par = parent_screen_geomean(led, args.parent) if args.parent else None
    verdict, geo, detail = decide(res["ms"], res["failures"], compiled, par)

    prov = provenance()
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "candidate": args.candidate,
           "parent": args.parent, "screen_ids": list(SCREEN_IDS), "verdict": verdict,
           "screen_geomean_vs_compiled": geo, "detail": detail,
           "commit_sha": prov["commit_sha"], "dirty": prov["dirty"],
           "per_config_ms": res["ms"], "failures": res["failures"]}
    with SCREEN_LOG.open("a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")

    print(f"\n  screen configs : {SCREEN_IDS}")
    print(f"  candidate      : {args.candidate}")
    print(f"  parent         : {args.parent}")
    print(f"  VERDICT        : {verdict}")
    print(f"  {detail}")
    if res["stderr"]:
        print(f"\n  stderr tail:\n{res['stderr']}")
    return 0 if verdict == "PROMOTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
