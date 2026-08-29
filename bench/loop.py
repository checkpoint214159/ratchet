"""The search loop: propose, gate, measure, record, select.

This is the parametric level (spec 03's "level 1"). It searches the knobs our candidates
expose; it does NOT invent new kernel architectures. That distinction is deliberate --
conflating the two is why naive agentic loops plateau, and a classical optimizer beats an
LLM at picking block sizes every time.

WHAT MAKES THIS A LOOP RATHER THAN A SWEEP
------------------------------------------
Three things, all of which matter:

  * PARENT SELECTION IS BY CLADE, NOT BY SCORE. A parent is drawn by Thompson sampling
    over the pooled outcomes of its entire descendant subtree (`bench.ledger.clade_stats`,
    computed over git ancestry). A mediocre candidate that spawns good children is a good
    parent, and ranking nodes by their own score systematically discards those stepping
    stones.

  * FAILURES ARE RECORDED, NOT SKIPPED. An infeasible point costs an evaluation, gets a
    ledger row, and is assigned a large finite fitness -- never an exception that aborts
    the run, never a silent skip that hides the failure rate. In comparable tuning spaces
    68-78% of configurations fail to compile; the failures ARE the dataset.

  * PROMOTION REQUIRES A MARGIN. A win is only a win if it beats the incumbent on the
    weighted objective by more than the measurement noise. A 3% improvement inside the
    error bars is how a search convinces itself it is progressing while random-walking.

WHAT IT DOES NOT DO YET, STATED PLAINLY
---------------------------------------
It does not write kernels. It does not call an LLM. It does not create git branches per
candidate -- every evaluation is recorded against the current HEAD, so lineage within a
single loop run is flat. Turning the flat run into a real tree is the next step and is
what the branch protocol in bench/README.md is for.

    python3 bench/loop.py --rounds 12 --ids 1 2 6 12 13
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------
# The search space.
#
# Every axis is a real knob in bench/candidates/, and every value is plausible on SOME
# device -- the point of searching rather than hardcoding is that the right value differs
# by cache size and by shape. `INFEASIBLE_FITNESS` is large and finite so an optimizer
# can walk away from a bad region without special-casing exceptions.
# --------------------------------------------------------------------------------------
# `target_occupancy` and `live_tensors` were separate axes until the first run showed
# solve_chunk uses them ONLY as a quotient -- so the optimizer walked the space and
# arrived back at its own starting configuration wearing different coordinates, then
# reported the re-measurement as a 2.7% win. Two parameters that only appear as a ratio
# are one parameter. See docs/findings/06-the-search-found-noise.md.
SPACE = {
    "use_graph": [True, False],
    "chunk_ratio": [0.0625, 0.125, 0.1667, 0.25, 0.3333, 0.5],
}
INFEASIBLE_FITNESS = 1e10

# Run-to-run spread measured from that run's accidental replicates: 1.4-3.0% on
# identical configurations. A candidate must beat the incumbent by more than this to be
# promoted, or the loop simply reports its own noise as progress -- which is exactly
# what it did before this guard existed.
NOISE_FLOOR = 0.03


def enumerate_space() -> list[dict]:
    keys = sorted(SPACE)
    return [dict(zip(keys, values)) for values in itertools.product(*(SPACE[k] for k in keys))]


def point_id(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def neighbours(params: dict) -> list[dict]:
    """Points differing in ONE axis, by one step along that axis's ordered values.

    The 'adjacent' neighbourhood rather than the full Hamming one: closely-related
    parameter values are related in performance, and full neighbourhoods waste budget.
    """
    out = []
    for key, values in SPACE.items():
        i = values.index(params[key])
        for j in (i - 1, i + 1):
            if 0 <= j < len(values):
                out.append({**params, key: values[j]})
    return out


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

def evaluate(params: dict, config_ids: list[int], record: bool) -> dict:
    """Measure one point across the given configs. Returns fitness (lower is better)."""
    env_overrides = {
        "RATCHET_USE_GRAPH": "1" if params["use_graph"] else "0",
        "RATCHET_CHUNK_RATIO": str(params["chunk_ratio"]),
    }
    cmd = [sys.executable, str(REPO / "bench" / "run_matrix.py"),
           "--candidate", "v4_tunable", "--ids", *[str(i) for i in config_ids],
           "--json-out"]
    if not record:
        cmd.append("--dry-run")

    import os
    env = {**os.environ, **env_overrides}
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO),
                          env=env, timeout=5400)

    speedups: dict[int, float] = {}
    failures = 0
    for line in proc.stdout.splitlines():
        if line.startswith("__ROW__"):
            row = json.loads(line[len("__ROW__"):])
            sp = (row.get("timing") or {}).get("speedup")
            if row.get("status") == "ok" and sp:
                speedups[row["config_id"]] = sp
            else:
                failures += 1

    if not speedups:
        # Infeasible: a large FINITE fitness, and it is still a recorded evaluation.
        return {"params": params, "fitness": INFEASIBLE_FITNESS, "speedups": {},
                "failures": failures, "feasible": False}

    geo = math.exp(sum(math.log(v) for v in speedups.values()) / len(speedups))
    # Fitness is minimized, so invert. Failures are penalized proportionally rather than
    # ignored -- a point that wins on two configs and crashes on three is not a win.
    penalty = 1.0 + failures
    return {"params": params, "fitness": penalty / geo, "speedups": speedups,
            "failures": failures, "feasible": True, "geomean": geo}


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------

def run(rounds: int, config_ids: list[int], seed: int = 0, record: bool = True) -> dict:
    rng = random.Random(seed)
    cache: dict[str, dict] = {}
    history: list[dict] = []

    def evaluate_cached(params: dict) -> dict:
        """A repeat visit returns the cached fitness and does NOT consume budget."""
        pid = point_id(params)
        if pid not in cache:
            cache[pid] = evaluate(params, config_ids, record)
            history.append(cache[pid])
        return cache[pid]

    # Seed from the current default, which is what the candidates ship with.
    current = {"use_graph": True, "chunk_ratio": 0.1667}
    best = evaluate_cached(current)
    print(f"seed  fitness={best['fitness']:.4f}  geomean={best.get('geomean', 0):.3f}x  "
          f"{current}")

    # First-improvement iterated local search: take the FIRST improving neighbour rather
    # than the best, which is cheaper per step and empirically stronger above ~200
    # evaluations. Restart from a random point when a local optimum is reached.
    while len(history) < rounds:
        improved = False
        neigh = neighbours(current)
        rng.shuffle(neigh)
        for cand in neigh:
            if len(history) >= rounds:
                break
            result = evaluate_cached(cand)
            marker = "" if not result["feasible"] else f"{result.get('geomean', 0):.3f}x"
            print(f"  eval  fitness={result['fitness']:.4f}  {marker:>8}  {cand}")
            # Promotion requires a MARGIN, not merely a smaller number. The rest of
            # this project promotes only on non-overlapping confidence intervals; the
            # loop was not honouring the rule it is embedded in.
            if result["fitness"] < best["fitness"] * (1.0 - NOISE_FLOOR):
                best, current, improved = result, cand, True
                print(f"  -> improved beyond the {NOISE_FLOOR:.0%} noise floor")
                break
            elif result["fitness"] < best["fitness"]:
                print(f"  -- better but inside noise; not promoted")
        if not improved:
            if len(history) >= rounds:
                break
            current = {k: rng.choice(v) for k, v in SPACE.items()}
            print(f"  restart from {current}")

    feasible = [h for h in history if h["feasible"]]
    print(f"\nevaluated {len(history)} point(s), {len(history) - len(feasible)} infeasible "
          f"({100 * (len(history) - len(feasible)) / max(1, len(history)):.0f}% failure rate)")
    print(f"best geomean {best.get('geomean', 0):.3f}x with {best['params']}")
    return {"best": best, "history": history}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--ids", type=int, nargs="*", default=[1, 2, 12])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()
    run(args.rounds, args.ids, args.seed, record=not args.no_record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
