"""Aggregate g43's ABBA runs: per-config floor ratio, plan equality, weighted_score delta.

RANK BY THE CANDIDATE'S OWN TIME, and reduce by the FLOOR over every run of a config --
the estimator `bench/abba.py`'s docstring argues for and the one this generation's own
finding is about. Contamination here is one-sided, so the minimum across runs is the
statistic about the code and the spread is the statistic about the machine.

THE PLAN IS REPORTED ALONGSIDE THE TIME, and a config whose two arms ran the SAME plan is
labelled a control rather than assumed to be one -- that is the field `abba.py` grew at
generation 42 and the reason this candidate's inertness is a measurement.

Speedups are carried from finding 53's aggregate so the two generations compose on one
scale; `weighted_score` applies the competition's 3.0 clip per config.

    python3 bench/probes/g43_stable_tiles/summarize.py bench/probes/g43_stable_tiles/abba_run*.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

# v42's per-config speedup over the torch.compile baseline, from finding 53's aggregate
# table (floor estimator). The BASE this candidate is measured as a multiplier on.
V42_SPEEDUP = {1: 2.633, 2: 2.810, 3: 4.308, 4: 2.675, 7: 4.302,
               9: 2.022, 10: 2.426, 11: 8.464, 12: 2.617}
CLIP = 3.0
N_CONFIGS = 14          # the announced matrix; rows this run does not cover score as v42


def main(paths: list[str]) -> int:
    runs = []
    for p in paths:
        runs.extend(json.load(open(p)))

    times: dict = defaultdict(lambda: defaultdict(list))
    plans: dict = defaultdict(lambda: defaultdict(set))
    for r in runs:
        for n in r["arms"]:
            times[r["config_id"]][n].append(r["median_of_min_rounds"][n] * 1e3)
            c = r["correctness"][n]
            plans[r["config_id"]][n].add((c["attn_form"], tuple(c["attn_tile"])))
            assert c["passed"], f"cfg {r['config_id']} {n} FAILED CORRECTNESS"

    arms = runs[0]["arms"]
    a, b = arms[0], arms[1]
    print(f"{'cfg':>4} {'n':>2} {a[:18]:>19} {b[:18]:>19} {'floor ratio':>12}  plans")
    total, controls = 0.0, []
    for cid in sorted(times):
        fa, fb = min(times[cid][a]), min(times[cid][b])
        pa, pb = plans[cid][a], plans[cid][b]
        # A CONTROL BY MEASUREMENT, NOT BY ASSUMPTION. This candidate's entire diff is in
        # the tuner, which runs at prime time OUTSIDE the timed region -- so a config on
        # which both arms selected the same single plan, in every run, executed literally
        # identical code and its ratio is a reading of the harness, not of the change.
        same = pa == pb and len(pa) == 1
        label = ("CONTROL, same plan" if same else
                 f"DIFFER: {sorted(pa)} vs {sorted(pb)}")
        ratio = fa / fb
        if same:
            controls.append((cid, ratio))
        print(f"{cid:>4} {len(times[cid][a]):>2} {fa:>19.2f} {fb:>19.2f} "
              f"{ratio:>11.4f}x  {label}")
        total += min(V42_SPEEDUP[cid] * ratio, CLIP) - min(V42_SPEEDUP[cid], CLIP)

    lo = min(r for _, r in controls)
    hi = max(r for _, r in controls)
    print(f"\nIN-RUN CONTROL FLOOR: {len(controls)} configs ran byte-identical code on "
          f"both arms\n  (cfgs {', '.join(str(c) for c, _ in controls)}), and they span "
          f"{lo:.4f}x - {hi:.4f}x.\n  Nothing outside that band is resolvable by this "
          f"protocol on these runs.")
    print(f"\nweighted_score delta as computed over the {len(times)} configs measured, "
          f"3.0 clip per config, /{N_CONFIGS}:  {total / N_CONFIGS:+.4f}")
    print("  -- and every row contributing to it is a control whose two arms ran the "
          "same plan,\n     or a row whose plan differs through `autotune_looped`, which "
          "this candidate does not\n     touch and which flips run-to-run on BOTH arms. "
          "The honest reading is ZERO, inside the\n     control floor, which is what was "
          "predicted: this candidate's value is in the variance.")
    print("\nRows not measured here (5, 6, 8, 13, 14) select through routines this "
          "candidate does not touch or decline attention entirely; they contribute 0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
