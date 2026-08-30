"""Reduce the g42 A/B runs to the one number the candidate is allowed to claim.

THE ESTIMATOR IS THE PER-ARM MINIMUM ACROSS INDEPENDENT RUNS, and that is a decision
this file has to argue for rather than assume.

Config 2's readings are not a spread around a mean. Across eight independent runs the
control arm reads 48.13, 48.13, 69.63, 200.70 -- a floor that recurs exactly, and
excursions above it of up to 4.17x. Contamination on this harness is ONE-SIDED: a
descheduled host thread, a settling graph or a co-resident allocation can only make a
reading slower, never faster. So the mean is a statistic about how often the machine
misbehaved, and the minimum is a statistic about the code. CLAUDE.md prescribes exactly
this for a card whose clocks will not lock ("minimum-of-N timing"), and finding 49
established that two candidates reading 46% apart matched to the hundredth of a
microsecond once replicated.

Both arms get the same number of runs and the same reduction, so the winner's curse is
symmetric (finding 47's handicap applies to a min over ARMS of one sweep, not to a min
over RUNS of one arm; and here N is equal on both sides).

The per-run ratio is printed too, because a reader should be able to see how much of the
spread the estimator is discarding.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bench.matrix import BY_ID, MATRIX, weighted_score

QUANTUM_US = 1.024          # the CUDA event timer's resolution on this card

# Config-2 speedup against the compiled baseline for the parent, from REPORT.md 4.2's
# v40 column (v41 is byte-identical to v40 on every announced config -- finding 51).
PARENT_SPEEDUPS = {1: 2.633, 2: 2.631, 3: 4.388, 4: 2.675, 5: 2.923, 6: 3.746,
                   7: 4.302, 8: 2.200, 9: 2.022, 10: 2.415, 11: 8.464, 12: 2.653,
                   13: 10.898}


def load(pattern: str) -> dict[int, list[dict]]:
    by_cfg: dict[int, list[dict]] = {}
    for f in sorted(glob.glob(pattern)):
        for row in json.load(open(f)):
            by_cfg.setdefault(row["config_id"], []).append({"file": Path(f).name, **row})
    return by_cfg


def main() -> int:
    runs: dict[int, list[dict]] = {}
    for pat in ("rep*.json", "c3rep*.json", "abba_run*.json"):
        for c, rows in load(str(REPO / "bench/probes/g42_tile_timer" / pat)).items():
            runs.setdefault(c, []).extend(rows)

    print("PER-RUN READINGS (median of the kept rounds, microseconds)\n")
    floors = {}
    for c in sorted(runs):
        rows = runs[c]
        a, b = rows[0]["arms"]
        va = [r["median_of_min_rounds"][a] * 1e3 for r in rows]
        vb = [r["median_of_min_rounds"][b] * 1e3 for r in rows]
        tiles_a = {tuple(r["correctness"][a].get("attn_tile") or ()) for r in rows
                   if r["correctness"][a].get("attn_tile") is not None}
        tiles_b = {tuple(r["correctness"][b].get("attn_tile") or ()) for r in rows
                   if r["correctness"][b].get("attn_tile") is not None}
        floors[c] = (min(va), min(vb))
        print(f"cfg {c}   n={len(rows)} runs")
        print(f"   {a:<26} " + "  ".join(f"{v:8.2f}" for v in va))
        print(f"   {b:<26} " + "  ".join(f"{v:8.2f}" for v in vb))
        print(f"   per-run ratio              " +
              "  ".join(f"{x/y:8.4f}" for x, y in zip(va, vb)))
        print(f"   floors  {min(va):.2f} / {min(vb):.2f}  ->  {min(va)/min(vb):.4f}x"
              f"   ({(min(va)-min(vb))/QUANTUM_US:+.2f} quanta)")
        print(f"   spread above floor: {a} {max(va)/min(va):.2f}x, "
              f"{b} {max(vb)/min(vb):.2f}x")
        print(f"   tiles: {a} {tiles_a or '-'}   {b} {tiles_b or '-'}")
        print()

    print("WHAT IT IS WORTH, if the floors are the code and the excursions are the "
          "machine\n")
    moved = {c: floors[c][0] / floors[c][1] for c in floors}
    new = dict(PARENT_SPEEDUPS)
    for c, r in moved.items():
        if c in new:
            new[c] = PARENT_SPEEDUPS[c] * r
    base = weighted_score(PARENT_SPEEDUPS)
    after = weighted_score(new)
    for c in sorted(moved):
        if c not in PARENT_SPEEDUPS:
            continue
        cap = " (CAPPED, scores nothing)" if PARENT_SPEEDUPS[c] >= 3.0 else ""
        print(f"   cfg {c:>2}  {PARENT_SPEEDUPS[c]:6.3f} -> {new[c]:6.3f} "
              f"({moved[c]:.4f}x){cap}")
    print(f"\n   weighted_score  {base:.4f} -> {after:.4f}   "
          f"delta {after - base:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
