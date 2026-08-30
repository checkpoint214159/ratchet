"""Diff two candidates' kernel profiles from the ledger.

The question this answers is the one the ledger could not: not "which is faster" but
"WHERE did the time move, and did the mechanism we claimed actually run".

    python3 bench/profile_diff.py v26_causal_correct v34_launch_bound --config 2

Reads recorded profiles only -- it never touches the GPU, so it is safe to run while
agents are measuring (finding 26: two processes on one GPU produce two wrong numbers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.ledger import BenchLedger  # noqa: E402


def latest_profile(ledger: BenchLedger, candidate: str, config_id: int) -> dict | None:
    best = None
    for r in ledger.clean_rows():
        if r.get("candidate") != candidate or r.get("config_id") != config_id:
            continue
        prof = (r.get("profile") or {}).get("candidate")
        if prof and not prof.get("error"):
            if best is None or r["ts"] > best[0]:
                best = (r["ts"], prof, r)
    return None if best is None else {"ts": best[0], **best[1], "_row": best[2]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--config", type=int, required=True)
    args = ap.parse_args()

    led = BenchLedger()
    pa = latest_profile(led, args.a, args.config)
    pb = latest_profile(led, args.b, args.config)
    for name, p in ((args.a, pa), (args.b, pb)):
        if p is None:
            print(f"no recorded profile for {name} at config {args.config}. Rows measured "
                  f"before profiling was added carry none; re-sweep to populate it.")
            return 2

    print(f"config {args.config}:  {args.a}  ->  {args.b}\n")
    for label, key in (("launches", "launches"), ("distinct kernels", "distinct_kernels"),
                       ("Memcpy launches", "memcpy_launches")):
        va, vb = pa.get(key, 0), pb.get(key, 0)
        print(f"  {label:<20}{va:>6} -> {vb:<6}  {vb - va:+d}")
    # profiled_ms is NOT the measurement; the ledger's timing is.
    ta = ((pa["_row"].get("timing") or {}).get("candidate_ms"))
    tb = ((pb["_row"].get("timing") or {}).get("candidate_ms"))
    if ta and tb:
        print(f"  {'measured ms':<20}{ta:>6.3f} -> {tb:<6.3f}  {100*(tb/ta-1):+.1f}%")

    ka = {k["name"]: k for k in pa["kernels"]}
    kb = {k["name"]: k for k in pb["kernels"]}
    print(f"\n  {'kernel':<52}{'A us':>9}{'B us':>9}{'delta':>10}")
    for name in sorted(set(ka) | set(kb),
                       key=lambda n: -(kb.get(n, {}).get("us", 0) + ka.get(n, {}).get("us", 0))):
        ua = ka.get(name, {}).get("us", 0.0)
        ub = kb.get(name, {}).get("us", 0.0)
        tag = "  GONE" if ub == 0 else ("  NEW" if ua == 0 else "")
        print(f"  {name[:50]:<52}{ua:>9.1f}{ub:>9.1f}{ub - ua:>+10.1f}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
