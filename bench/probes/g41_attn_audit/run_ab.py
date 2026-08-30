"""Hold the GPU lock for a whole A/B, then drive `bench/abba.py` under it.

`bench/abba.py` says plainly that "the GPU lock is the CALLER's job: this file measures,
it does not arbitrate". This is that caller, so a long interleaved run cannot be
descheduled halfway through by another agent's probe (finding 26, and finding 05's 4.1x).

Two modes, because the two halves of the matrix need different protocols (finding 50):

    --mode abba      all arms resident, ABBA-interleaved, cold round discarded.
                     CORRECT on the sub-millisecond configs and WRONG on the large ones,
                     where three resident arms hit finding 05's co-residency spill.
    --mode isolated  one arm per process, replicated, compared across runs. The other
                     way round: correct on the large configs, drifts on the small ones.

    python3 bench/probes/g41_attn_audit/run_ab.py --mode abba \
        --ids 1 2 3 4 7 9 10 11 12 --arms v40_looped_attn v41_vendor_aware_attn
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

ABBA = REPO / "bench" / "abba.py"


def main() -> int:
    from bench.gpu_lock import gpu_lock

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("abba", "isolated"), required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--trials", type=int, default=2,
                    help="isolated mode: independent processes per (config, arm)")
    ap.add_argument("--out-prefix", default=None)
    a = ap.parse_args()

    with gpu_lock(f"g41 A/B {a.mode}", timeout_s=21600):
        if a.mode == "abba":
            cmd = [sys.executable, str(ABBA), "--ids", *map(str, a.ids),
                   "--arms", *a.arms, "--rounds", str(a.rounds),
                   "--warmup", str(a.warmup)]
            if a.out_prefix:
                cmd += ["--out", f"{a.out_prefix}.json"]
            return subprocess.run(cmd, cwd=str(REPO)).returncode
        # ISOLATED: one arm per process, so no two arms are ever co-resident. Replicated,
        # because a single row is not a measurement (finding 49).
        for trial in range(a.trials):
            for arm in a.arms:
                print(f"\n########## trial {trial}  arm {arm} ##########", flush=True)
                cmd = [sys.executable, str(ABBA), "--ids", *map(str, a.ids),
                       "--arms", arm, "--rounds", str(a.rounds),
                       "--warmup", str(a.warmup)]
                if a.out_prefix:
                    cmd += ["--out", f"{a.out_prefix}_t{trial}_{arm}.json"]
                subprocess.run(cmd, cwd=str(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
