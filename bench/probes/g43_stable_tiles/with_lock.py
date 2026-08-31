"""Run a command holding `bench/gpu_lock.py`, waiting for it rather than failing.

    python3 bench/probes/g43_stable_tiles/with_lock.py -- pytest tests/bench/...

The lock is the project's one arbiter of "only one process measures at a time"
(CLAUDE.md, method A2, finding 26). An executor that measures without it produces two
wrong numbers rather than two measurements, so every GPU-touching command in this
generation goes through here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from bench.gpu_lock import gpu_lock  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print(__doc__)
        return 2
    with gpu_lock(purpose=f"g43 {argv[0]}", timeout_s=3600.0):
        return subprocess.run(argv, cwd=str(REPO)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
