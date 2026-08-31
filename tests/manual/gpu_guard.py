"""GPU exclusivity guard for the measurement harnesses.

Every speedup in the ledger is a ratio of two timings taken on one device. A ratio is
robust to clock ramp (measured: 0.3% between a 208MHz-idle and a boosted GPU, because
baseline and candidate scale together), but it is NOT robust to a second process taking
SMs away for part of the run -- that lands on whichever side happens to be executing.

This repo's search loop is single-process by design, and `search_loop.py` argues that one
process gives the same serialization guarantee a GPU lock would. That argument holds
inside a process and fails silently when a second *session* runs concurrently on the same
box, which is exactly what an agent workflow with several open sessions produces.

So: check before measuring, and record the check in the ledger row. The point is not to
prevent contention -- it is to make "measured on an idle device" an auditable fact rather
than an assumption.
"""

from __future__ import annotations

import os
import subprocess


def _own_process_tree() -> set[int]:
    """Our pid plus every ancestor, so we never flag ourselves."""
    pids = set()
    pid = os.getpid()
    for _ in range(32):  # bounded; a cycle or a missing /proc entry just stops the walk
        if pid <= 0 or pid in pids:
            break
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                # ppid is field 4, but comm (field 2) may contain spaces -- split past it.
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except Exception:
            break
    return pids


def gpu_processes() -> list[dict]:
    """Compute processes holding a CUDA context, excluding our own process tree."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout
    except Exception:
        return []  # no nvidia-smi -> cannot check; callers treat this as "unknown"

    ours = _own_process_tree()
    found = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid in ours:
            continue
        found.append({"pid": pid, "name": parts[1], "used_mib": parts[2]})
    return found


def exclusivity_record() -> dict:
    """Ledger-shaped fact about who else was on the device when we measured."""
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=15, check=True)
    except Exception:
        return {"checked": False, "exclusive": None, "others": []}
    others = gpu_processes()
    return {"checked": True, "exclusive": not others, "others": others}


def require_exclusive(allow_contention: bool = False) -> dict:
    """Abort before measuring if another process holds the GPU.

    Set `allow_contention=True` (or RATCHET_ALLOW_CONTENTION=1) to downgrade to a warning
    -- the resulting ledger rows stay honest because the record still says exclusive=False.
    """
    record = exclusivity_record()
    if not record["checked"]:
        print("[gpu_guard] nvidia-smi unavailable; exclusivity NOT verified")
        return record
    if record["exclusive"]:
        print("[gpu_guard] GPU is exclusive to this process")
        return record

    others = ", ".join(f"pid {o['pid']} ({o['name']}, {o['used_mib']} MiB)"
                       for o in record["others"])
    msg = f"GPU is shared with {len(record['others'])} other process(es): {others}"
    if allow_contention or os.environ.get("RATCHET_ALLOW_CONTENTION") == "1":
        print(f"[gpu_guard] WARNING: {msg} -- measuring anyway, rows marked non-exclusive")
        return record
    raise SystemExit(
        f"[gpu_guard] REFUSING TO MEASURE: {msg}\n"
        "  Timings taken under contention are not comparable across runs.\n"
        "  Stop the other process, or set RATCHET_ALLOW_CONTENTION=1 to override."
    )
