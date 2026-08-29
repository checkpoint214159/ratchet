"""Exclusive access to the one GPU, for the duration of a measurement.

WHY THIS EXISTS
---------------
The architecture this project is moving to runs several agents at once: research agents
that probe, expander agents that implement, and a controller that measures. They share one
RTX 4070 Ti SUPER. **Two processes on one GPU do not produce two independent measurements;
they produce two wrong ones.**

This is not hypothetical here. A co-resident model once inflated config 6's baseline 4.1x
(2037 ms against a true 446 ms) by forcing a spill to host memory -- finding 05. That was
two models inside ONE process, and the fix was to time the arms in isolation. Two
PROCESSES is the same failure with none of the same defences: the allocator cannot see the
other tenant, and the timing loop cannot know it was descheduled.

The dirty-tree guard already refuses rather than warns, on the reasoning that a warning
routinely ignored is not a guardrail. This applies the same standard to a hazard that
corrupts numbers instead of discarding them -- which is worse, because a discarded row is
visibly absent and a corrupted one is not.

TWO CHECKS, AND ONE OF THEM DOES NOT WORK HERE
----------------------------------------------
  * **The lock file is the real mechanism.** Cooperative, names the holding pid, reclaims
    a lock whose owner died. Every tool in this repo that measures must take it.

  * The nvidia-smi foreign-process check is **BEST EFFORT AND UNRELIABLE ON WSL2.**
    Measured directly: with a process holding a 16 MB CUDA tensor and confirmed alive,
    two identical trials seven seconds apart gave

        trial 1   nvidia-smi -> "893453, [N/A]"     detected
        trial 2   nvidia-smi -> ""                  NOT detected

    Same command, same kind of holder, opposite answers. Under WSL2 the compute-apps
    query reports intermittently (and always with used_memory as [N/A]).

    **So a clean report from this check means nothing.** It is kept because a positive
    result is still true -- if it names a process, that process is really there -- and a
    subagent's throwaway probe knows nothing about our lock file. But it must never be
    read as evidence that the GPU is free. That inversion is L36: a check that cannot
    observe converts absence of visibility into positive evidence.

CONSEQUENCE FOR EXISTING ROWS. Any sweep run while a research agent was probing may be
contended and is not re-derivable after the fact. See finding 26.

The lock is advisory: `--allow-contended` exists because a capability probe (does this
shape OOM?) is not a timing measurement and does not need the GPU to itself.
"""

from __future__ import annotations

import errno
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_PATH = Path(os.environ.get("RATCHET_GPU_LOCK", "/tmp/ratchet-gpu.lock"))


def foreign_cuda_processes() -> list[tuple[int, str]]:
    """(pid, used_memory) for every CUDA process that is not us or our children.

    BEST EFFORT ONLY -- see the module docstring. An empty list does NOT mean the GPU is
    free; under WSL2 this query reports intermittently. A non-empty list IS trustworthy.

    Returns [] when nvidia-smi is unavailable rather than raising: on a machine without
    it the guard simply cannot help, and refusing to run would be worse than proceeding.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    mine = {os.getpid(), os.getppid()}
    found = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if pid in mine:
            continue
        found.append((pid, parts[1]))
    return found


def _stale(path: Path) -> bool:
    """A lock whose owner is gone. Checked so one crashed run does not block the queue."""
    try:
        pid = int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return True
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.ESRCH
    return False


@contextmanager
def gpu_lock(purpose: str = "measurement", timeout_s: float = 0.0):
    """Hold the GPU exclusively. Raises RuntimeError if it cannot, unless timeout_s > 0,
    in which case it waits that long first."""
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {purpose} {time.time():.0f}\n".encode())
            os.close(fd)
            break
        except FileExistsError:
            if _stale(LOCK_PATH):
                LOCK_PATH.unlink(missing_ok=True)
                continue
            if time.time() >= deadline:
                holder = LOCK_PATH.read_text().strip() if LOCK_PATH.exists() else "?"
                raise RuntimeError(f"GPU lock held by: {holder}")
            time.sleep(1.0)
    try:
        yield
    finally:
        try:
            if LOCK_PATH.exists() and LOCK_PATH.read_text().startswith(str(os.getpid())):
                LOCK_PATH.unlink()
        except OSError:
            pass


def contention_report() -> str | None:
    """A human-readable reason to refuse, or None if the GPU looks free."""
    foreign = foreign_cuda_processes()
    if foreign:
        who = ", ".join(f"pid {p} ({m} MiB)" for p, m in foreign)
        return (f"another CUDA process is resident: {who}. Timings taken alongside it "
                f"are not measurements of this candidate -- a co-resident model once "
                f"inflated a baseline 4.1x (finding 05).")
    if LOCK_PATH.exists() and not _stale(LOCK_PATH):
        return f"the GPU lock is held: {LOCK_PATH.read_text().strip()}"
    return None
