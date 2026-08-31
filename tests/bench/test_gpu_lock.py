"""The GPU exclusivity guard. One device, several agents."""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bench.gpu_lock import (contention_report, foreign_cuda_processes, gpu_lock,
                            LOCK_PATH, _stale)


def test_lock_is_exclusive_and_names_its_holder():
    with gpu_lock("test-a"):
        assert LOCK_PATH.exists()
        assert "test-a" in LOCK_PATH.read_text()
        with pytest.raises(RuntimeError, match="held by"):
            with gpu_lock("test-b"):
                pass
    assert not LOCK_PATH.exists(), "lock must be released on exit"


def test_lock_is_released_even_when_the_body_raises():
    with pytest.raises(ValueError):
        with gpu_lock("test-boom"):
            raise ValueError("boom")
    assert not LOCK_PATH.exists()


def test_a_crashed_owner_does_not_block_the_queue_forever():
    """One crashed measurement must not wedge every later one. A lock naming a pid that
    no longer exists is reclaimed."""
    LOCK_PATH.write_text("999999999 ghost 0\n")
    try:
        assert _stale(LOCK_PATH) is True
        with gpu_lock("after-ghost"):
            assert str(os.getpid()) in LOCK_PATH.read_text()
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def test_a_live_owner_is_not_treated_as_stale():
    LOCK_PATH.write_text(f"{os.getpid()} live 0\n")
    try:
        assert _stale(LOCK_PATH) is False
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def test_foreign_detection_is_documented_as_unreliable_not_as_a_guarantee():
    """L36, applied to our own guard.

    Under WSL2 `nvidia-smi --query-compute-apps` reports intermittently. Measured: a
    process holding a 16 MB CUDA tensor, confirmed alive, was detected on one trial and
    invisible on the next seven seconds later, same command. So a CLEAN report proves
    nothing, and anything that reads it as "the GPU is free" is converting blindness
    into evidence.

    This test pins the honesty of the docstring rather than asserting behaviour the
    platform does not provide -- a flaky assertion here would be the same mistake one
    level up.
    """
    from bench import gpu_lock as gl
    doc = gl.__doc__ or ""
    assert "UNRELIABLE" in doc.upper(), "the docstring must not imply this check is reliable"
    assert "means nothing" in doc, "a clean report must be documented as non-evidence"
    assert (gl.foreign_cuda_processes.__doc__ or "").find("BEST EFFORT") >= 0


def test_foreign_detection_never_raises_on_this_platform():
    """It may see nothing, but it must not crash the guard that wraps it."""
    assert isinstance(foreign_cuda_processes(), list)


def test_run_matrix_refuses_on_contention_rather_than_warning():
    """Same standard as the dirty-tree rule: a warning routinely ignored is not a
    guardrail. Pinned at the source so the refusal cannot be softened to a print."""
    src = (Path(__file__).resolve().parents[2] / "bench" / "run_matrix.py").read_text()
    assert "REFUSING" in src and "contention_report" in src
    assert "return 3" in src, "contention must exit non-zero, not fall through"


def test_every_measured_row_records_whether_the_run_held_the_lock():
    """With several agents on one GPU, a row that cannot say whether it had exclusive
    access is a row nobody can trust later.

    The nvidia-smi detector cannot answer 'was this clean?' -- finding 26 measured it
    reporting a live CUDA process on one trial and nothing seven seconds later. Our own
    lock CAN answer 'did this run have exclusive access by our protocol?', which is the
    question that matters, so that is what gets written down.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "bench" / "run_matrix.py").read_text()
    assert "gpu_exclusive" in src
    assert "gpu_exclusive={gpu_exclusive}" in src, "it must reach the ledger notes"


def test_release_compares_the_pid_field_not_a_string_prefix():
    """A lock held by pid 994932 must not be released by pid 9949.

    `read_text().startswith(str(os.getpid()))` prefix-matches, so any pid that is a string
    prefix of the holder's would release someone else's lock. A lock that silently lets the
    wrong process through is worse than no lock -- and this one guards every timing
    measurement in the project (finding 05: a co-resident process inflated a baseline 4.1x).
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "bench" / "gpu_lock.py").read_text()
    assert "startswith(str(os.getpid()))" not in src, "release is prefix-matching pids again"
    assert 'held[0] == str(os.getpid())' in src, "release must compare the pid field"
