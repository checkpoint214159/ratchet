"""CPU-only regression coverage for the legacy GPU golden-suite boundary."""

from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def _load_golden_conftest_without_torch():
    """Load the collection guard while simulating an absent optional dependency."""
    conftest_path = Path(__file__).parent / "golden" / "conftest.py"
    original_import = __import__

    def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=import_without_torch):
        return runpy.run_path(conftest_path)


def test_golden_conftest_ignores_gpu_tests_when_torch_is_missing():
    """The missing-runtime branch must work independently of this runner's packages."""
    conftest = _load_golden_conftest_without_torch()

    assert conftest["torch"] is None
    assert conftest["collect_ignore_glob"] == ["test_*.py"]


def test_golden_collection_handles_missing_torch_cleanly():
    """GPU tests must not turn an absent optional runtime into a collection error."""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/golden", "--collect-only", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    expected_exit = 0 if importlib.util.find_spec("torch") else 5
    assert result.returncode == expected_exit, result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stdout + result.stderr
