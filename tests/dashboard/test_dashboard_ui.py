"""The dashboard's own checks, run from the project gate.

The dashboard is JavaScript with no build step and no dependencies, so its tests are
`node --test` files under `dashboard/test/`. They are wrapped here rather than left to
be remembered, because a check nobody arranged to run is the same as no check (L40).

What they assert, and why those and not screenshots: no browser is available in this
environment, so the evolution tree is verified on its DATA and its GEOMETRY --
    * every registry entry appears in the layout exactly once,
    * no two node boxes overlap and no two lineage edges cross,
    * the client module, executed against a real snapshot in a minimal DOM, draws one
      node per candidate, keeps the recombination edge, and no longer claims that git
      ancestry is the lineage (finding 28).
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TESTS = sorted((REPO / "dashboard" / "test").glob("*.test.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_dashboard_node_tests_pass():
    assert TESTS, "no dashboard test files found"
    proc = subprocess.run(
        ["node", "--test", *[str(p) for p in TESTS]],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        "dashboard node tests failed\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
