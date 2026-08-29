"""The stage-1 screen: policy tests, no GPU."""
import json
import math

import pytest

from bench.screen import SCREEN_IDS, NOISE, decide, parent_screen_geomean
from bench.ledger import BenchLedger
from bench.matrix import BY_ID, regime_of


COMPILED = {i: 10.0 for i in SCREEN_IDS}


def _ms(speedup):
    return {i: 10.0 / speedup for i in SCREEN_IDS}


def test_screen_covers_four_distinct_regimes():
    """A screen that samples one regime would promote a candidate that destroys another."""
    regimes = {regime_of(i) for i in SCREEN_IDS}
    assert len(regimes) == len(SCREEN_IDS), f"screen regimes collapsed: {regimes}"


def test_screen_includes_the_head_dim_8_blind_spot():
    """Configs 7 and 11 have head_dim=8, where vendor fast paths may refuse. The rubric
    puts +5 on that regime and it has never been investigated -- a screen that cannot see
    it cannot promote work aimed at it."""
    assert any(BY_ID[i].head_dim == 8 for i in SCREEN_IDS)


def test_correctness_failure_rejects_regardless_of_speed():
    v, geo, detail = decide(_ms(100.0), [{"config_id": 7, "status": "ok", "correct": False}],
                            COMPILED, parent_geo=1.0)
    assert v == "REJECT" and "correctness" in detail


def test_incomplete_screen_is_a_reject_not_a_pass():
    """A missing config must not be silently averaged over -- skipping would reward a
    candidate that crashed on the config it was worst at."""
    partial = _ms(2.0)
    partial.pop(SCREEN_IDS[0])
    assert decide(partial, [], COMPILED, parent_geo=1.0)[0] == "REJECT"


def test_promotes_within_the_noise_floor_rather_than_demanding_an_improvement():
    """One pass cannot resolve anything inside +/-7%. Demanding a win the screen lacks
    the resolution to see would reject good candidates at random (L29)."""
    parent = 2.0
    v, geo, _ = decide(_ms(parent * (1 - NOISE / 2)), [], COMPILED, parent_geo=parent)
    assert v == "PROMOTE"


def test_rejects_a_candidate_clearly_worse_than_its_parent():
    v, geo, _ = decide(_ms(2.0 * (1 - NOISE * 3)), [], COMPILED, parent_geo=2.0)
    assert v == "REJECT"


def test_no_parent_baseline_still_promotes():
    """A first candidate on a new lineage has nothing to compare against; it must reach
    the confirm stage rather than be rejected for lacking a parent."""
    assert decide(_ms(1.5), [], COMPILED, parent_geo=None)[0] == "PROMOTE"


def test_parent_geomean_is_none_for_an_unmeasured_parent():
    assert parent_screen_geomean(BenchLedger(), "no_such_candidate") is None


def test_parent_geomean_reads_a_real_candidate():
    """v13 has a complete recorded sweep, so its screen-restricted geomean must exist."""
    g = parent_screen_geomean(BenchLedger(), "v13_safe_capture")
    assert g is not None and 0.5 < g < 20.0


def test_screen_results_never_enter_the_measurement_ledger():
    """Screen rows are partial sweeps. Letting them into clade statistics would swamp the
    full sweeps they are meant to gate."""
    from bench.screen import SCREEN_LOG
    from bench.ledger import DEFAULT_PATH
    assert SCREEN_LOG.name != str(DEFAULT_PATH)
    assert "screen" in SCREEN_LOG.name


def test_row_payload_carries_correctness_so_the_screen_can_see_it():
    """The __ROW__ line omitted correctness, so every screened candidate hard-rejected.

    status=="ok" already implies correctness passed (run_matrix returns "incorrect"
    before it ever times a candidate), but an invariant no consumer can observe is one
    no consumer can enforce.
    """
    import subprocess, sys, json
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent.parent
    src = (repo / "bench" / "run_matrix.py").read_text()
    emit = src[src.index('print("__ROW__"'):src.index('print("__ROW__"') + 400]
    assert '"correctness"' in emit, "__ROW__ must carry correctness"


def test_run_matrix_never_times_an_incorrect_candidate():
    """CLAUDE.md rule 3, pinned at the source: a failed correctness check must return
    before the timing block, not merely be recorded alongside it."""
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent.parent
    src = (repo / "bench" / "run_matrix.py").read_text()
    gate = src.index('out["status"] = "incorrect"')
    timing = src.index("cand_ms = min(median_ms")
    assert gate < timing, "correctness gate must precede timing"
    assert "return out" in src[gate:timing], "the gate must RETURN, not fall through"
