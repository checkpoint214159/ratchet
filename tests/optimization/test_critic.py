"""CPU-only contracts for the no-measurement critic boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ratchet.optimization import (
    CriticDecision,
    CriticEpoch,
    CriticState,
    dormant_critic_decision,
)

ROOT = Path(__file__).resolve().parents[2]


def test_critic_epoch_holds_out_whole_candidates_and_is_frozen():
    epoch = CriticEpoch(
        "CRITIC-EPOCH-001",
        ("CAND-alpha", "CAND-beta"),
        ("CAND-gamma",),
    )

    assert epoch.training_candidate_ids == ("CAND-alpha", "CAND-beta")
    assert epoch.held_out_candidate_ids == ("CAND-gamma",)
    with pytest.raises(ValueError, match="hold out"):
        CriticEpoch("CRITIC-EPOCH-001", ("CAND-alpha",), ("CAND-alpha",))
    with pytest.raises(ValueError, match="hold out"):
        CriticEpoch("CRITIC-EPOCH-001", (), ("CAND-gamma",))


def test_current_critic_records_only_a_dormant_non_score_decision():
    epoch = CriticEpoch("CRITIC-EPOCH-001", ("CAND-alpha",), ("CAND-gamma",))

    decision = dormant_critic_decision(epoch, "CAND-next", 0)

    assert decision == CriticDecision(
        "CRITIC-EPOCH-001",
        "CAND-next",
        CriticState.DORMANT,
        "no_empirical_measurements",
    )
    assert not any(
        name in decision.__dataclass_fields__
        for name in ("score", "probability", "speedup")
    )
    with pytest.raises(ValueError, match="zero empirical"):
        dormant_critic_decision(epoch, "CAND-next", 1)


def test_critic_has_no_runtime_or_archive_dependency():
    source = ROOT / "ratchet" / "optimization" / "critic.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not {
        name
        for name in imports
        if name.split(".")[0] in {"torch", "triton"}
        or name.startswith("ratchet.backends")
        or name.startswith("ratchet.experiments")
        or name.startswith("ratchet.measurement")
    }
