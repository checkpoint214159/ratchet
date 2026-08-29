"""CPU-only contracts for citation-aware planning scout behavior."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ratchet.optimization import (
    ScoutIntentState,
    ScoutProposal,
    assess_scout_proposal,
)

ROOT = Path(__file__).resolve().parents[2]


def _proposal(citation_key: str = "dao2024flashattention2") -> ScoutProposal:
    return ScoutProposal(
        "SCOUT-001",
        "change work partitioning rather than tune a scalar",
        "future hardware-qualified attention regime",
        citation_key,
        "FlashAttention-2, work partitioning discussion",
    )


def test_scout_opens_only_reviewed_citation_backed_architectural_intents():
    intent = assess_scout_proposal(_proposal(), ("dao2024flashattention2",))

    assert intent.state is ScoutIntentState.OPEN
    assert intent.reason is None
    assert intent.scope == "planning_only"
    assert intent.qualification_gate == "FG-01"
    assert intent.execution_permitted is False


def test_scout_preserves_an_unreviewed_citation_as_an_explicit_rejection():
    rejected = assess_scout_proposal(
        _proposal("unknown_source"), ("dao2024flashattention2",)
    )

    assert rejected.state is ScoutIntentState.REJECTED
    assert rejected.reason == "citation_not_reviewed"
    with pytest.raises(ValueError, match="reviewed citation"):
        assess_scout_proposal(
            _proposal(), ("dao2024flashattention2", "dao2024flashattention2")
        )


def test_scout_has_no_runtime_candidate_or_archive_path():
    source = ROOT / "ratchet" / "optimization" / "scout.py"
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
        or name.startswith("ratchet.models")
    }
