"""CPU-only integrity checks for the literature-only Intel Arc survey."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SURVEY_JSON = ROOT / "research" / "literature" / "intel_arc_bottleneck_survey.json"
SURVEY_MARKDOWN = ROOT / "research" / "literature" / "intel_arc_bottleneck_survey.md"
READ = ROOT / "papers_read.md"
BIBLIOGRAPHY = ROOT / "research" / "paper" / "bibliography.bib"


def _survey() -> dict[str, object]:
    return json.loads(SURVEY_JSON.read_text())


def _reviewed_keys() -> set[str]:
    return set(re.findall(r"^\| `([^`]+)` \|", READ.read_text(), re.MULTILINE))


def _bibliography_keys() -> set[str]:
    return set(re.findall(r"@\w+\{([^,]+),", BIBLIOGRAPHY.read_text()))


def test_survey_is_explicitly_literature_only_and_not_a_project_profile():
    survey = _survey()
    markdown = SURVEY_MARKDOWN.read_text()

    assert set(survey) == {
        "schema_version",
        "survey_id",
        "scope",
        "target",
        "project_profile_status",
        "literature_observations",
        "hypotheses",
    }
    assert survey["schema_version"] == 1
    assert survey["survey_id"] == "LIT-SURVEY-0001"
    assert survey["scope"] == "literature_only_not_project_profile"
    assert survey["target"] == "future_intel_arc_xpu"
    assert survey["project_profile_status"] == "not_measured"
    assert "not a\nproject profile" in markdown
    assert "no project\nprofiling trace, timing, or kernel result" in markdown


def test_every_literature_observation_traces_to_reviewed_bibliography_keys():
    survey = _survey()
    observations = survey["literature_observations"]
    assert isinstance(observations, list) and observations
    reviewed = _reviewed_keys()
    bibliography = _bibliography_keys()

    assert "torch.compile" not in reviewed
    assert len(reviewed) == 9

    for observation in observations:
        assert set(observation) == {"claim_id", "statement", "citation_keys"}
        assert observation["claim_id"].startswith("OBS-")
        assert observation["statement"]
        assert observation["citation_keys"]
        assert set(observation["citation_keys"]) <= reviewed
        assert set(observation["citation_keys"]) <= bibliography

    observation_ids = [observation["claim_id"] for observation in observations]
    assert len(observation_ids) == len(set(observation_ids))
    observation_by_id = {
        observation["claim_id"]: observation for observation in observations
    }
    assert observation_by_id["OBS-003"]["statement"] == (
        "PyTorch 2 introduces optional torch.compile graph capture/compilation "
        "alongside eager execution."
    )
    assert "baseline" not in observation_by_id["OBS-003"]["statement"].lower()


def test_every_project_specific_statement_is_a_cited_hypothesis():
    survey = _survey()
    hypotheses = survey["hypotheses"]
    assert isinstance(hypotheses, list) and hypotheses
    reviewed = _reviewed_keys()
    bibliography = _bibliography_keys()

    hypothesis_ids = [hypothesis["hypothesis_id"] for hypothesis in hypotheses]
    assert len(hypothesis_ids) == len(set(hypothesis_ids))
    assert not set(hypothesis_ids) & {
        observation["claim_id"] for observation in survey["literature_observations"]
    }

    markdown = SURVEY_MARKDOWN.read_text()
    blocks = {
        match.group("identifier"): match.group("block")
        for match in re.finditer(
            r"^- `(?P<identifier>HYP-LIT-\d+)`(?P<block>.*?)(?=^- `HYP-LIT-|\nThe machine-readable record)",
            markdown,
            re.MULTILINE | re.DOTALL,
        )
    }
    assert set(blocks) == set(hypothesis_ids)

    for hypothesis in hypotheses:
        assert set(hypothesis) == {
            "hypothesis_id",
            "label",
            "candidate_bottleneck",
            "statement",
            "citation_keys",
        }
        assert hypothesis["hypothesis_id"].startswith("HYP-LIT-")
        assert hypothesis["label"] == "hypothesis"
        assert hypothesis["statement"].startswith("Hypothesis:")
        assert hypothesis["candidate_bottleneck"]
        assert hypothesis["citation_keys"]
        assert set(hypothesis["citation_keys"]) <= reviewed
        assert set(hypothesis["citation_keys"]) <= bibliography
        block = blocks[hypothesis["hypothesis_id"]]
        assert "**Hypothesis:**" in block
        assert "future" in block.lower() or "qualified xpu gate" in block.lower()
        for citation_key in hypothesis["citation_keys"]:
            assert f"`{citation_key}`" in block
        assert not re.search(
            r"\b(?:ratchet|this project)\s+(?:has|measured|profiled|produced|observed)\b",
            block,
            re.IGNORECASE,
        )

    graph_hypothesis = next(
        hypothesis
        for hypothesis in hypotheses
        if hypothesis["hypothesis_id"] == "HYP-LIT-003"
    )
    assert "separate baseline conditions" in graph_hypothesis["statement"]
