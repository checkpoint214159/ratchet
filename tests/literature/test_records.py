"""CPU-only checks for the literature-to-future-hypothesis record."""

from __future__ import annotations

import json
import re
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
READ = ROOT / "papers_read.md"
TO_READ = ROOT / "papers_to_read.md"
BIBLIOGRAPHY = ROOT / "research" / "paper" / "bibliography.bib"
HISTORY = ROOT / "research" / "literature" / "history"
IDEA = ROOT / "research" / "ideas" / "IDEA-0001.json"
KEYS = {
    "dao2022flashattention",
    "dao2024flashattention2",
    "ansel2024pytorch",
    "schoonhoven2022autotuning",
    "pytorch_xpu_2026",
    "intel_ipex_retirement_2026",
    "intel_joint_matrix_2024",
    "intel_triton_xpu_2026",
    "spoczynski2026xeforge",
}
PRIMARY_URLS = (
    "https://arxiv.org/",
    "https://pytorch.org/",
    "https://docs.pytorch.org/",
    "https://www.intel.com/",
    "https://github.com/intel/",
    "https://github.com/IntelLabs/",
)


def _bib_entries() -> dict[str, str]:
    text = BIBLIOGRAPHY.read_text()
    return {
        key: body
        for _, key, body in re.findall(r"@(\w+)\{([^,]+),\n(.*?)\n\}", text, re.DOTALL)
    }


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _verify_history(directory: Path) -> list[dict[str, object]]:
    head = json.loads((directory / "head.json").read_text())
    records = sorted(directory.glob("LTR-*.json"))
    assert len(records) == head["record_count"]
    assert [path.name for path in records] == [
        f"LTR-{number:04d}.json" for number in range(1, len(records) + 1)
    ]
    previous = "0" * 64
    parsed: list[dict[str, object]] = []
    for number, path in enumerate(records, 1):
        record = json.loads(path.read_text())
        assert record["transition_id"] == f"LTR-{number:04d}"
        assert record["previous_digest"] == previous
        claimed = record.pop("record_digest")
        assert claimed == sha256(_canonical(record)).hexdigest()
        record["record_digest"] = claimed
        previous = claimed
        parsed.append(record)
    assert head == {
        "schema_version": 1,
        "record_count": len(records),
        "head_id": f"LTR-{len(records):04d}",
        "head_digest": previous,
    }
    return parsed


def test_exact_root_trackers_separate_read_sources_from_uncited_backlog():
    assert READ.name == "papers_read.md"
    assert TO_READ.name == "papers_to_read.md"
    read = READ.read_text()
    to_read = TO_READ.read_text()
    assert KEYS <= set(re.findall(r"`([^`]+)`", read))
    assert (
        "A methodology for comparing optimization algorithms for auto-tuning" in to_read
    )
    assert "10.1016/j.future.2024.05.021" in to_read
    assert "cpp_custom_ops_sycl.html" in to_read
    assert "optimization-guide-gpu/2024-1/overview.html" in to_read
    assert "not been read" in to_read
    assert "10.1016/j.future.2024.05.021" not in BIBLIOGRAPHY.read_text()
    assert "IPEX retirement for this project" in read


def test_bibliography_keys_have_primary_urls_and_verifiable_dois_where_claimed():
    entries = _bib_entries()
    assert set(entries) == KEYS
    for key, body in entries.items():
        urls = re.findall(r"https://[^}\s]+", body)
        assert urls, key
        assert all(url.startswith(PRIMARY_URLS) for url in urls), (key, urls)
        if "doi =" in body:
            assert re.search(r"doi = \{10\.\d+/.+\}", body), key
    assert "10.1145/3620665.3640366" in entries["ansel2024pytorch"]
    assert "10.48550/arXiv.2605.26118" in entries["spoczynski2026xeforge"]
    assert "10.1109/TEVC.2022.3210654" in entries["schoonhoven2022autotuning"]


def test_read_transitions_preserve_history_without_citing_unread_backlog():
    transitions = _verify_history(HISTORY)
    assert {item["key"] for item in transitions} == KEYS
    for item in transitions:
        assert item["from"] == "to_read"
        assert item["to"] == "read"
        assert item["on"] == "2026-08-29"
        assert item["source_url"].startswith(PRIMARY_URLS)
    assert all(key in READ.read_text() for key in KEYS)
    assert all(key not in TO_READ.read_text() for key in KEYS)


@pytest.mark.parametrize("tamper", ["delete", "reorder", "rewrite"])
def test_chained_history_rejects_deletion_reordering_and_rewrites(
    tmp_path: Path, tamper: str
):
    copied = tmp_path / "history"
    shutil.copytree(HISTORY, copied)
    if tamper == "delete":
        (copied / "LTR-0008.json").unlink()
    elif tamper == "reorder":
        first, second = copied / "LTR-0001.json", copied / "LTR-0002.json"
        first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
        first.write_bytes(second_bytes)
        second.write_bytes(first_bytes)
    else:
        record = json.loads((copied / "LTR-0004.json").read_text())
        record["key"] = "rewritten"
        (copied / "LTR-0004.json").write_bytes(_canonical(record))
    with pytest.raises(AssertionError):
        _verify_history(copied)


def test_idea_links_only_read_sources_without_preempting_ib13_protocol():
    idea = json.loads(IDEA.read_text())
    assert idea["idea_id"] == "IDEA-0001"
    assert idea["status"] == "queued"
    assert set(idea["literature_keys"]) == KEYS
    assert "torch.compile" in idea["question"]
    assert not (ROOT / "research" / "hypotheses" / "HYP-0001.json").exists()
    assert {"intended_protocol", "candidate_state", "empirical_result"}.isdisjoint(idea)
