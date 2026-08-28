"""CPU-only contract tests for the literature-only paper pipeline."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

import ratchet.reporting.paper as paper_module
from ratchet.experiments import FileExperimentArchive
from ratchet.reporting.paper import (
    PaperBuildError,
    build_paper,
    generate_sources,
    generated_digest,
    reject_empirical_claims,
    select_paper_content,
)

ROOT = Path(__file__).resolve().parents[2]


def _paper_root(tmp_path: Path) -> Path:
    shutil.copy2(ROOT / "papers_read.md", tmp_path / "papers_read.md")
    shutil.copytree(ROOT / "research", tmp_path / "research")
    return tmp_path


def test_generation_is_byte_identical_and_declares_the_no_run_state(tmp_path: Path):
    root = _paper_root(tmp_path)

    selection = generate_sources(root)
    first_digest = generated_digest(root)
    generated = root / "research" / "paper" / "generated"
    first = {path.name: path.read_bytes() for path in sorted(generated.iterdir())}

    assert generate_sources(root) == selection
    assert generated_digest(root) == first_digest
    assert {
        path.name: path.read_bytes() for path in sorted(generated.iterdir())
    } == first
    assert selection.event_count == 0
    assert json.loads((generated / "catalogue.json").read_text()) == {
        "citation_keys": list(selection.citation_keys),
        "empirical_claims_permitted": False,
        "evidence_boundary": {
            "conclusion_kind": "literature_synthesis_only",
            "experiment_events": 0,
            "reviewed_primary_sources": 9,
        },
        "event_count": 0,
        "projection_id": selection.projection_id,
        "schema_version": 1,
        "selection": "reviewed_literature_and_catalogue_only",
    }
    assert "zero experiment events" in (generated / "no_run.tex").read_text()
    assert (
        "Immutable experiment events & 0" in (generated / "catalogue.tex").read_text()
    )
    boundary = (generated / "evidence_boundary.tex").read_text()
    assert "9 reviewed primary sources" in boundary
    assert "0 experiment events" in boundary
    assert "Literature synthesis only; no empirical conclusions" in boundary
    assert (
        r"\input{generated/evidence_boundary.tex}"
        in (root / "research" / "paper" / "main.tex").read_text()
    )


def test_reviewed_literature_citations_resolve_exactly_to_bibliography():
    selection = select_paper_content(ROOT)
    archive = FileExperimentArchive(ROOT / "research" / "archive")
    archive.verify()
    projection = archive.projection()
    bibliography = (ROOT / "research" / "paper" / "bibliography.bib").read_text()

    assert selection.projection_id == projection.projection_id
    assert selection.event_count == projection.event_count
    assert projection.event_ids == ()
    assert set(selection.citation_keys) == set(
        re.findall(r"@\w+\{([^,]+),", bibliography)
    )
    assert re.findall(r"\\(?:[A-Za-z@]+|.)", bibliography) == [r"\url"]
    assert len(selection.citation_keys) == 9


def test_empty_catalogue_rejects_empirical_result_language():
    with pytest.raises(PaperBuildError, match="zero-event catalogue"):
        reject_empirical_claims(0, ("Measured latency improved by a speedup.",))


@pytest.mark.parametrize(
    "claim",
    (
        "No measurements exist, but latency was measured.",
        "No results; the candidate is 2x faster.",
        "No benchmark exists: throughput improved by 20%.",
        "The setup did not change, but measured latency was recorded.",
        "The candidate runs twice as fast as the baseline.",
    ),
)
def test_empty_catalogue_rejects_mixed_clause_and_comparative_claims(claim: str):
    with pytest.raises(PaperBuildError, match="zero-event catalogue"):
        reject_empirical_claims(0, (claim,))


def test_pipeline_rejects_an_unverified_raw_manifest(tmp_path: Path):
    root = _paper_root(tmp_path)
    manifest_path = root / "research" / "archive" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["events"].append({"event_id": "EVT-000001"})
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(PaperBuildError, match="verified archive projection"):
        generate_sources(root)


def test_pipeline_scans_included_hand_authored_prose_for_empirical_claims(
    tmp_path: Path,
):
    root = _paper_root(tmp_path)
    scope = root / "research" / "paper" / "sections" / "scope.tex"
    scope.write_text(scope.read_text() + "\nMeasured latency improved by a speedup.\n")

    with pytest.raises(PaperBuildError, match="scope.tex"):
        generate_sources(root)


def test_pipeline_rejects_an_empirical_token_even_when_negated(tmp_path: Path):
    root = _paper_root(tmp_path)
    scope = root / "research" / "paper" / "sections" / "scope.tex"
    scope.write_text(scope.read_text() + "\nNo latency was measured.\n")

    with pytest.raises(PaperBuildError, match="scope.tex"):
        generate_sources(root)


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        (r"\bibliography{../../outside}", "dependency path"),
        (r"\input ../outside", "braced local path"),
        (r"\include{sections/scope}", "unsupported TeX"),
        (r"\includegraphics{../../outside.pdf}", "unsupported TeX"),
        (r"\addbibresource{../../outside.bib}", "unsupported TeX"),
    ),
)
def test_pipeline_rejects_escaping_or_unsupported_tex_dependencies(
    tmp_path: Path, replacement: str, message: str
):
    root = _paper_root(tmp_path)
    main = root / "research" / "paper" / "main.tex"
    main.write_text(
        main.read_text().replace(r"\bibliography{bibliography}", replacement)
    )

    with pytest.raises(PaperBuildError, match=message):
        generate_sources(root)


def test_pipeline_handles_local_input_cycles_without_recursion(tmp_path: Path):
    root = _paper_root(tmp_path)
    sections = root / "research" / "paper" / "sections"
    scope = sections / "scope.tex"
    scope.write_text(scope.read_text() + r"\input{cycle}" + "\n")
    (sections / "cycle.tex").write_text(r"\input{scope}" + "\n")

    assert generate_sources(root).event_count == 0


def test_pipeline_rejects_bibliography_input_escape(tmp_path: Path):
    root = _paper_root(tmp_path)
    bibliography = root / "research" / "paper" / "bibliography.bib"
    bibliography.write_text(bibliography.read_text() + "\n\\input{../../outside}\n")

    with pytest.raises(
        PaperBuildError, match="unsupported bibliography command: input"
    ):
        generate_sources(root)


def test_tectonic_invocation_is_cached_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = _paper_root(tmp_path)
    commands: list[list[str]] = []
    run_options: list[dict[str, object]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        run_options.append(kwargs)
        output = Path(command[command.index("--outdir") + 1])
        (output / "main.pdf").write_bytes(b"%PDF-1.5\n")
        return subprocess.CompletedProcess(command, 0, "")

    monkeypatch.setattr(paper_module.shutil, "which", lambda _tool: "/fake/tectonic")
    monkeypatch.setattr(paper_module.subprocess, "run", fake_run)

    assert build_paper(root) == root / "research" / "paper" / "latest.pdf"
    assert commands and "--only-cached" in commands[0]
    assert "--untrusted" in commands[0]
    assert run_options[0]["env"] == {
        **paper_module.os.environ,
        "SOURCE_DATE_EPOCH": "0",
    }


def test_tectonic_builds_a_valid_pdf_when_available(tmp_path: Path):
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        pytest.skip("Tectonic is unavailable")

    root = _paper_root(tmp_path)
    built = build_paper(root, tectonic=tectonic)
    first = built.read_bytes()
    first_digest = sha256(first).hexdigest()

    assert build_paper(root, tectonic=tectonic) == built

    assert built.name == "latest.pdf"
    assert first.startswith(b"%PDF-")
    assert built.stat().st_size > 1_000
    assert built.read_bytes() == first
    assert sha256(built.read_bytes()).hexdigest() == first_digest
