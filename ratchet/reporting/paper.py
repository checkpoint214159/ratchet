"""Deterministic literature-only LaTeX paper generation."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Mapping

from ratchet.experiments import ArchiveIntegrityError, FileExperimentArchive

_BIB_KEY = re.compile(r"@\w+\{([^,]+),")
_TRACKER_KEY = re.compile(r"^\| `([^`]+)`", re.MULTILINE)
_EMPIRICAL_LANGUAGE = re.compile(
    r"\b("
    r"measur(?:ed|ement(?:s)?)|benchmark(?:ed|ing)?|speedup|latency|throughput|"
    r"correctness(?:\s+(?:result|passed|failure))?|"
    r"performance(?:\s+(?:result|improvement|gain|measurement))?"
    r")\b",
    re.IGNORECASE,
)
_COMPARATIVE_LANGUAGE = re.compile(
    r"\b("
    r"\d+(?:\.\d+)?\s*(?:x|×)\s*(?:faster|slower|better)|"
    r"\d+(?:\.\d+)?%\s*(?:faster|slower|improvement|gain|reduction)|"
    r"(?:once|twice|three|four|five|six|seven|eight|nine|ten)(?:\s+times?)?\s+"
    r"(?:as\s+)?(?:fast|faster|slow|slower|good|better)|"
    r"(?:outperform(?:s|ed|ing)?|improv(?:e|ed|ement)|reduc(?:e|ed|tion)|"
    r"increas(?:e|ed|e)|faster|slower|better|beats?|wins?)\b|"
    r"\b(?:higher|lower)\s+(?:latency|throughput|memory)"
    r")",
    re.IGNORECASE,
)
_INPUT = re.compile(r"\\input\s*\{([^}]+)\}")
_INPUT_COMMAND = re.compile(r"\\input\b")
_COMMAND = re.compile(r"\\([A-Za-z@]+)")
_BIB_COMMAND = re.compile(r"\\([A-Za-z@]+|.)")
_ALLOWED_COMMANDS = frozenset(
    {
        "author",
        "begin",
        "bibliography",
        "bibliographystyle",
        "caption",
        "centering",
        "cite",
        "date",
        "documentclass",
        "end",
        "figure",
        "hline",
        "input",
        "maketitle",
        "section",
        "title",
        "usepackage",
    }
)
_DEPENDENCY_COMMAND = re.compile(
    r"\\(documentclass|usepackage|bibliography|bibliographystyle|include|"
    r"includegraphics|addbibresource)\s*(?:\[[^]]*\]\s*)?\{([^}]*)\}"
)
_ALLOWED_PACKAGES = frozenset({"fontenc", "hyperref", "url"})
_ALLOWED_BIB_COMMANDS = frozenset({"url"})


class PaperBuildError(ValueError):
    """Raised when a paper would overstate the available evidence."""


@dataclass(frozen=True, slots=True)
class PaperSelection:
    """Reviewed literature and one verified archive projection for a paper build."""

    citation_keys: tuple[str, ...]
    projection_id: str
    event_count: int


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _bibliography_keys(path: Path) -> set[str]:
    return set(_BIB_KEY.findall(path.read_text()))


def _validate_bibliography(path: Path) -> None:
    """Allow only the one harmless control word required by the checked-in BibTeX."""

    commands = set(_BIB_COMMAND.findall(path.read_text()))
    unsupported = commands - _ALLOWED_BIB_COMMANDS
    if unsupported:
        raise PaperBuildError(
            "unsupported bibliography command: " + ", ".join(sorted(unsupported))
        )


def select_paper_content(root: Path | None = None) -> PaperSelection:
    """Select reviewed literature and immutable catalogue facts without execution."""

    project = root or repository_root()
    tracker = (project / "papers_read.md").read_text()
    citations = tuple(sorted(set(_TRACKER_KEY.findall(tracker))))
    bibliography_path = project / "research" / "paper" / "bibliography.bib"
    _validate_bibliography(bibliography_path)
    bibliography = _bibliography_keys(bibliography_path)
    if not citations or set(citations) != bibliography:
        raise PaperBuildError(
            "reviewed literature and bibliography keys must resolve exactly"
        )
    archive = FileExperimentArchive(project / "research" / "archive")
    try:
        archive.verify()
        projection = archive.projection()
    except ArchiveIntegrityError as error:
        raise PaperBuildError("paper requires a verified archive projection") from error
    return PaperSelection(citations, projection.projection_id, projection.event_count)


def reject_empirical_claims(event_count: int, claims: Iterable[str]) -> None:
    """Reject all empirical-result language in zero-event hand-authored prose."""

    if event_count != 0:
        return
    for claim in claims:
        if _EMPIRICAL_LANGUAGE.search(claim) or _COMPARATIVE_LANGUAGE.search(claim):
            raise PaperBuildError(
                "zero-event catalogue cannot contain empirical claims"
            )


def _dependency_path(paper: Path, source: Path, value: str, suffix: str) -> Path:
    if not value or Path(value).is_absolute():
        raise PaperBuildError("paper dependency path must remain under research/paper")
    target = source.parent / value
    if suffix and target.suffix != suffix:
        target = target.with_suffix(suffix)
    resolved = target.resolve()
    try:
        resolved.relative_to(paper.resolve())
    except ValueError as error:
        raise PaperBuildError(
            "paper dependency path must remain under research/paper"
        ) from error
    return resolved


def _validate_tex_dependencies(
    paper: Path, sources: Iterable[tuple[Path, str]]
) -> None:
    """Enforce the fixed, local TeX dependency surface before invoking Tectonic."""

    for source, content in sources:
        braced_inputs = {match.start() for match in _INPUT.finditer(content)}
        if any(
            match.start() not in braced_inputs
            for match in _INPUT_COMMAND.finditer(content)
        ):
            raise PaperBuildError("paper input must use a braced local path")
        commands = set(_COMMAND.findall(content))
        unsupported = commands - _ALLOWED_COMMANDS
        if unsupported:
            raise PaperBuildError(
                "unsupported TeX command in paper source: "
                + ", ".join(sorted(unsupported))
            )
        for command, value in _DEPENDENCY_COMMAND.findall(content):
            if command in {"include", "includegraphics", "addbibresource"}:
                raise PaperBuildError(f"unsupported TeX dependency command: {command}")
            if command == "documentclass" and value != "article":
                raise PaperBuildError("document class must be article")
            if command == "usepackage":
                packages = {package.strip() for package in value.split(",")}
                if not packages or not packages <= _ALLOWED_PACKAGES:
                    raise PaperBuildError("paper package is not in the fixed allowlist")
            if command == "bibliographystyle" and value != "plain":
                raise PaperBuildError("bibliography style must be plain")
            if command == "bibliography":
                target = _dependency_path(paper, source, value, ".bib")
                if target != (paper / "bibliography.bib").resolve():
                    raise PaperBuildError(
                        "bibliography must be research/paper/bibliography.bib"
                    )


def _included_prose(
    paper: Path, generated: Mapping[Path, str]
) -> tuple[tuple[Path, str], ...]:
    """Read the hand-authored and generated TeX sources reachable from ``main.tex``."""

    root = paper.resolve()
    visited: set[Path] = set()
    sources: list[tuple[Path, str]] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise PaperBuildError(
                "paper input must remain under research/paper"
            ) from error
        try:
            content = (
                generated[resolved] if resolved in generated else resolved.read_text()
            )
        except OSError as error:
            raise PaperBuildError(
                f"cannot read included paper source: {resolved}"
            ) from error
        visited.add(resolved)
        sources.append((resolved, content))
        for match in _INPUT.finditer(content):
            target = _dependency_path(root, resolved, match.group(1), ".tex")
            visit(target)

    visit(root / "main.tex")
    return tuple(sources)


def _validate_no_run_prose(
    selection: PaperSelection,
    sources: Iterable[tuple[Path, str]],
    controlled_generated: Mapping[Path, str],
) -> None:
    if selection.event_count != 0:
        return
    for path, content in sources:
        if controlled_generated.get(path) == content:
            continue
        try:
            reject_empirical_claims(selection.event_count, (content,))
        except PaperBuildError as error:
            raise PaperBuildError(
                f"zero-event catalogue rejects empirical prose in {path.name}"
            ) from error


def _generated_literature(selection: PaperSelection) -> str:
    citations = ",".join(selection.citation_keys)
    return "\n".join(
        (
            "% GENERATED: do not edit; generated by ratchet.reporting.paper.",
            r"\section{Reviewed literature}",
            "This paper reports literature claims only, not measurements from this workspace. "
            + rf"The reviewed primary-source set is \cite{{{citations}}}.",
            "The sources motivate future, hardware-gated investigation of IO-aware attention, "
            "compiler baselines, target-specific autotuning, and Intel XPU programming paths.",
            "They do not establish performance for this project or its unavailable runtime.",
            "",
        )
    )


def _generated_no_run(selection: PaperSelection) -> str:
    if selection.event_count != 0:
        raise PaperBuildError(
            "IB-09 survey scaffold only supports an empty event catalogue"
        )
    return "\n".join(
        (
            "% GENERATED: do not edit; generated by ratchet.reporting.paper.",
            r"\section{Evidence status}",
            "The immutable catalogue contains zero experiment events. ENV-0001 records an "
            "unavailable PyTorch XPU runtime, so this build generated no candidate, performed "
            "no compilation, and collected no correctness, timing, memory, trace, or speedup result.",
            "Consequently this document is a reproducible literature survey scaffold, not an "
            "empirical optimization report.",
            "",
        )
    )


def _generated_catalogue(selection: PaperSelection) -> str:
    return "\n".join(
        (
            "% GENERATED: do not edit; generated by ratchet.reporting.paper.",
            r"\section{Catalogue summary}",
            r"\begin{center}",
            r"\begin{tabular}{lr}",
            r"\hline",
            r"Catalogue fact & Value \\",
            r"\hline",
            rf"Immutable experiment events & {selection.event_count} \\",
            rf"Reviewed primary sources & {len(selection.citation_keys)} \\",
            r"\hline",
            r"\end{tabular}",
            r"\end{center}",
            "",
        )
    )


def _generated_evidence_boundary(selection: PaperSelection) -> str:
    """Render the verified inputs and the zero-event outcome as a local TeX figure."""

    if selection.event_count != 0:
        raise PaperBuildError(
            "IB-09 survey scaffold only supports an empty event catalogue"
        )
    return "\n".join(
        (
            "% GENERATED: do not edit; generated by ratchet.reporting.paper.",
            r"\begin{figure}[ht]",
            r"\centering",
            r"\begin{tabular}{c}",
            r"\hline",
            rf"{len(selection.citation_keys)} reviewed primary sources \\\\",
            r"\hline",
            r"Verified immutable archive projection \\\\",
            r"\hline",
            rf"{selection.event_count} experiment events \\\\",
            r"\hline",
            r"Literature synthesis only; no empirical conclusions \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\caption{Evidence boundary for this literature-only build.}",
            r"\end{figure}",
            "",
        )
    )


def _generated_data(selection: PaperSelection) -> dict[str, object]:
    return {
        "schema_version": 1,
        "selection": "reviewed_literature_and_catalogue_only",
        "citation_keys": list(selection.citation_keys),
        "projection_id": selection.projection_id,
        "event_count": selection.event_count,
        "empirical_claims_permitted": selection.event_count > 0,
        "evidence_boundary": {
            "reviewed_primary_sources": len(selection.citation_keys),
            "experiment_events": selection.event_count,
            "conclusion_kind": "literature_synthesis_only",
        },
    }


def _write_if_changed(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def generate_sources(root: Path | None = None) -> PaperSelection:
    """Regenerate only marked generated paper inputs from facts and reviewed literature."""

    project = root or repository_root()
    selection = select_paper_content(project)
    paper = project / "research" / "paper"
    generated = {
        (paper / "generated" / "literature.tex").resolve(): _generated_literature(
            selection
        ),
        (paper / "generated" / "catalogue.tex").resolve(): _generated_catalogue(
            selection
        ),
        (paper / "generated" / "evidence_boundary.tex").resolve(): (
            _generated_evidence_boundary(selection)
        ),
        (paper / "generated" / "no_run.tex").resolve(): _generated_no_run(selection),
    }
    sources = _included_prose(paper, generated)
    _validate_tex_dependencies(paper, sources)
    _validate_no_run_prose(selection, sources, generated)
    _write_if_changed(
        paper / "generated" / "catalogue.json", _canonical(_generated_data(selection))
    )
    _write_if_changed(
        paper / "generated" / "literature.tex",
        generated[(paper / "generated" / "literature.tex").resolve()].encode(),
    )
    _write_if_changed(
        paper / "generated" / "catalogue.tex",
        generated[(paper / "generated" / "catalogue.tex").resolve()].encode(),
    )
    _write_if_changed(
        paper / "generated" / "evidence_boundary.tex",
        generated[(paper / "generated" / "evidence_boundary.tex").resolve()].encode(),
    )
    _write_if_changed(
        paper / "generated" / "no_run.tex",
        generated[(paper / "generated" / "no_run.tex").resolve()].encode(),
    )
    return selection


def build_paper(root: Path | None = None, *, tectonic: str = "tectonic") -> Path:
    """Regenerate sources and build ``latest.pdf`` with Tectonic."""

    project = root or repository_root()
    generate_sources(project)
    paper = project / "research" / "paper"
    executable = shutil.which(tectonic)
    if executable is None:
        raise PaperBuildError("Tectonic is required to build research/paper/latest.pdf")
    with tempfile.TemporaryDirectory(
        prefix=".paper-build-", dir=paper
    ) as build_directory:
        build = Path(build_directory)
        try:
            subprocess.run(
                [
                    executable,
                    "-X",
                    "compile",
                    "--untrusted",
                    "--only-cached",
                    "--outdir",
                    str(build),
                    "main.tex",
                ],
                cwd=paper,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ | {"SOURCE_DATE_EPOCH": "0"},
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stdout or "").strip()[-4_000:]
            raise PaperBuildError(
                "Tectonic failed in --only-cached mode; install Tectonic and prime its "
                f"local bundle before building the paper.\n{detail}"
            ) from error
        built = build / "main.pdf"
        if not built.is_file() or not built.read_bytes().startswith(b"%PDF-"):
            raise PaperBuildError("Tectonic did not produce a valid PDF")
        latest = paper / "latest.pdf"
        _write_if_changed(latest, built.read_bytes())
    return latest


def generated_digest(root: Path | None = None) -> str:
    """Return a stable digest over generated data and sections."""

    project = root or repository_root()
    paper = project / "research" / "paper" / "generated"
    return sha256(
        b"".join(
            (paper / name).read_bytes()
            for name in (
                "catalogue.json",
                "catalogue.tex",
                "evidence_boundary.tex",
                "literature.tex",
                "no_run.tex",
            )
        )
    ).hexdigest()
