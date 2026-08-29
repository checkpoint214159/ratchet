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
        "rule",
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
class NoRunPaperEvidence:
    """One verified no-run event rendered as evidence, never as a performance result."""

    event_id: str
    experiment_id: str
    environment_id: str
    intended_protocol: str
    stop_reason: str
    hypothesis: str
    literature_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PaperSelection:
    """Reviewed literature and one verified archive projection for a paper build."""

    citation_keys: tuple[str, ...]
    projection_id: str
    event_count: int
    empirical_event_count: int
    no_run_events: tuple[NoRunPaperEvidence, ...]


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


def _no_run_evidence(payload: Mapping[str, object]) -> NoRunPaperEvidence:
    required = (
        "event_id",
        "experiment_id",
        "environment_id",
        "intended_protocol",
        "stop_reason",
        "hypothesis",
    )
    if any(not isinstance(payload.get(name), str) for name in required):
        raise PaperBuildError("verified no-run payload has invalid paper fields")
    references = payload.get("literature_refs")
    if not isinstance(references, list) or not all(
        isinstance(reference, str) for reference in references
    ):
        raise PaperBuildError(
            "verified no-run payload has invalid literature references"
        )
    return NoRunPaperEvidence(*(payload[name] for name in required), tuple(references))


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
        projection_data = json.loads(archive.projection_bytes())
    except (ArchiveIntegrityError, ValueError, json.JSONDecodeError) as error:
        raise PaperBuildError("paper requires a verified archive projection") from error
    events = projection_data.get("events")
    if not isinstance(events, list) or len(events) != projection.event_count:
        raise PaperBuildError("verified archive projection has invalid event data")
    no_run_events: list[NoRunPaperEvidence] = []
    empirical_event_count = 0
    for entry in events:
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
            raise PaperBuildError(
                "verified archive projection has invalid event payload"
            )
        payload = entry["payload"]
        kind = payload.get("event_kind")
        if kind == "no_run":
            no_run_events.append(_no_run_evidence(payload))
        elif kind == "empirical":
            empirical_event_count += 1
        else:
            raise PaperBuildError(
                "verified archive projection has an unknown event kind"
            )
    if not set(
        reference for event in no_run_events for reference in event.literature_refs
    ) <= set(citations):
        raise PaperBuildError("no-run literature references must be reviewed citations")
    return PaperSelection(
        citations,
        projection.projection_id,
        projection.event_count,
        empirical_event_count,
        tuple(no_run_events),
    )


def reject_empirical_claims(empirical_event_count: int, claims: Iterable[str]) -> None:
    """Reject empirical-result language unless verified empirical events exist."""

    if empirical_event_count != 0:
        return
    for claim in claims:
        if _EMPIRICAL_LANGUAGE.search(claim) or _COMPARATIVE_LANGUAGE.search(claim):
            raise PaperBuildError(
                "catalogue without empirical events cannot contain empirical claims"
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
    if selection.empirical_event_count != 0:
        return
    for path, content in sources:
        if controlled_generated.get(path) == content:
            continue
        try:
            reject_empirical_claims(selection.empirical_event_count, (content,))
        except PaperBuildError as error:
            raise PaperBuildError(
                f"catalogue without empirical events rejects empirical prose in {path.name}"
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


def _paper_text(value: str) -> str:
    if any(character in value for character in r"\\{}%$#&_~^"):
        raise PaperBuildError("verified paper fact contains unsupported TeX characters")
    return value


def _generated_no_run(selection: PaperSelection) -> str:
    if not selection.no_run_events:
        return "\n".join(
            (
                "% GENERATED: do not edit; generated by ratchet.reporting.paper.",
                r"\section{Evidence status}",
                "The immutable catalogue contains no experiment events. This document is a "
                "reproducible literature survey scaffold, not an empirical optimization report.",
                "",
            )
        )
    event = selection.no_run_events[0]
    return "\n".join(
        (
            "% GENERATED: do not edit; generated by ratchet.reporting.paper.",
            r"\section{Evidence status}",
            f"Verified event {_paper_text(event.event_id)} / {_paper_text(event.experiment_id)} "
            f"records a no-run decision for {_paper_text(event.environment_id)}: "
            f"{_paper_text(event.stop_reason)}.",
            "The controller generated no candidate and recorded no compilation, correctness, "
            "timing, memory, profile, trace, counter, comparison, speedup, or current-best result.",
            "This is evidence of an unavailable execution environment, not a failed or slow implementation.",
            "",
        )
    ) + _generated_next_hypothesis(selection)


def _generated_next_hypothesis(selection: PaperSelection) -> str:
    if not selection.no_run_events:
        return ""
    event = selection.no_run_events[0]
    citations = ",".join(event.literature_refs)
    return "\n".join(
        (
            "% GENERATED: do not edit; generated by ratchet.reporting.paper.",
            r"\section{Next research direction}",
            "After FG-01 qualifies an Intel Arc/XPU runtime, test the queued protocol "
            f"{_paper_text(event.intended_protocol)}: {_paper_text(event.hypothesis)} "
            rf"Its motivation is traceable to \cite{{{citations}}}.",
            "Until that hardware gate passes, this remains a protocol hypothesis rather than a project result.",
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
    """Render the verified source and current evidence boundary as a local TeX figure."""

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
            rf"{len(selection.no_run_events)} no-run event(s); {selection.empirical_event_count} empirical event(s) \\\\",
            r"\hline",
            r"Literature synthesis only; no empirical conclusions \\\\",
            r"\hline",
            r"\end{tabular}",
            r"\caption{Evidence boundary for this literature-only build.}",
            r"\end{figure}",
            "",
        )
    )


def _evidence_bar(count: int, largest: int) -> str:
    """A proportional horizontal bar drawn with the LaTeX \\rule primitive only.

    Width is integer points scaled against the largest series value, so the figure is
    byte-deterministic and needs no drawing package. A zero count renders as a zero-width
    bar, which is the honest depiction of "no events of this kind" rather than a hidden row.
    """

    span = max(largest, 1)
    width_pt = round(120 * count / span)
    return rf"\rule{{{width_pt}pt}}{{6pt}}"


def _generated_evidence_figure(selection: PaperSelection) -> str:
    """Render the evidence composition as a deterministic bar chart.

    The chart answers one research question: what is this build's evidence made of? With
    only a no-run event it shows reviewed sources against zero empirical events, which is
    the paper's central boundary. The SAME generator produces a meaningful speedup or
    accepted/rejected chart once empirical events exist, so it is not decoration.
    """

    reviewed = len(selection.citation_keys)
    no_run = len(selection.no_run_events)
    empirical = selection.empirical_event_count
    largest = max(reviewed, no_run, empirical)
    rows = (
        ("Reviewed primary sources", reviewed),
        ("No-run evidence events", no_run),
        ("Empirical result events", empirical),
    )
    body = [
        "% GENERATED: do not edit; generated by ratchet.reporting.paper.",
        r"\begin{figure}[ht]",
        r"\centering",
        r"\begin{tabular}{lrl}",
        r"\hline",
        r"Evidence category & Count & Proportion \\",
        r"\hline",
    ]
    for label, count in rows:
        body.append(rf"{label} & {count} & {_evidence_bar(count, largest)} \\")
    body.extend(
        (
            r"\hline",
            r"\end{tabular}",
            r"\caption{Evidence composition of the current build (bars are proportional "
            r"to counts). Empirical result events remain zero until a hardware gate passes.}",
            r"\end{figure}",
            "",
        )
    )
    return "\n".join(body)


def _generated_data(selection: PaperSelection) -> dict[str, object]:
    return {
        "schema_version": 1,
        "selection": "reviewed_literature_and_catalogue_only",
        "citation_keys": list(selection.citation_keys),
        "projection_id": selection.projection_id,
        "event_count": selection.event_count,
        "empirical_event_count": selection.empirical_event_count,
        "empirical_claims_permitted": selection.empirical_event_count > 0,
        "no_run_events": [
            {
                "event_id": event.event_id,
                "experiment_id": event.experiment_id,
                "environment_id": event.environment_id,
                "intended_protocol": event.intended_protocol,
                "literature_refs": list(event.literature_refs),
                "stop_reason": event.stop_reason,
            }
            for event in selection.no_run_events
        ],
        "evidence_boundary": {
            "reviewed_primary_sources": len(selection.citation_keys),
            "experiment_events": selection.event_count,
            "no_run_events": len(selection.no_run_events),
            "empirical_events": selection.empirical_event_count,
            "conclusion_kind": "literature_synthesis_with_no_run_evidence",
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
        (paper / "generated" / "evidence_figure.tex").resolve(): (
            _generated_evidence_figure(selection)
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
        paper / "generated" / "evidence_figure.tex",
        generated[(paper / "generated" / "evidence_figure.tex").resolve()].encode(),
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
