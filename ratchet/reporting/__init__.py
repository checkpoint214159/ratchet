"""Public contracts for deterministic research synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ratchet.experiments import CatalogueProjection

from .paper import PaperBuildError, PaperSelection, build_paper, generate_sources


@dataclass(frozen=True, slots=True)
class ReportRequest:
    report_id: str
    projection: CatalogueProjection

    def __post_init__(self) -> None:
        if not self.report_id:
            raise ValueError("report_id must not be empty")


@dataclass(frozen=True, slots=True)
class ReportArtifact:
    report_id: str
    content_digest: str

    def __post_init__(self) -> None:
        if not self.report_id or not self.content_digest:
            raise ValueError("report artifact fields must not be empty")


class ReportBuilder(Protocol):
    def build(self, request: ReportRequest) -> ReportArtifact: ...


__all__ = [
    "PaperBuildError",
    "PaperSelection",
    "ReportArtifact",
    "ReportBuilder",
    "ReportRequest",
    "build_paper",
    "generate_sources",
]
