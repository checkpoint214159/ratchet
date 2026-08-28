"""Public contracts for append-only experiment facts and projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExperimentId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("experiment id must not be empty")


@dataclass(frozen=True, slots=True)
class ExperimentEvent:
    experiment_id: ExperimentId
    sequence: int
    kind: str
    payload_digest: str

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence must be non-negative")
        if not self.kind or not self.payload_digest:
            raise ValueError("event kind and payload digest must not be empty")


@dataclass(frozen=True, slots=True)
class CatalogueProjection:
    projection_id: str
    event_count: int

    def __post_init__(self) -> None:
        if not self.projection_id:
            raise ValueError("projection_id must not be empty")
        if self.event_count < 0:
            raise ValueError("event_count must be non-negative")


class ExperimentCatalogue(Protocol):
    def append(self, event: ExperimentEvent) -> None: ...

    def projection(self) -> CatalogueProjection: ...
