"""Public contracts for weight-compatible transformer candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Identity and provenance for a model selectable at the evaluator seam."""

    model_id: str
    family: str
    source_hash: str

    def __post_init__(self) -> None:
        if not all((self.model_id, self.family, self.source_hash)):
            raise ValueError("model descriptor fields must not be empty")


class ModelFactory(Protocol):
    """Builds a model internally; vendor objects never enter the public contract."""

    def create(self, descriptor: ModelDescriptor) -> object: ...
