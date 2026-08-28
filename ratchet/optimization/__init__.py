"""Public contracts for human- and agent-directed optimization policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class HypothesisSource(str, Enum):
    HUMAN = "human"
    LITERATURE = "literature"
    PROPOSER = "proposer"


@dataclass(frozen=True, slots=True)
class Hypothesis:
    hypothesis_id: str
    source: HypothesisSource
    statement: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.statement:
            raise ValueError("hypothesis fields must not be empty")


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    request_id: str
    hypothesis: Hypothesis

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")


class OptimizationController(Protocol):
    def request(self, request: OptimizationRequest) -> None: ...
