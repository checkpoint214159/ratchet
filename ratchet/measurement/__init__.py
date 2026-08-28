"""Public contracts for correctness-first measurement orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ratchet.backends import BackendIdentity, MemoryEvidence, TimingEvidence
from ratchet.evaluation import EvaluationCase
from ratchet.models import ModelDescriptor


class MeasurementStatus(str, Enum):
    OK = "ok"
    INCORRECT = "incorrect"
    COMPILE_ERROR = "compile_error"
    TIMEOUT = "timeout"
    CRASH = "crash"


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    request_id: str
    candidate: ModelDescriptor
    evaluation_case: EvaluationCase
    backend: BackendIdentity

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")


@dataclass(frozen=True, slots=True)
class MeasurementEvidence:
    request_id: str
    status: MeasurementStatus
    correctness_passed: bool
    timing: TimingEvidence | None
    memory: MemoryEvidence | None
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.status is MeasurementStatus.OK:
            if not self.correctness_passed or self.timing is None:
                raise ValueError("successful evidence requires correctness and timing")
        else:
            if self.status is MeasurementStatus.INCORRECT and self.correctness_passed:
                raise ValueError(
                    "incorrect measurement evidence cannot pass correctness"
                )
            if self.timing is not None:
                raise ValueError("failed measurement evidence must not include timing")


class MeasurementHarness(Protocol):
    def measure(self, request: MeasurementRequest) -> MeasurementEvidence: ...
