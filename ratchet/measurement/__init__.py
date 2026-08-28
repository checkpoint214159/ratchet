"""Public contracts for correctness-first measurement orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ratchet.backends import (
    BackendIdentity,
    MemoryEvidence,
    TimingEvidence,
)
from ratchet.evaluation import EvaluationCase
from ratchet.models import ModelDescriptor


class MeasurementStatus(str, Enum):
    OK = "ok"
    INCORRECT = "incorrect"
    COMPILE_ERROR = "compile_error"
    TIMEOUT = "timeout"
    CRASH = "crash"
    UNAVAILABLE = "unavailable"


class EvidenceClassification(str, Enum):
    """Archive-facing label that keeps no-run and synthetic facts distinct."""

    NO_RUN = "no_run"
    SYNTHETIC = "synthetic"


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
    classification: EvidenceClassification = EvidenceClassification.NO_RUN

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.classification is EvidenceClassification.NO_RUN:
            if self.status is not MeasurementStatus.UNAVAILABLE:
                raise ValueError("no-run evidence requires unavailable status")
            if (
                self.correctness_passed
                or self.timing is not None
                or self.memory is not None
            ):
                raise ValueError("no-run evidence cannot contain empirical fields")
            if not self.diagnostic:
                raise ValueError("no-run evidence requires a diagnostic")
        elif self.status is MeasurementStatus.UNAVAILABLE:
            raise ValueError("unavailable status requires no-run classification")
        elif self.status is MeasurementStatus.OK:
            if not self.correctness_passed or self.timing is None:
                raise ValueError("successful evidence requires correctness and timing")
        else:
            if self.correctness_passed:
                raise ValueError(
                    "non-successful measurement evidence cannot pass correctness"
                )
            if self.timing is not None or self.memory is not None:
                raise ValueError(
                    "failed measurement evidence must not include timing or memory"
                )
            if not self.diagnostic:
                raise ValueError("failed measurement evidence requires a diagnostic")


class MeasurementHarness(Protocol):
    def measure(self, request: MeasurementRequest) -> MeasurementEvidence: ...


class SubprocessMeasurementHarness:
    """Public no-run gate for this unqualified-accelerator build."""

    def measure(self, request: MeasurementRequest) -> MeasurementEvidence:
        return MeasurementEvidence(
            request_id=request.request_id,
            status=MeasurementStatus.UNAVAILABLE,
            correctness_passed=False,
            timing=None,
            memory=None,
            diagnostic="production measurement is disabled by the no-accelerator gate",
        )


__all__ = [
    "EvidenceClassification",
    "MeasurementEvidence",
    "MeasurementHarness",
    "MeasurementRequest",
    "MeasurementStatus",
    "SubprocessMeasurementHarness",
]
