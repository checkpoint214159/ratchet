"""CPU-only contracts for correctness-first subprocess measurement orchestration."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import pytest

from ratchet.backends import (
    AvailabilityState,
    BackendCapabilities,
    BackendIdentity,
    BackendKind,
    CompilationPolicy,
    CompiledModel,
    MemoryEvidence,
    TimingConfiguration,
    TimingEvidence,
    ValidationState,
)
from ratchet.evaluation import (
    CorrectnessResult,
    EvaluationCase,
    TransformerConfiguration,
)
from ratchet.measurement import (
    EvidenceClassification,
    MeasurementEvidence,
    MeasurementRequest,
    MeasurementStatus,
    SubprocessMeasurementHarness,
)
from ratchet.models import ModelDescriptor
from tests.fixtures.measurement import (
    SyntheticSubprocessHarness,
    _run_synthetic_measurement,
)


def _request() -> MeasurementRequest:
    return MeasurementRequest(
        request_id="request-1",
        candidate=ModelDescriptor("candidate-1", "transformer", "source-digest"),
        evaluation_case=EvaluationCase(
            "default",
            TransformerConfiguration(1, 8, 16, 2, 32, 1, False),
            "float32",
            1234,
        ),
        backend=BackendIdentity(
            BackendKind.XPU,
            "fake xpu",
            "fake driver",
            "fake runtime",
            "fake framework",
            "fake compiler",
        ),
    )


class FakeBackend:
    def __init__(
        self,
        events: Any,
        availability: AvailabilityState,
        validation: ValidationState = ValidationState.UNVALIDATED,
        identity: BackendIdentity | None = None,
    ) -> None:
        self.events = events
        self.availability = availability
        self.validation = validation
        self.identity = identity

    def probe(self) -> BackendIdentity:
        return self.identity or _request().backend

    def capabilities(self) -> BackendCapabilities:
        if self.availability is AvailabilityState.UNAVAILABLE:
            return BackendCapabilities(
                AvailabilityState.UNAVAILABLE,
                ValidationState.UNAVAILABLE,
                False,
                False,
                False,
                (),
            )
        return BackendCapabilities(
            AvailabilityState.AVAILABLE,
            self.validation,
            True,
            True,
            True,
            ("float32",),
        )

    def synchronize(self) -> None:
        self.events.append(("synchronize", os.getpid()))

    def time(
        self, operation: Callable[[], None], configuration: TimingConfiguration
    ) -> TimingEvidence:
        self.events.append(("time", os.getpid(), configuration.measured_calls))
        operation()
        return TimingEvidence("fake_event", (100,), True)

    def reset_memory_stats(self) -> None:
        self.events.append(("reset_memory", os.getpid()))

    def memory_stats(self) -> MemoryEvidence:
        self.events.append(("memory", os.getpid()))
        return MemoryEvidence(10, 20)

    def compile_model(self, model: object, policy: CompilationPolicy) -> CompiledModel:
        return CompiledModel(model, BackendKind.XPU, policy, "fake")


class Workload:
    def __init__(self, events: Any, outcome: str = "passed") -> None:
        self.events = events
        self.outcome = outcome

    def bind(self, request: MeasurementRequest) -> BoundExecution:
        self.events.append(("bind_operation", os.getpid(), request.request_id))
        return BoundExecution(self, request)


class BoundExecution:
    def __init__(self, workload: Workload, request: MeasurementRequest) -> None:
        self.workload = workload
        self.request = request

    def check_correctness(self) -> CorrectnessResult:
        self.workload.events.append(("correct", os.getpid(), self.request.request_id))
        if self.workload.outcome == "failed":
            return CorrectnessResult(False, "output mismatch")
        if self.workload.outcome == "crash":
            raise RuntimeError(f"worker failed for {self.request.request_id}")
        if self.workload.outcome == "hang":
            time.sleep(1)
        return CorrectnessResult(True)

    def run(self) -> None:
        if self.workload.outcome == "failed":
            raise AssertionError(
                "timed execution must not run after failed correctness"
            )
        self.workload.events.append(("operation", os.getpid(), self.request.request_id))


def _harness(
    workload: Workload,
    backend: FakeBackend,
    *,
    timeout_seconds: float = 2,
) -> SyntheticSubprocessHarness:
    return SyntheticSubprocessHarness(
        workload,
        backend,
        TimingConfiguration(0, 1),
        timeout_seconds,
    )


class InlineResultQueue:
    def __init__(self) -> None:
        self.result: object | None = None

    def put(self, result: object) -> None:
        self.result = result


def test_worker_enforces_correctness_before_timing_and_memory_lifecycle():
    events: list[tuple[object, ...]] = []
    result_queue = InlineResultQueue()

    _run_synthetic_measurement(
        result_queue,
        _request(),
        Workload(events),
        FakeBackend(events, AvailabilityState.AVAILABLE),
        TimingConfiguration(0, 1),
    )

    evidence = result_queue.result.evidence  # type: ignore[union-attr]
    assert evidence.status is MeasurementStatus.OK
    assert evidence.correctness_passed is True
    assert evidence.classification is EvidenceClassification.SYNTHETIC
    assert evidence.timing is not None
    assert evidence.memory == MemoryEvidence(10, 20)
    assert [event[0] for event in events] == [
        "bind_operation",
        "correct",
        "synchronize",
        "reset_memory",
        "time",
        "operation",
        "synchronize",
        "memory",
    ]
    assert [event[0] for event in events].count("bind_operation") == 1


def test_incorrect_candidates_do_not_start_timing_or_memory_collection():
    events: list[tuple[object, ...]] = []
    harness = _harness(
        Workload(events, "failed"),
        FakeBackend(events, AvailabilityState.AVAILABLE),
    )

    evidence = harness.measure(_request())

    assert evidence.status is MeasurementStatus.INCORRECT
    assert evidence.correctness_passed is False
    assert evidence.timing is None
    assert evidence.memory is None
    assert evidence.classification is EvidenceClassification.SYNTHETIC


def test_synthetic_harness_records_unavailable_backend_as_synthetic_failure():
    events: list[tuple[object, ...]] = []
    harness = _harness(
        Workload(events),
        FakeBackend(events, AvailabilityState.UNAVAILABLE),
    )

    evidence = harness.measure(_request())

    assert evidence.status is MeasurementStatus.CRASH
    assert evidence.correctness_passed is False
    assert evidence.timing is None
    assert evidence.memory is None
    assert evidence.classification is EvidenceClassification.SYNTHETIC
    assert events == []


def test_worker_crash_is_returned_as_non_timing_evidence():
    events: list[tuple[object, ...]] = []
    harness = _harness(
        Workload(events, "crash"),
        FakeBackend(events, AvailabilityState.AVAILABLE),
    )

    evidence = harness.measure(_request())

    assert evidence.status is MeasurementStatus.CRASH
    assert evidence.timing is None
    assert evidence.memory is None
    assert evidence.diagnostic is not None
    assert "RuntimeError" in evidence.diagnostic
    assert evidence.classification is EvidenceClassification.SYNTHETIC


def test_worker_timeout_is_recorded_without_timing():
    events: list[tuple[object, ...]] = []
    harness = _harness(
        Workload(events, "hang"),
        FakeBackend(events, AvailabilityState.AVAILABLE),
        timeout_seconds=0.05,
    )

    evidence = harness.measure(_request())

    assert evidence.status is MeasurementStatus.TIMEOUT
    assert evidence.timing is None
    assert evidence.memory is None
    assert evidence.classification is EvidenceClassification.SYNTHETIC


def test_mismatched_backend_identity_is_rejected_before_synthetic_work_starts():
    events: list[tuple[object, ...]] = []
    mismatched_identity = BackendIdentity(
        BackendKind.XPU,
        "other xpu",
        "fake driver",
        "fake runtime",
        "fake framework",
        "fake compiler",
    )
    harness = _harness(
        Workload(events),
        FakeBackend(events, AvailabilityState.AVAILABLE, identity=mismatched_identity),
    )

    evidence = harness.measure(_request())

    assert evidence.status is MeasurementStatus.CRASH
    assert "does not match" in evidence.diagnostic
    assert evidence.classification is EvidenceClassification.SYNTHETIC
    assert events == []


def test_public_harness_is_unconditionally_no_run():
    events: list[tuple[object, ...]] = []
    harness = SubprocessMeasurementHarness()

    evidence = harness.measure(_request())

    assert evidence.status is MeasurementStatus.UNAVAILABLE
    assert "no-accelerator gate" in evidence.diagnostic
    assert evidence.classification is EvidenceClassification.NO_RUN
    assert events == []


def test_public_measurement_surface_excludes_the_synthetic_engine():
    from ratchet import measurement

    assert "SyntheticSubprocessHarness" not in measurement.__all__
    assert not hasattr(measurement, "SyntheticSubprocessHarness")


def test_parent_subprocess_setup_failure_is_contained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.fixtures import measurement as synthetic_measurement

    class BrokenContext:
        def Queue(self, *, maxsize: int) -> object:
            raise OSError(f"queue setup failed for {maxsize}")

    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(synthetic_measurement, "get_context", lambda _: BrokenContext())
    evidence = _harness(
        Workload(events), FakeBackend(events, AvailabilityState.AVAILABLE)
    ).measure(_request())

    assert evidence.status is MeasurementStatus.CRASH
    assert evidence.correctness_passed is False
    assert evidence.timing is None
    assert evidence.memory is None
    assert "parent subprocess orchestration failed" in evidence.diagnostic
    assert evidence.classification is EvidenceClassification.SYNTHETIC


@pytest.mark.parametrize(
    ("timing", "memory"),
    [
        (TimingEvidence("event", (100,), True), None),
        (None, MemoryEvidence(1, 2)),
    ],
)
def test_non_successful_evidence_rejects_timing_or_memory(
    timing: TimingEvidence | None, memory: MemoryEvidence | None
) -> None:
    with pytest.raises(ValueError, match="timing or memory"):
        MeasurementEvidence(
            "request-1",
            MeasurementStatus.CRASH,
            False,
            timing,
            memory,
            diagnostic="worker failed",
            classification=EvidenceClassification.SYNTHETIC,
        )


def test_evidence_classification_has_no_empirical_variant_or_invalid_no_run():
    assert set(EvidenceClassification) == {
        EvidenceClassification.NO_RUN,
        EvidenceClassification.SYNTHETIC,
    }
    with pytest.raises(ValueError, match="no-run evidence requires unavailable"):
        MeasurementEvidence(
            "request-1",
            MeasurementStatus.OK,
            True,
            TimingEvidence("event", (100,), True),
            MemoryEvidence(1, 2),
        )
    with pytest.raises(ValueError, match="unavailable status requires no-run"):
        MeasurementEvidence(
            "request-1",
            MeasurementStatus.UNAVAILABLE,
            False,
            None,
            None,
            diagnostic="synthetic unavailable",
            classification=EvidenceClassification.SYNTHETIC,
        )
