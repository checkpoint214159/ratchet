"""Test-only synthetic measurement engine; never import this from production code."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context
from queue import Empty
from traceback import format_exc
from typing import Protocol

from ratchet.backends import (
    AcceleratorBackend,
    AvailabilityState,
    BackendUnavailableError,
    TimingConfiguration,
)
from ratchet.evaluation import CorrectnessResult
from ratchet.measurement import (
    EvidenceClassification,
    MeasurementEvidence,
    MeasurementRequest,
    MeasurementStatus,
)


class BoundExecution(Protocol):
    def check_correctness(self) -> CorrectnessResult: ...

    def run(self) -> None: ...


class RequestBoundWorkload(Protocol):
    def bind(self, request: MeasurementRequest) -> BoundExecution: ...


@dataclass(frozen=True, slots=True)
class _WorkerResult:
    evidence: MeasurementEvidence


def _synthetic_failure(
    request: MeasurementRequest, status: MeasurementStatus, diagnostic: str
) -> MeasurementEvidence:
    return MeasurementEvidence(
        request_id=request.request_id,
        status=status,
        correctness_passed=False,
        timing=None,
        memory=None,
        diagnostic=diagnostic,
        classification=EvidenceClassification.SYNTHETIC,
    )


def _run_synthetic_measurement(
    result_queue: object,
    request: MeasurementRequest,
    workload: RequestBoundWorkload,
    backend: AcceleratorBackend,
    timing_configuration: TimingConfiguration,
) -> None:
    try:
        execution = workload.bind(request)
        correctness_result = execution.check_correctness()
        if not correctness_result.passed:
            evidence = _synthetic_failure(
                request,
                MeasurementStatus.INCORRECT,
                correctness_result.diagnostic or "correctness failed",
            )
        else:
            backend.synchronize()
            backend.reset_memory_stats()
            timing = backend.time(execution.run, timing_configuration)
            backend.synchronize()
            memory = backend.memory_stats()
            evidence = MeasurementEvidence(
                request_id=request.request_id,
                status=MeasurementStatus.OK,
                correctness_passed=True,
                timing=timing,
                memory=memory,
                classification=EvidenceClassification.SYNTHETIC,
            )
    except BackendUnavailableError as error:
        evidence = _synthetic_failure(request, MeasurementStatus.CRASH, error.reason)
    except BaseException:
        evidence = _synthetic_failure(request, MeasurementStatus.CRASH, format_exc())
    result_queue.put(_WorkerResult(evidence))  # type: ignore[attr-defined]


class SyntheticSubprocessHarness:
    """Synthetic-only process harness for deterministic measurement-contract tests."""

    def __init__(
        self,
        workload: RequestBoundWorkload,
        backend: AcceleratorBackend,
        timing_configuration: TimingConfiguration,
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._workload = workload
        self._backend = backend
        self._timing_configuration = timing_configuration
        self._timeout_seconds = timeout_seconds

    def measure(self, request: MeasurementRequest) -> MeasurementEvidence:
        try:
            capabilities = self._backend.capabilities()
            identity = self._backend.probe()
        except BackendUnavailableError as error:
            return _synthetic_failure(request, MeasurementStatus.CRASH, error.reason)
        if capabilities.availability is AvailabilityState.UNAVAILABLE:
            return _synthetic_failure(
                request,
                MeasurementStatus.CRASH,
                "backend runtime or device is unavailable",
            )
        if identity != request.backend:
            return _synthetic_failure(
                request,
                MeasurementStatus.CRASH,
                "backend identity does not match the measurement request",
            )

        process: object | None = None
        try:
            context = get_context("spawn")
            result_queue = context.Queue(maxsize=1)
            process = context.Process(
                target=_run_synthetic_measurement,
                args=(
                    result_queue,
                    request,
                    self._workload,
                    self._backend,
                    self._timing_configuration,
                ),
            )
            process.start()
            process.join(self._timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join()
                return _synthetic_failure(
                    request,
                    MeasurementStatus.TIMEOUT,
                    f"measurement exceeded {self._timeout_seconds:g} seconds",
                )
            result = result_queue.get(timeout=0.1)
        except Empty:
            return _synthetic_failure(
                request,
                MeasurementStatus.CRASH,
                f"measurement subprocess exited without evidence (exit code {process.exitcode})",
            )
        except BaseException:
            if process is not None:
                try:
                    if process.is_alive():  # type: ignore[union-attr]
                        process.terminate()  # type: ignore[union-attr]
                        process.join()  # type: ignore[union-attr]
                except BaseException:
                    pass
            return _synthetic_failure(
                request,
                MeasurementStatus.CRASH,
                "parent subprocess orchestration failed: " + format_exc(),
            )
        return result.evidence  # type: ignore[union-attr]
