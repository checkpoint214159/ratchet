"""CPU-only construction tests for public bounded-context contracts."""

from dataclasses import FrozenInstanceError

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
from ratchet.dispatch import DispatchDecision, DispatchRequest
from ratchet.evaluation import (
    CorrectnessPolicy,
    EvaluationCase,
    TransformerConfiguration,
)
from ratchet.experiments import (
    EMPIRICAL_EVENT,
    CatalogueProjection,
    EventId,
    ExperimentEvent,
    ExperimentId,
)
from ratchet.measurement import (
    EvidenceClassification,
    MeasurementEvidence,
    MeasurementRequest,
    MeasurementStatus,
)
from ratchet.models import ModelDescriptor
from ratchet.optimization import Hypothesis, HypothesisSource, OptimizationRequest
from ratchet.reporting import ReportArtifact, ReportRequest


def _identity() -> BackendIdentity:
    return BackendIdentity(
        backend=BackendKind.XPU,
        device_name="Intel Arc",
        driver_version="driver",
        runtime_version="runtime",
        framework_version="framework",
        compiler_version="compiler",
    )


def _case() -> EvaluationCase:
    return EvaluationCase(
        case_id="default",
        configuration=TransformerConfiguration(8, 128, 512, 8, 2048, 6, False),
        dtype="float32",
        seed=1234,
    )


def test_evaluation_and_model_contracts_are_immutable_and_validated():
    configuration = TransformerConfiguration(2, 127, 256, 8, 1024, 2, True)
    candidate = ModelDescriptor("candidate-1", "transformer", "source-digest")

    assert configuration.model_width // configuration.head_count == 32
    assert CorrectnessPolicy(0.002, 0.02).absolute_tolerance == 0.002
    assert candidate.model_id == "candidate-1"
    with pytest.raises(FrozenInstanceError):
        candidate.model_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfiguration(1, 1, 10, 3, 1, 1, False)


def test_backend_contracts_are_vendor_neutral_and_validated():
    identity = _identity()
    capabilities = BackendCapabilities(
        availability=AvailabilityState.AVAILABLE,
        validation=ValidationState.UNVALIDATED,
        supports_events=False,
        supports_compilation=True,
        supports_peak_memory=False,
        supported_dtypes=("float32", "bfloat16"),
    )
    assert identity.backend is BackendKind.XPU
    assert capabilities.supported_dtypes == ("float32", "bfloat16")
    assert TimingConfiguration(0, 1).measured_calls == 1
    assert MemoryEvidence(None, None).peak_allocated_bytes is None
    executable = object()
    compiled = CompiledModel(
        executable,
        BackendKind.XPU,
        CompilationPolicy("eager", False),
        compiler="eager",
    )
    assert compiled.model is executable
    with pytest.raises(ValueError, match="unique"):
        BackendCapabilities(
            AvailabilityState.AVAILABLE,
            ValidationState.AVAILABLE,
            True,
            True,
            True,
            ("float32", "float32"),
        )
    with pytest.raises(ValueError, match="positive"):
        TimingEvidence("event", (), True)


@pytest.mark.parametrize(
    (
        "validation",
        "supports_events",
        "supports_compilation",
        "supports_peak_memory",
        "supported_dtypes",
        "message",
    ),
    [
        (ValidationState.AVAILABLE, False, False, False, (), "require unavailable"),
        (ValidationState.UNAVAILABLE, True, False, False, (), "support capabilities"),
        (ValidationState.UNAVAILABLE, False, True, False, (), "support capabilities"),
        (ValidationState.UNAVAILABLE, False, False, True, (), "support capabilities"),
        (ValidationState.UNAVAILABLE, False, False, False, ("float32",), "dtypes"),
    ],
)
def test_unavailable_backend_capabilities_reject_each_contradiction(
    validation: ValidationState,
    supports_events: bool,
    supports_compilation: bool,
    supports_peak_memory: bool,
    supported_dtypes: tuple[str, ...],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        BackendCapabilities(
            AvailabilityState.UNAVAILABLE,
            validation,
            supports_events,
            supports_compilation,
            supports_peak_memory,
            supported_dtypes,
        )


def test_measurement_experiment_dispatch_optimization_and_reporting_contracts():
    candidate = ModelDescriptor("candidate-1", "transformer", "source-digest")
    request = MeasurementRequest("request-1", candidate, _case(), _identity())
    timing = TimingEvidence("event", (100,), True)
    memory = MemoryEvidence(1, 2)
    evidence = MeasurementEvidence(
        "request-1",
        MeasurementStatus.OK,
        True,
        timing,
        memory,
        classification=EvidenceClassification.SYNTHETIC,
    )
    event = ExperimentEvent(
        EventId("EVT-000001"), ExperimentId("EXP-0001"), 0, EMPIRICAL_EVENT, "a" * 64
    )
    projection = CatalogueProjection("b" * 64, 1)

    assert request.candidate is candidate
    assert evidence.timing is timing
    assert event.event_id.value == "EVT-000001"
    assert DispatchRequest("default", _identity()).regime == "default"
    assert DispatchDecision("candidate-1", False, "untuned fallback").is_tuned is False
    assert OptimizationRequest(
        "opt-1", Hypothesis("hyp-1", HypothesisSource.HUMAN, "test")
    )
    assert ReportRequest("report-1", projection).projection is projection
    assert ReportArtifact("report-1", "digest").content_digest == "digest"
    with pytest.raises(ValueError, match="timing or memory"):
        MeasurementEvidence(
            "request-1",
            MeasurementStatus.INCORRECT,
            False,
            timing,
            None,
            classification=EvidenceClassification.SYNTHETIC,
        )
    with pytest.raises(ValueError, match="cannot pass correctness"):
        MeasurementEvidence(
            "request-1",
            MeasurementStatus.INCORRECT,
            True,
            None,
            None,
            classification=EvidenceClassification.SYNTHETIC,
        )
