"""CPU-only contracts for deterministic evidence-driven dispatch."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ratchet.backends import (
    AvailabilityState,
    BackendCapabilities,
    BackendIdentity,
    BackendKind,
    ValidationState,
)
from ratchet.dispatch import (
    EAGER_IMPLEMENTATION_IDS,
    CandidateDispatchEvidence,
    DispatchProfile,
    DispatchRequest,
    EvidenceDrivenDispatch,
    TunedDispatch,
    UntunedFallback,
)
from ratchet.experiments import CatalogueProjection, FileExperimentArchive

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_DIGEST = "a" * 64
CONFIGURATION_DIGEST = "b" * 64
PROJECTION = CatalogueProjection(
    "c" * 64, 3, ("EVT-000001", "EVT-000002", "EVT-000003")
)


def _identity(
    kind: BackendKind = BackendKind.XPU,
    suffix: str = "a",
    compiler_version: str | None = None,
) -> BackendIdentity:
    return BackendIdentity(
        backend=kind,
        device_name=f"device-{suffix}",
        driver_version=f"driver-{suffix}",
        runtime_version=f"runtime-{suffix}",
        framework_version=f"framework-{suffix}",
        compiler_version=(
            f"compiler-{suffix}" if compiler_version is None else compiler_version
        ),
    )


def _capabilities(
    validation: ValidationState = ValidationState.QUALIFIED,
    *,
    supports_events: bool = True,
    supports_peak_memory: bool = True,
    supported_dtypes: tuple[str, ...] = ("float32",),
) -> BackendCapabilities:
    return BackendCapabilities(
        AvailabilityState.AVAILABLE,
        validation,
        supports_events,
        True,
        supports_peak_memory,
        supported_dtypes,
    )


def _profile(**changes: object) -> DispatchProfile:
    values: dict[str, object] = {
        "regime": "reference-transformer",
        "evaluator_contract_digest": EVALUATOR_DIGEST,
        "benchmark_configuration_digest": CONFIGURATION_DIGEST,
        "dtype": "float32",
    }
    values.update(changes)
    return DispatchProfile(**values)  # type: ignore[arg-type]


def _request(
    profile: DispatchProfile | None = None,
    identity: BackendIdentity | None = None,
    capabilities: BackendCapabilities | None = None,
) -> DispatchRequest:
    return DispatchRequest(
        profile or _profile(),
        identity or _identity(),
        capabilities or _capabilities(),
    )


def _evidence(
    candidate: str,
    request: DispatchRequest,
    *,
    event_id: str = "EVT-000001",
    **changes: object,
) -> CandidateDispatchEvidence:
    values: dict[str, object] = {
        "implementation_id": candidate,
        "source_event_id": event_id,
        "source_projection_id": PROJECTION.projection_id,
        "profile": request.profile,
        "backend_identity": request.backend,
        "correctness_passed": True,
        "synchronized": True,
        "latency_intervals_disjoint": True,
        "paired_speedup_lower_bound": 1.03,
        "peak_memory_increase_ratio": 0.05,
        "dispatch_verified": True,
    }
    values.update(changes)
    return CandidateDispatchEvidence(**values)  # type: ignore[arg-type]


def test_exact_eager_fallback_ids_cover_all_public_backends():
    assert EAGER_IMPLEMENTATION_IDS == {
        BackendKind.CPU: "ratchet.reference.cpu.eager.v1",
        BackendKind.XPU: "ratchet.intel.xpu.eager.v1",
        BackendKind.CUDA: "ratchet.nvidia.cuda.eager.v1",
        BackendKind.HIP: "ratchet.amd.hip.eager.v1",
    }
    empty = CatalogueProjection("d" * 64, 0, ())
    for kind in BackendKind:
        request = _request(identity=_identity(kind))
        decision = EvidenceDrivenDispatch(empty).choose(request)
        assert isinstance(decision, UntunedFallback)
        assert decision.implementation_id == EAGER_IMPLEMENTATION_IDS[kind]
        assert decision.is_tuned is False


def test_eager_fallback_ids_are_immutable_and_cannot_change_decisions():
    request = _request(identity=_identity(BackendKind.XPU))
    before = EvidenceDrivenDispatch(CatalogueProjection("d" * 64, 0, ())).choose(
        request
    )

    with pytest.raises(TypeError):
        EAGER_IMPLEMENTATION_IDS[BackendKind.XPU] = "changed"  # type: ignore[index]

    after = EvidenceDrivenDispatch(CatalogueProjection("d" * 64, 0, ())).choose(request)
    assert after == before


def test_qualified_matching_evidence_ranks_speedup_then_implementation_and_binds_source():
    request = _request()
    evidence = (
        _evidence(
            "candidate-z",
            request,
            event_id="EVT-000001",
            paired_speedup_lower_bound=1.20,
        ),
        _evidence(
            "candidate-a",
            request,
            event_id="EVT-000002",
            paired_speedup_lower_bound=1.20,
        ),
        _evidence(
            "candidate-slower",
            request,
            event_id="EVT-000003",
            paired_speedup_lower_bound=1.10,
        ),
    )

    decision = EvidenceDrivenDispatch(PROJECTION, evidence).choose(request)

    assert isinstance(decision, TunedDispatch)
    assert (
        decision.implementation_id,
        decision.source_event_id,
        decision.source_projection_id,
    ) == ("candidate-a", "EVT-000002", PROJECTION.projection_id)
    assert decision.is_tuned is True


def test_dispatch_ranking_is_deterministic_when_input_evidence_order_changes():
    request = _request()
    first = _evidence(
        "candidate", request, event_id="EVT-000002", paired_speedup_lower_bound=1.10
    )
    second = _evidence(
        "candidate", request, event_id="EVT-000001", paired_speedup_lower_bound=1.10
    )

    forward = EvidenceDrivenDispatch(PROJECTION, (first, second)).choose(request)
    reversed_order = EvidenceDrivenDispatch(PROJECTION, (second, first)).choose(request)

    assert forward == reversed_order
    assert isinstance(forward, TunedDispatch)
    assert forward.source_event_id == "EVT-000001"


def test_dispatch_rejects_mutable_evidence_at_construction():
    request = _request()
    mutable_evidence = [_evidence("candidate", request)]

    with pytest.raises(TypeError, match="immutable tuple"):
        EvidenceDrivenDispatch(PROJECTION, mutable_evidence)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "changes",
    (
        {"profile": _profile(regime="other-regime")},
        {"profile": _profile(evaluator_contract_digest="d" * 64)},
        {"profile": _profile(benchmark_configuration_digest="e" * 64)},
        {"profile": _profile(dtype="bfloat16")},
        {"backend_identity": _identity(suffix="different")},
        {"correctness_passed": False},
        {"synchronized": False},
        {"latency_intervals_disjoint": False},
        {"paired_speedup_lower_bound": 1.02},
        {"peak_memory_increase_ratio": 0.051},
        {"dispatch_verified": False},
    ),
)
def test_every_profile_identity_and_evidence_gate_blocks_promotion(
    changes: dict[str, object],
):
    request = _request()
    evidence = _evidence("candidate", request, **changes)

    decision = EvidenceDrivenDispatch(PROJECTION, (evidence,)).choose(request)

    assert isinstance(decision, UntunedFallback)
    assert decision.reason == "no qualifying evidence for dispatch profile"


@pytest.mark.parametrize(
    "changes",
    (
        {"source_event_id": "EVT-000004"},
        {"source_projection_id": "f" * 64},
    ),
)
def test_event_or_projection_provenance_perturbation_forces_fallback(
    changes: dict[str, object],
):
    request = _request()
    evidence = _evidence("candidate", request, **changes)

    decision = EvidenceDrivenDispatch(PROJECTION, (evidence,)).choose(request)

    assert isinstance(decision, UntunedFallback)
    assert decision.reason == "no qualifying evidence for dispatch profile"


def test_malformed_dispatch_digests_and_event_ids_reject_at_construction():
    request = _request()

    with pytest.raises(ValueError, match="SHA-256"):
        DispatchProfile("reference-transformer", "not-a-digest", "b" * 64, "float32")
    with pytest.raises(ValueError, match="event id"):
        _evidence("candidate", request, source_event_id="not-an-event")
    with pytest.raises(ValueError, match="SHA-256"):
        _evidence("candidate", request, source_projection_id="not-a-digest")


def test_real_zero_event_projection_always_uses_fallback():
    projection = FileExperimentArchive(ROOT / "research" / "archive").projection()
    request = _request()

    assert projection.event_count == 0
    assert projection.event_ids == ()
    decision = EvidenceDrivenDispatch(projection).choose(request)

    assert isinstance(decision, UntunedFallback)
    assert decision.reason == "verified projection contains no events"


def test_cpu_never_selects_tuned_evidence():
    request = _request(identity=_identity(BackendKind.CPU))
    evidence = _evidence("candidate", request)

    decision = EvidenceDrivenDispatch(PROJECTION, (evidence,)).choose(request)

    assert isinstance(decision, UntunedFallback)
    assert decision.reason == "CPU is diagnostic-only and cannot be tuned"


@pytest.mark.parametrize(
    "capabilities",
    (
        _capabilities(ValidationState.AVAILABLE),
        _capabilities(supports_events=False),
        _capabilities(supports_peak_memory=False),
        _capabilities(supported_dtypes=("bfloat16",)),
    ),
)
def test_unqualified_backend_forces_fallback_despite_eligible_test_fixture(
    capabilities: BackendCapabilities,
):
    request = _request(capabilities=capabilities)
    evidence = _evidence("candidate", request)

    decision = EvidenceDrivenDispatch(PROJECTION, (evidence,)).choose(request)

    assert isinstance(decision, UntunedFallback)
    assert decision.reason == "backend is not qualification-ready"


def test_empty_compiler_identity_forces_fallback_despite_eligible_evidence():
    request = _request(identity=_identity(compiler_version=""))
    evidence = _evidence("candidate", request)

    decision = EvidenceDrivenDispatch(PROJECTION, (evidence,)).choose(request)

    assert isinstance(decision, UntunedFallback)
    assert decision.reason == "backend is not qualification-ready"


def test_tuned_and_untuned_decisions_enforce_their_xor_state():
    with pytest.raises(ValueError, match="tuned dispatch"):
        TunedDispatch("candidate", "EVT-000001", PROJECTION.projection_id, False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="untuned fallback"):
        UntunedFallback("ratchet.intel.xpu.eager.v1", "fallback", True)  # type: ignore[arg-type]


def test_dispatch_is_pure_and_does_not_inspect_evaluator_framework_or_sources():
    source = ROOT / "ratchet" / "dispatch" / "__init__.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert imports == {
        "__future__",
        "dataclasses",
        "ratchet.backends",
        "ratchet.experiments",
        "types",
        "typing",
    }
    assert not {"ast", "inspect", "Path", "torch", "triton"} & names
