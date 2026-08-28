"""Pure, evidence-driven implementation selection without framework inspection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Protocol, TypeAlias

from ratchet.backends import (
    AvailabilityState,
    BackendCapabilities,
    BackendIdentity,
    BackendKind,
    ValidationState,
)
from ratchet.experiments import CatalogueProjection, EventId

_SHA256 = re.compile(r"[a-f0-9]{64}")
EAGER_IMPLEMENTATION_IDS: Mapping[BackendKind, str] = MappingProxyType(
    {
        BackendKind.CPU: "ratchet.reference.cpu.eager.v1",
        BackendKind.XPU: "ratchet.intel.xpu.eager.v1",
        BackendKind.CUDA: "ratchet.nvidia.cuda.eager.v1",
        BackendKind.HIP: "ratchet.amd.hip.eager.v1",
    }
)


def _sha256(name: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class DispatchProfile:
    """The exact evaluator/configuration regime evidence must match."""

    regime: str
    evaluator_contract_digest: str
    benchmark_configuration_digest: str
    dtype: str

    def __post_init__(self) -> None:
        if not self.regime or not self.dtype:
            raise ValueError("dispatch profile fields must not be empty")
        _sha256("evaluator_contract_digest", self.evaluator_contract_digest)
        _sha256("benchmark_configuration_digest", self.benchmark_configuration_digest)


@dataclass(frozen=True, slots=True)
class CandidateDispatchEvidence:
    """A provenance-bound candidate record considered by the pure policy."""

    implementation_id: str
    source_event_id: str
    source_projection_id: str
    profile: DispatchProfile
    backend_identity: BackendIdentity
    correctness_passed: bool
    synchronized: bool
    latency_intervals_disjoint: bool
    paired_speedup_lower_bound: float
    peak_memory_increase_ratio: float
    dispatch_verified: bool

    def __post_init__(self) -> None:
        if not self.implementation_id:
            raise ValueError("implementation_id must not be empty")
        EventId(self.source_event_id)
        _sha256("source_projection_id", self.source_projection_id)
        if (
            not math.isfinite(self.paired_speedup_lower_bound)
            or self.paired_speedup_lower_bound <= 0
        ):
            raise ValueError("paired_speedup_lower_bound must be finite and positive")
        if not math.isfinite(self.peak_memory_increase_ratio):
            raise ValueError("peak_memory_increase_ratio must be finite")


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    """The current profile and backend facts against which evidence is matched."""

    profile: DispatchProfile
    backend: BackendIdentity
    capabilities: BackendCapabilities


@dataclass(frozen=True, slots=True)
class TunedDispatch:
    """A deterministic winner bound to the exact event and projection that support it."""

    implementation_id: str
    source_event_id: str
    source_projection_id: str
    is_tuned: Literal[True] = True

    def __post_init__(self) -> None:
        if not self.implementation_id or self.is_tuned is not True:
            raise ValueError("tuned dispatch fields must be populated")
        EventId(self.source_event_id)
        _sha256("source_projection_id", self.source_projection_id)


@dataclass(frozen=True, slots=True)
class UntunedFallback:
    """The visible portable eager choice when promotion thresholds are not met."""

    implementation_id: str
    reason: str
    is_tuned: Literal[False] = False

    def __post_init__(self) -> None:
        if not self.implementation_id or not self.reason or self.is_tuned is not False:
            raise ValueError("untuned fallback fields must not be empty")


DispatchDecision: TypeAlias = TunedDispatch | UntunedFallback


class EvidenceDrivenDispatch:
    """Rank only provenance-bound records satisfying every promotion gate."""

    def __init__(
        self,
        projection: CatalogueProjection,
        evidence: tuple[CandidateDispatchEvidence, ...] = (),
    ) -> None:
        if not isinstance(evidence, tuple):
            raise TypeError("dispatch evidence must be an immutable tuple")
        self._projection = projection
        self._evidence = evidence

    @staticmethod
    def _qualified(request: DispatchRequest) -> bool:
        capabilities = request.capabilities
        return (
            capabilities.availability is AvailabilityState.AVAILABLE
            and capabilities.validation is ValidationState.QUALIFIED
            and request.profile.dtype in capabilities.supported_dtypes
            and capabilities.supports_events
            and capabilities.supports_peak_memory
            and bool(request.backend.compiler_version)
        )

    def _eligible(
        self, evidence: CandidateDispatchEvidence, request: DispatchRequest
    ) -> bool:
        return (
            evidence.source_projection_id == self._projection.projection_id
            and evidence.source_event_id in self._projection.event_ids
            and evidence.profile == request.profile
            and evidence.backend_identity == request.backend
            and evidence.correctness_passed
            and evidence.synchronized
            and evidence.latency_intervals_disjoint
            and evidence.paired_speedup_lower_bound > 1.02
            and evidence.peak_memory_increase_ratio <= 0.05
            and evidence.dispatch_verified
        )

    @staticmethod
    def _fallback(request: DispatchRequest, reason: str) -> UntunedFallback:
        return UntunedFallback(
            EAGER_IMPLEMENTATION_IDS[request.backend.backend], reason
        )

    def choose(self, request: DispatchRequest) -> DispatchDecision:
        if request.backend.backend is BackendKind.CPU:
            return self._fallback(request, "CPU is diagnostic-only and cannot be tuned")
        if self._projection.event_count == 0:
            return self._fallback(request, "verified projection contains no events")
        if not self._qualified(request):
            return self._fallback(request, "backend is not qualification-ready")
        eligible = [item for item in self._evidence if self._eligible(item, request)]
        if not eligible:
            return self._fallback(
                request, "no qualifying evidence for dispatch profile"
            )
        winner = sorted(
            eligible,
            key=lambda item: (
                -item.paired_speedup_lower_bound,
                item.implementation_id,
                item.source_event_id,
            ),
        )[0]
        return TunedDispatch(
            winner.implementation_id,
            winner.source_event_id,
            winner.source_projection_id,
        )


class DispatchPolicy(Protocol):
    def choose(self, request: DispatchRequest) -> DispatchDecision: ...


__all__ = [
    "CandidateDispatchEvidence",
    "DispatchDecision",
    "DispatchPolicy",
    "DispatchProfile",
    "DispatchRequest",
    "EAGER_IMPLEMENTATION_IDS",
    "EvidenceDrivenDispatch",
    "TunedDispatch",
    "UntunedFallback",
]
