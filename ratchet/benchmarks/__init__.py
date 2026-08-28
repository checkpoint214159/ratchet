"""Definition-only contracts for reproducible future baseline portfolios.

These records describe what a qualified backend must run later. They do not
load a framework, construct a model, or provide an execution entry point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeVar

REFERENCE_BENCHMARK_PATH = "benchmarks/reference/torch_transformer_benchmark.py"
REFERENCE_BENCHMARK_SHA256 = (
    "5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e"
)
CONFIGURATION_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EnumT = TypeVar("_EnumT", bound=Enum)


class BaselineKind(str, Enum):
    """The portable baseline families retained for a future measurement run."""

    EAGER = "eager"
    COMPILED = "compiled"
    SDPA = "sdpa"
    VENDOR_LIBRARY = "vendor_library"


class BaselineBackend(str, Enum):
    """Vendor-neutral accelerator targets for future definitions."""

    XPU = "xpu"
    CUDA = "cuda"
    HIP = "hip"


class BaselineVendor(str, Enum):
    """The provider namespace associated with each accelerator target."""

    INTEL = "intel"
    NVIDIA = "nvidia"
    AMD = "amd"


class BaselineAvailability(str, Enum):
    """Whether the selected runtime/device can run this definition now."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class BaselineValidation(str, Enum):
    """Whether a definition has passed an empirical backend gate."""

    UNVALIDATED = "unvalidated"
    QUALIFIED = "qualified"


class DispatchVerificationState(str, Enum):
    """The required evidence state for a substituted attention provider."""

    NOT_REQUIRED = "not_required"
    REQUIRED_UNVERIFIED = "required_unverified"


@dataclass(frozen=True, slots=True)
class ReferenceBenchmarkCustody:
    """The byte-preserved authoritative evaluator required by every baseline."""

    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.relative_path != REFERENCE_BENCHMARK_PATH:
            raise ValueError("baseline must use the authoritative reference benchmark")
        if self.sha256 != REFERENCE_BENCHMARK_SHA256 or not _SHA256.fullmatch(
            self.sha256
        ):
            raise ValueError("baseline reference benchmark hash is not authoritative")


@dataclass(frozen=True, slots=True)
class CompilationProtocol:
    """A named compilation policy, recorded independently from latency phases."""

    policy: str
    enabled: bool
    backend: str | None
    mode: str | None
    fullgraph: bool | None
    dynamic: bool | None

    def __post_init__(self) -> None:
        if not self.policy:
            raise ValueError("compilation policy must not be empty")
        options = (self.backend, self.mode, self.fullgraph, self.dynamic)
        if self.enabled and (
            not isinstance(self.backend, str)
            or not self.backend
            or not isinstance(self.mode, str)
            or not self.mode
            or not isinstance(self.fullgraph, bool)
            or not isinstance(self.dynamic, bool)
        ):
            raise ValueError(
                "enabled compilation requires backend, mode, fullgraph, dynamic"
            )
        if not self.enabled and any(option is not None for option in options):
            raise ValueError("disabled compilation cannot carry compiler options")


@dataclass(frozen=True, slots=True)
class TimingProtocol:
    """The required separated compilation, first-run, and steady-state protocol."""

    compilation_recorded_separately: bool
    first_run_recorded_separately: bool
    steady_state_recorded_separately: bool
    synchronization: str
    timing_method: str
    warmup_completed_calls: int
    ordering_blocks: int
    completed_calls_per_model_per_block: int
    ordering_patterns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.compilation_recorded_separately:
            raise ValueError("compilation must be recorded separately")
        if not self.first_run_recorded_separately:
            raise ValueError("first run must be recorded separately")
        if not self.steady_state_recorded_separately:
            raise ValueError("steady state must be recorded separately")
        if (
            self.synchronization
            != "backend synchronize before and after measured region"
        ):
            raise ValueError("baseline synchronization protocol is unsupported")
        if self.timing_method != "backend device events with host-timer cross-check":
            raise ValueError("baseline timing method is unsupported")
        if self.warmup_completed_calls != 20:
            raise ValueError("baseline protocol requires 20 completed warmup calls")
        if self.ordering_blocks != 10:
            raise ValueError("baseline protocol requires 10 ordering blocks")
        if self.completed_calls_per_model_per_block != 30:
            raise ValueError("baseline protocol requires 30 completed calls per block")
        if self.ordering_patterns != ("ABBA", "BAAB"):
            raise ValueError("baseline protocol requires alternating ABBA/BAAB blocks")


@dataclass(frozen=True, slots=True)
class DispatchVerification:
    """A future-run requirement proving the requested provider was dispatched."""

    state: DispatchVerificationState
    method: str
    expected_provider: str

    def __post_init__(self) -> None:
        if not self.method or not self.expected_provider:
            raise ValueError("dispatch verification must name method and provider")


@dataclass(frozen=True, slots=True)
class AttentionSubstitution:
    """A constrained replacement of only the authoritative attention core."""

    primitive: str
    provider: str
    scope: str
    weight_copy_compatible: bool
    valid_mask_semantics: str
    causal_semantics: str
    output_contract: str
    dispatch_verification: DispatchVerification

    def __post_init__(self) -> None:
        if not self.primitive or not self.provider:
            raise ValueError("attention substitution must name primitive and provider")
        if self.scope != "attention_core_only":
            raise ValueError(
                "attention substitution must be limited to the attention core"
            )
        if not self.weight_copy_compatible:
            raise ValueError(
                "attention substitution must preserve weight-copy compatibility"
            )
        if self.valid_mask_semantics != "preserved":
            raise ValueError(
                "attention substitution must preserve valid-mask semantics"
            )
        if self.causal_semantics != "preserved":
            raise ValueError("attention substitution must preserve causal semantics")
        if self.output_contract != "unchanged":
            raise ValueError("attention substitution must preserve the output contract")


@dataclass(frozen=True, slots=True)
class BaselineDefinition:
    """A non-executable, future-run full-workload baseline definition."""

    baseline_id: str
    kind: BaselineKind
    vendor: BaselineVendor
    target_backend: BaselineBackend
    availability: BaselineAvailability
    validation: BaselineValidation
    custody: ReferenceBenchmarkCustody
    workload: str
    attention_substitution: AttentionSubstitution | None
    compilation: CompilationProtocol
    timing: TimingProtocol
    definition_only: bool

    def __post_init__(self) -> None:
        if not self.baseline_id:
            raise ValueError("baseline id must not be empty")
        expected_vendor = {
            BaselineBackend.XPU: BaselineVendor.INTEL,
            BaselineBackend.CUDA: BaselineVendor.NVIDIA,
            BaselineBackend.HIP: BaselineVendor.AMD,
        }[self.target_backend]
        if self.vendor is not expected_vendor:
            raise ValueError("baseline vendor does not match its backend")
        expected_id = f"ratchet.{self.vendor.value}.{self.target_backend.value}.{self.kind.value}.v1"
        if self.baseline_id != expected_id:
            raise ValueError(
                "baseline id must bind namespace, vendor, backend, and kind"
            )
        if self.workload != "reference_transformer":
            raise ValueError(
                "every baseline must retain the reference transformer workload"
            )
        if (
            self.availability is BaselineAvailability.UNAVAILABLE
            and self.validation is BaselineValidation.QUALIFIED
        ):
            raise ValueError("an unavailable baseline cannot be qualified")
        if not self.definition_only:
            raise ValueError("baseline definitions cannot enable execution")
        expected_compilation = {
            BaselineKind.EAGER: ("eager", False, None, None, None, None),
            BaselineKind.COMPILED: (
                "torch.compile",
                True,
                "inductor",
                "default",
                False,
                False,
            ),
            BaselineKind.SDPA: ("eager", False, None, None, None, None),
            BaselineKind.VENDOR_LIBRARY: (
                "vendor-library",
                False,
                None,
                None,
                None,
                None,
            ),
        }
        actual_compilation = (
            self.compilation.policy,
            self.compilation.enabled,
            self.compilation.backend,
            self.compilation.mode,
            self.compilation.fullgraph,
            self.compilation.dynamic,
        )
        if actual_compilation != expected_compilation[self.kind]:
            raise ValueError(
                "baseline kind does not match its fixed compilation protocol"
            )
        if self.kind in {BaselineKind.EAGER, BaselineKind.COMPILED}:
            if self.attention_substitution is not None:
                raise ValueError(
                    "full-workload baseline cannot replace its attention core"
                )
            return
        substitution = self.attention_substitution
        if substitution is None:
            raise ValueError(
                "attention baseline requires an explicit substitution seam"
            )
        expected_substitution = {
            BaselineKind.SDPA: (
                "torch.nn.functional.scaled_dot_product_attention",
                "PyTorch framework SDPA",
                DispatchVerificationState.NOT_REQUIRED,
                "framework dispatch diagnostics",
                "PyTorch framework SDPA",
            ),
            BaselineKind.VENDOR_LIBRARY: (
                "oneDNN Graph SDPA",
                "oneDNN Graph",
                DispatchVerificationState.REQUIRED_UNVERIFIED,
                "oneDNN Graph partition inspection",
                "oneDNN Graph SDPA",
            ),
        }[self.kind]
        if (
            substitution.primitive,
            substitution.provider,
            substitution.dispatch_verification.state,
            substitution.dispatch_verification.method,
            substitution.dispatch_verification.expected_provider,
        ) != expected_substitution:
            raise ValueError("baseline kind does not match its attention substitution")


def baseline_from_configuration(
    configuration: Mapping[str, object],
) -> BaselineDefinition:
    """Validate one checked-in JSON definition without loading an execution runtime."""

    expected_fields = {
        "schema_version",
        "baseline_id",
        "kind",
        "vendor",
        "target_backend",
        "availability",
        "validation",
        "reference_benchmark",
        "workload",
        "attention_substitution",
        "compilation",
        "timing",
        "definition_only",
    }
    if set(configuration) != expected_fields:
        raise ValueError("baseline configuration fields do not match its allow-list")
    if configuration.get("schema_version") != CONFIGURATION_SCHEMA_VERSION:
        raise ValueError("baseline configuration schema version is unsupported")
    custody_data = _mapping(configuration.get("reference_benchmark"), "reference")
    compilation_data = _mapping(configuration.get("compilation"), "compilation")
    timing_data = _mapping(configuration.get("timing"), "timing")
    if set(custody_data) != {"relative_path", "sha256"}:
        raise ValueError("reference benchmark fields do not match its allow-list")
    if set(compilation_data) != {
        "policy",
        "enabled",
        "backend",
        "mode",
        "fullgraph",
        "dynamic",
    }:
        raise ValueError("compilation fields do not match its allow-list")
    if set(timing_data) != {
        "compilation_recorded_separately",
        "first_run_recorded_separately",
        "steady_state_recorded_separately",
        "synchronization",
        "timing_method",
        "warmup_completed_calls",
        "ordering_blocks",
        "completed_calls_per_model_per_block",
        "ordering_patterns",
    }:
        raise ValueError("timing fields do not match its allow-list")
    patterns = timing_data["ordering_patterns"]
    if not isinstance(patterns, list) or not all(
        isinstance(pattern, str) for pattern in patterns
    ):
        raise ValueError("ordering patterns must be a string list")
    return BaselineDefinition(
        baseline_id=_string(configuration, "baseline_id"),
        kind=_enum(BaselineKind, configuration, "kind"),
        vendor=_enum(BaselineVendor, configuration, "vendor"),
        target_backend=_enum(BaselineBackend, configuration, "target_backend"),
        availability=_enum(BaselineAvailability, configuration, "availability"),
        validation=_enum(BaselineValidation, configuration, "validation"),
        custody=ReferenceBenchmarkCustody(
            relative_path=_string(custody_data, "relative_path"),
            sha256=_string(custody_data, "sha256"),
        ),
        workload=_string(configuration, "workload"),
        attention_substitution=_substitution(
            configuration.get("attention_substitution")
        ),
        compilation=CompilationProtocol(
            policy=_string(compilation_data, "policy"),
            enabled=_boolean(compilation_data, "enabled"),
            backend=_optional_string(compilation_data, "backend"),
            mode=_optional_string(compilation_data, "mode"),
            fullgraph=_optional_boolean(compilation_data, "fullgraph"),
            dynamic=_optional_boolean(compilation_data, "dynamic"),
        ),
        timing=TimingProtocol(
            compilation_recorded_separately=_boolean(
                timing_data, "compilation_recorded_separately"
            ),
            first_run_recorded_separately=_boolean(
                timing_data, "first_run_recorded_separately"
            ),
            steady_state_recorded_separately=_boolean(
                timing_data, "steady_state_recorded_separately"
            ),
            synchronization=_string(timing_data, "synchronization"),
            timing_method=_string(timing_data, "timing_method"),
            warmup_completed_calls=_integer(timing_data, "warmup_completed_calls"),
            ordering_blocks=_integer(timing_data, "ordering_blocks"),
            completed_calls_per_model_per_block=_integer(
                timing_data, "completed_calls_per_model_per_block"
            ),
            ordering_patterns=tuple(patterns),
        ),
        definition_only=_boolean(configuration, "definition_only"),
    )


def _substitution(value: object) -> AttentionSubstitution | None:
    if value is None:
        return None
    item = _mapping(value, "attention substitution")
    if set(item) != {
        "primitive",
        "provider",
        "scope",
        "weight_copy_compatible",
        "valid_mask_semantics",
        "causal_semantics",
        "output_contract",
        "dispatch_verification",
    }:
        raise ValueError("attention substitution fields do not match its allow-list")
    dispatch = _mapping(item["dispatch_verification"], "dispatch verification")
    if set(dispatch) != {"state", "method", "expected_provider"}:
        raise ValueError("dispatch verification fields do not match its allow-list")
    return AttentionSubstitution(
        primitive=_string(item, "primitive"),
        provider=_string(item, "provider"),
        scope=_string(item, "scope"),
        weight_copy_compatible=_boolean(item, "weight_copy_compatible"),
        valid_mask_semantics=_string(item, "valid_mask_semantics"),
        causal_semantics=_string(item, "causal_semantics"),
        output_contract=_string(item, "output_contract"),
        dispatch_verification=DispatchVerification(
            state=_enum(DispatchVerificationState, dispatch, "state"),
            method=_string(dispatch, "method"),
            expected_provider=_string(dispatch, "expected_provider"),
        ),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _string(value: Mapping[str, object], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{name} must be non-empty text")
    return item


def _optional_string(value: Mapping[str, object], name: str) -> str | None:
    item = value.get(name)
    if item is None:
        return None
    if not isinstance(item, str) or not item:
        raise ValueError(f"{name} must be null or non-empty text")
    return item


def _boolean(value: Mapping[str, object], name: str) -> bool:
    item = value.get(name)
    if not isinstance(item, bool):
        raise ValueError(f"{name} must be a boolean")
    return item


def _optional_boolean(value: Mapping[str, object], name: str) -> bool | None:
    item = value.get(name)
    if item is None:
        return None
    if not isinstance(item, bool):
        raise ValueError(f"{name} must be null or a boolean")
    return item


def _integer(value: Mapping[str, object], name: str) -> int:
    item = value.get(name)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"{name} must be an integer")
    return item


def _enum(enum: type[_EnumT], value: Mapping[str, object], name: str) -> _EnumT:
    item = _string(value, name)
    try:
        return enum(item)
    except ValueError as error:
        raise ValueError(f"{name} is unsupported") from error


__all__ = [
    "AttentionSubstitution",
    "BaselineAvailability",
    "BaselineBackend",
    "BaselineDefinition",
    "BaselineKind",
    "BaselineValidation",
    "BaselineVendor",
    "CONFIGURATION_SCHEMA_VERSION",
    "CompilationProtocol",
    "DispatchVerification",
    "DispatchVerificationState",
    "REFERENCE_BENCHMARK_PATH",
    "REFERENCE_BENCHMARK_SHA256",
    "ReferenceBenchmarkCustody",
    "TimingProtocol",
    "baseline_from_configuration",
]
