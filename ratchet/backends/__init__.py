"""Vendor-neutral public contracts for accelerator execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol


class BackendKind(str, Enum):
    CPU = "cpu"
    XPU = "xpu"
    CUDA = "cuda"
    HIP = "hip"


class ValidationState(str, Enum):
    UNVALIDATED = "unvalidated"
    AVAILABLE = "available"
    QUALIFIED = "qualified"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    backend: BackendKind
    device_name: str
    driver_version: str
    runtime_version: str
    framework_version: str
    compiler_version: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.device_name,
                self.driver_version,
                self.runtime_version,
                self.framework_version,
            )
        ):
            raise ValueError("backend identity fields must not be empty")


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    validation: ValidationState
    supports_events: bool
    supports_compilation: bool
    supports_peak_memory: bool
    supported_dtypes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.supported_dtypes)) != len(self.supported_dtypes):
            raise ValueError("supported_dtypes must be unique")


@dataclass(frozen=True, slots=True)
class TimingConfiguration:
    warmup_calls: int
    measured_calls: int

    def __post_init__(self) -> None:
        if self.warmup_calls < 0:
            raise ValueError("warmup_calls must be non-negative")
        if self.measured_calls <= 0:
            raise ValueError("measured_calls must be positive")


@dataclass(frozen=True, slots=True)
class TimingEvidence:
    method: str
    samples_ns: tuple[int, ...]
    synchronized: bool

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("method must not be empty")
        if not self.samples_ns or any(sample <= 0 for sample in self.samples_ns):
            raise ValueError("samples_ns must contain positive samples")


@dataclass(frozen=True, slots=True)
class MemoryEvidence:
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    def __post_init__(self) -> None:
        for value in (self.peak_allocated_bytes, self.peak_reserved_bytes):
            if value is not None and value < 0:
                raise ValueError("memory values must be non-negative")


@dataclass(frozen=True, slots=True)
class CompilationPolicy:
    name: str
    enabled: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("compilation policy name must not be empty")


@dataclass(frozen=True, slots=True)
class CompiledModel:
    model_id: str
    backend: BackendKind
    policy: CompilationPolicy
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")


class AcceleratorBackend(Protocol):
    """The only public execution seam shared by all vendor adapters."""

    def probe(self) -> BackendIdentity: ...

    def capabilities(self) -> BackendCapabilities: ...

    def synchronize(self) -> None: ...

    def time(
        self, operation: Callable[[], None], configuration: TimingConfiguration
    ) -> TimingEvidence: ...

    def memory_stats(self) -> MemoryEvidence: ...

    def compile_model(
        self, model_id: str, policy: CompilationPolicy
    ) -> CompiledModel: ...
