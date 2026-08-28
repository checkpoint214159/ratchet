"""Vendor-neutral public contracts and lazy accelerator-backend registry."""

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


class AvailabilityState(str, Enum):
    """Whether this process can currently use a backend runtime and device."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class BackendUnavailableError(RuntimeError):
    """Raised when an operation requires a runtime or device that is not usable."""

    def __init__(self, backend: BackendKind, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        super().__init__(f"{backend.value} backend is unavailable: {reason}")


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
    availability: AvailabilityState
    validation: ValidationState
    supports_events: bool
    supports_compilation: bool
    supports_peak_memory: bool
    supported_dtypes: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.supported_dtypes)) != len(self.supported_dtypes):
            raise ValueError("supported_dtypes must be unique")
        if self.availability is AvailabilityState.UNAVAILABLE:
            if self.validation is not ValidationState.UNAVAILABLE:
                raise ValueError("unavailable backends require unavailable validation")
            if any(
                (
                    self.supports_events,
                    self.supports_compilation,
                    self.supports_peak_memory,
                )
            ):
                raise ValueError("unavailable backends cannot support capabilities")
            if self.supported_dtypes:
                raise ValueError("unavailable backends cannot support dtypes")


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
    """An executable opaque model paired with vendor-neutral compilation metadata."""

    model: object
    backend: BackendKind
    policy: CompilationPolicy
    compiler: str
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not self.compiler:
            raise ValueError("compiler must not be empty")


class AcceleratorBackend(Protocol):
    """The only public execution seam shared by all vendor adapters."""

    def probe(self) -> BackendIdentity: ...

    def capabilities(self) -> BackendCapabilities: ...

    def synchronize(self) -> None: ...

    def time(
        self, operation: Callable[[], None], configuration: TimingConfiguration
    ) -> TimingEvidence: ...

    def reset_memory_stats(self) -> None: ...

    def memory_stats(self) -> MemoryEvidence: ...

    def compile_model(
        self, model: object, policy: CompilationPolicy
    ) -> CompiledModel: ...


def get_backend(backend: BackendKind | str) -> AcceleratorBackend:
    """Construct a backend without importing a vendor runtime at package import time."""

    try:
        kind = backend if isinstance(backend, BackendKind) else BackendKind(backend)
    except ValueError as error:
        raise ValueError(f"unknown backend: {backend!r}") from error

    if kind is BackendKind.CPU:
        from ratchet.backends.cpu import CpuBackend

        return CpuBackend()
    if kind is BackendKind.XPU:
        from ratchet.backends.xpu import XpuBackend

        return XpuBackend()
    if kind is BackendKind.CUDA:
        from ratchet.backends.cuda import CudaBackend

        return CudaBackend()
    if kind is BackendKind.HIP:
        from ratchet.backends.hip import HipBackend

        return HipBackend()
    raise AssertionError(f"unhandled backend kind: {kind}")


__all__ = [
    "AcceleratorBackend",
    "AvailabilityState",
    "BackendCapabilities",
    "BackendIdentity",
    "BackendKind",
    "BackendUnavailableError",
    "CompilationPolicy",
    "CompiledModel",
    "MemoryEvidence",
    "TimingConfiguration",
    "TimingEvidence",
    "ValidationState",
    "get_backend",
]
