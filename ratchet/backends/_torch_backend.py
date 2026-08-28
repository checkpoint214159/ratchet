"""Shared lazy PyTorch adapter mechanics; vendor selection lives in subpackages."""

from __future__ import annotations

from collections.abc import Callable

from ratchet.backends import (
    AvailabilityState,
    BackendCapabilities,
    BackendIdentity,
    BackendKind,
    BackendUnavailableError,
    CompilationPolicy,
    CompiledModel,
    MemoryEvidence,
    TimingConfiguration,
    TimingEvidence,
    ValidationState,
)
from ratchet.backends._runtime import load_torch


class TorchAcceleratorBackend:
    """Adapter base that refuses host timing when a device cannot be used."""

    def __init__(
        self,
        kind: BackendKind,
        device_api: Callable[[object], object],
        runtime_version: Callable[[object], str],
        torch_loader: Callable[[], object] = load_torch,
    ) -> None:
        self._kind = kind
        self._device_api = device_api
        self._runtime_version = runtime_version
        self._torch_loader = torch_loader

    def _runtime_and_device(self) -> tuple[object, object]:
        try:
            runtime = self._torch_loader()
        except (ImportError, ModuleNotFoundError) as error:
            raise BackendUnavailableError(
                self._kind, "PyTorch is not installed"
            ) from error
        try:
            device = self._device_api(runtime)
        except (AttributeError, TypeError) as error:
            raise BackendUnavailableError(
                self._kind, "the PyTorch vendor API is unavailable"
            ) from error
        available = getattr(device, "is_available", None)
        if not callable(available) or not available():
            raise BackendUnavailableError(self._kind, "no compatible device is visible")
        return runtime, device

    def _unavailable_reason(self) -> str | None:
        try:
            self._runtime_and_device()
        except BackendUnavailableError as error:
            return error.reason
        return None

    def capabilities(self) -> BackendCapabilities:
        unavailable_reason = self._unavailable_reason()
        if unavailable_reason is not None:
            return BackendCapabilities(
                availability=AvailabilityState.UNAVAILABLE,
                validation=ValidationState.UNAVAILABLE,
                supports_events=False,
                supports_compilation=False,
                supports_peak_memory=False,
                supported_dtypes=(),
            )
        runtime, device = self._runtime_and_device()
        return BackendCapabilities(
            availability=AvailabilityState.AVAILABLE,
            validation=ValidationState.UNVALIDATED,
            supports_events=callable(getattr(device, "Event", None)),
            supports_compilation=callable(getattr(runtime, "compile", None)),
            supports_peak_memory=callable(
                getattr(device, "max_memory_allocated", None)
            ),
            supported_dtypes=("float32", "bfloat16"),
        )

    def probe(self) -> BackendIdentity:
        runtime, device = self._runtime_and_device()
        device_name = getattr(device, "get_device_name", None)
        if not callable(device_name):
            raise BackendUnavailableError(self._kind, "device identity is unavailable")
        version = getattr(runtime, "__version__", None)
        return BackendIdentity(
            backend=self._kind,
            device_name=str(device_name(0)),
            driver_version="reported-by-pytorch",
            runtime_version=self._runtime_version(runtime),
            framework_version=str(version or "unknown"),
            compiler_version="torch.compile"
            if callable(getattr(runtime, "compile", None))
            else "unavailable",
        )

    def synchronize(self) -> None:
        _, device = self._runtime_and_device()
        synchronize = getattr(device, "synchronize", None)
        if not callable(synchronize):
            raise BackendUnavailableError(
                self._kind, "device synchronization is unavailable"
            )
        synchronize()

    def time(
        self, operation: Callable[[], None], configuration: TimingConfiguration
    ) -> TimingEvidence:
        _, device = self._runtime_and_device()
        event_factory = getattr(device, "Event", None)
        synchronize = getattr(device, "synchronize", None)
        if not callable(event_factory) or not callable(synchronize):
            raise BackendUnavailableError(
                self._kind, "device event timing is unavailable"
            )

        for _ in range(configuration.warmup_calls):
            operation()
        synchronize()

        samples: list[int] = []
        for _ in range(configuration.measured_calls):
            start_event = event_factory(enable_timing=True)
            end_event = event_factory(enable_timing=True)
            start_event.record()
            operation()
            end_event.record()
            synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            samples.append(max(1, int(float(elapsed_ms) * 1_000_000)))
        return TimingEvidence(
            method=f"{self._kind.value}_device_event",
            samples_ns=tuple(samples),
            synchronized=True,
        )

    def reset_memory_stats(self) -> None:
        _, device = self._runtime_and_device()
        reset_peak = getattr(device, "reset_peak_memory_stats", None)
        if not callable(reset_peak):
            raise BackendUnavailableError(
                self._kind, "device peak-memory reset is unavailable"
            )
        reset_peak()

    def memory_stats(self) -> MemoryEvidence:
        _, device = self._runtime_and_device()
        allocated = getattr(device, "max_memory_allocated", None)
        reserved = getattr(device, "max_memory_reserved", None)
        return MemoryEvidence(
            peak_allocated_bytes=int(allocated()) if callable(allocated) else None,
            peak_reserved_bytes=int(reserved()) if callable(reserved) else None,
        )

    def compile_model(self, model: object, policy: CompilationPolicy) -> CompiledModel:
        runtime, _ = self._runtime_and_device()
        if not policy.enabled:
            return CompiledModel(model, self._kind, policy, compiler="eager")
        compiler = getattr(runtime, "compile", None)
        if not callable(compiler):
            raise RuntimeError(
                f"{self._kind.value} backend does not support compilation"
            )
        return CompiledModel(
            compiler(model, mode=policy.name),
            self._kind,
            policy,
            compiler="torch.compile",
        )
