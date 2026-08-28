"""CPU correctness and diagnostic adapter."""

from __future__ import annotations

from platform import processor, python_version
from time import perf_counter_ns
from typing import Callable

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


class CpuBackend:
    """Host timing for diagnostics only, never accelerator evidence."""

    def probe(self) -> BackendIdentity:
        return BackendIdentity(
            backend=BackendKind.CPU,
            device_name=processor() or "generic CPU",
            driver_version="host",
            runtime_version="host",
            framework_version=python_version(),
            compiler_version="none",
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            availability=AvailabilityState.AVAILABLE,
            validation=ValidationState.AVAILABLE,
            supports_events=False,
            supports_compilation=False,
            supports_peak_memory=False,
            supported_dtypes=("float32",),
        )

    def synchronize(self) -> None:
        return None

    def time(
        self, operation: Callable[[], None], configuration: TimingConfiguration
    ) -> TimingEvidence:
        for _ in range(configuration.warmup_calls):
            operation()
        samples: list[int] = []
        for _ in range(configuration.measured_calls):
            started_at = perf_counter_ns()
            operation()
            samples.append(max(1, perf_counter_ns() - started_at))
        return TimingEvidence("host_diagnostic", tuple(samples), synchronized=False)

    def reset_memory_stats(self) -> None:
        return None

    def memory_stats(self) -> MemoryEvidence:
        return MemoryEvidence(None, None)

    def compile_model(self, model: object, policy: CompilationPolicy) -> CompiledModel:
        if policy.enabled:
            raise RuntimeError("cpu backend does not support compilation")
        return CompiledModel(model, BackendKind.CPU, policy, compiler="identity")
