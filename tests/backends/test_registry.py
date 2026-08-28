"""CPU-only adapter tests using fake runtimes rather than accelerator hardware."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import pytest

from ratchet.backends import (
    AvailabilityState,
    BackendKind,
    BackendUnavailableError,
    CompilationPolicy,
    TimingConfiguration,
    ValidationState,
    get_backend,
)
from ratchet.backends.cpu import CpuBackend
from ratchet.backends.cuda import CudaBackend
from ratchet.backends.hip import HipBackend
from ratchet.backends.xpu import XpuBackend


@dataclass
class FakeEvent:
    recorded: bool = False

    def record(self) -> None:
        self.recorded = True

    def elapsed_time(self, other: FakeEvent) -> float:
        assert self.recorded and other.recorded
        return 0.25


class FakeDevice:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.synchronizations = 0
        self.created_events: list[FakeEvent] = []
        self.reset_calls = 0
        self.peak_allocated_bytes = 256
        self.peak_reserved_bytes = 512

    def is_available(self) -> bool:
        return self.available

    def Event(self, *, enable_timing: bool) -> FakeEvent:
        assert enable_timing
        event = FakeEvent()
        self.created_events.append(event)
        return event

    def synchronize(self) -> None:
        self.synchronizations += 1

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Fake accelerator"

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1
        self.peak_allocated_bytes = 0
        self.peak_reserved_bytes = 0

    def max_memory_allocated(self) -> int:
        return self.peak_allocated_bytes

    def max_memory_reserved(self) -> int:
        return self.peak_reserved_bytes


class FakeTorch:
    def __init__(self, device: FakeDevice, *, hip: str | None = None) -> None:
        self.xpu = device
        self.cuda = device
        self.version = SimpleNamespace(cuda="12.8", hip=hip, xpu="2.8")
        self.__version__ = "2.8.0-fake"
        self.compiled_models: list[tuple[object, str]] = []

    def compile(self, model: object, *, mode: str) -> object:
        self.compiled_models.append((model, mode))
        return ("compiled", model, mode)


@pytest.mark.parametrize(
    ("factory", "kind", "hip"),
    [
        (XpuBackend, BackendKind.XPU, None),
        (CudaBackend, BackendKind.CUDA, None),
        (HipBackend, BackendKind.HIP, "6.2"),
    ],
)
def test_vendor_adapters_use_device_events_compile_and_memory(
    factory: Callable[[Callable[[], object]], object],
    kind: BackendKind,
    hip: str | None,
) -> None:
    device = FakeDevice()
    runtime = FakeTorch(device, hip=hip)
    backend = factory(lambda: runtime)
    calls: list[str] = []

    capabilities = backend.capabilities()  # type: ignore[attr-defined]
    timing = backend.time(  # type: ignore[attr-defined]
        lambda: calls.append("operation"), TimingConfiguration(1, 2)
    )
    executable = object()
    compiled = backend.compile_model(  # type: ignore[attr-defined]
        executable, CompilationPolicy("reduce-overhead", True)
    )
    backend.reset_memory_stats()  # type: ignore[attr-defined]
    device.peak_allocated_bytes = 256
    device.peak_reserved_bytes = 512
    memory = backend.memory_stats()  # type: ignore[attr-defined]

    assert capabilities.availability is AvailabilityState.AVAILABLE
    assert capabilities.validation is ValidationState.UNVALIDATED
    assert backend.probe().backend is kind  # type: ignore[attr-defined]
    assert timing.method == f"{kind.value}_device_event"
    assert timing.samples_ns == (250_000, 250_000)
    assert timing.synchronized is True
    assert calls == ["operation", "operation", "operation"]
    assert len(device.created_events) == 4
    assert device.synchronizations == 3
    assert compiled.model == ("compiled", executable, "reduce-overhead")
    assert compiled.compiler == "torch.compile"
    assert runtime.compiled_models == [(executable, "reduce-overhead")]
    assert memory.peak_allocated_bytes == 256
    assert memory.peak_reserved_bytes == 512
    assert device.reset_calls == 1


@pytest.mark.parametrize(
    ("factory", "hip"),
    [(XpuBackend, None), (CudaBackend, None), (HipBackend, "6.2")],
)
def test_vendor_memory_lifecycle_resets_once_then_reads_current_peaks(
    factory: Callable[[Callable[[], object]], object], hip: str | None
) -> None:
    device = FakeDevice()
    backend = factory(lambda: FakeTorch(device, hip=hip))

    backend.reset_memory_stats()  # type: ignore[attr-defined]
    device.peak_allocated_bytes = 1024
    device.peak_reserved_bytes = 2048
    memory = backend.memory_stats()  # type: ignore[attr-defined]

    assert memory.peak_allocated_bytes == 1024
    assert memory.peak_reserved_bytes == 2048
    assert device.reset_calls == 1


@pytest.mark.parametrize(
    ("factory", "kind"),
    [
        (XpuBackend, BackendKind.XPU),
        (CudaBackend, BackendKind.CUDA),
        (HipBackend, BackendKind.HIP),
    ],
)
def test_vendor_adapters_report_missing_runtime_without_host_timing(
    factory: Callable[[Callable[[], object]], object], kind: BackendKind
) -> None:
    def missing_torch() -> object:
        raise ModuleNotFoundError("No module named 'torch'")

    backend = factory(missing_torch)

    capabilities = backend.capabilities()  # type: ignore[attr-defined]
    assert capabilities.availability is AvailabilityState.UNAVAILABLE
    assert capabilities.validation is ValidationState.UNAVAILABLE
    with pytest.raises(BackendUnavailableError, match=f"{kind.value} backend"):
        backend.time(lambda: None, TimingConfiguration(0, 1))  # type: ignore[attr-defined]


def test_vendor_adapters_reject_a_visible_device_for_the_wrong_cuda_family() -> None:
    runtime = FakeTorch(FakeDevice(), hip="6.2")

    capabilities = CudaBackend(lambda: runtime).capabilities()

    assert capabilities.availability is AvailabilityState.UNAVAILABLE
    with pytest.raises(BackendUnavailableError, match="not using HIP"):
        HipBackend(lambda: FakeTorch(FakeDevice())).probe()


@pytest.mark.parametrize(
    ("factory", "hip"),
    [(XpuBackend, None), (CudaBackend, None), (HipBackend, "6.2")],
)
def test_vendor_adapters_reject_an_absent_device(
    factory: Callable[[Callable[[], object]], object], hip: str | None
) -> None:
    backend = factory(lambda: FakeTorch(FakeDevice(available=False), hip=hip))

    capabilities = backend.capabilities()  # type: ignore[attr-defined]

    assert capabilities.availability is AvailabilityState.UNAVAILABLE
    with pytest.raises(BackendUnavailableError, match="no compatible device"):
        backend.synchronize()  # type: ignore[attr-defined]


def test_cpu_is_a_diagnostic_backend_and_preserves_eager_models() -> None:
    backend = CpuBackend()
    executable = object()
    calls: list[str] = []

    timing = backend.time(lambda: calls.append("operation"), TimingConfiguration(1, 2))
    compiled = backend.compile_model(executable, CompilationPolicy("eager", False))

    assert backend.capabilities().availability is AvailabilityState.AVAILABLE
    assert timing.method == "host_diagnostic"
    assert timing.synchronized is False
    assert len(timing.samples_ns) == 2
    assert calls == ["operation", "operation", "operation"]
    assert compiled.model is executable
    assert compiled.compiler == "identity"
    assert backend.reset_memory_stats() is None


def test_public_registry_is_lazy_and_rejects_unknown_backends() -> None:
    assert isinstance(get_backend("cpu"), CpuBackend)
    assert isinstance(get_backend(BackendKind.XPU), XpuBackend)
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("metal")


def test_doctor_reports_a_missing_xpu_runtime_without_claiming_qualification(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ratchet.backends import __main__ as doctor

    def missing_torch() -> object:
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(doctor, "get_backend", lambda _: XpuBackend(missing_torch))
    monkeypatch.setattr(sys, "argv", ["ratchet.backends", "--backend", "xpu"])

    assert doctor.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["capabilities"]["availability"] == "unavailable"
    assert "identity" not in payload
    assert payload["unavailable_reason"] == "PyTorch is not installed"
