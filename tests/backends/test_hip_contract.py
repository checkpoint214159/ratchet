"""CPU-only ROCm/HIP contract tests using fake runtimes; no runtime is imported."""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from ratchet.backends import (
    AvailabilityState,
    BackendCapabilities,
    BackendIdentity,
    BackendKind,
    BackendUnavailableError,
    CompilationPolicy,
    TimingConfiguration,
    ValidationState,
)
from ratchet.backends.cuda import CudaBackend
from ratchet.backends.hip import HipBackend
from ratchet.dispatch import (
    DispatchProfile,
    DispatchRequest,
    EvidenceDrivenDispatch,
    UntunedFallback,
)
from ratchet.experiments import CatalogueProjection

ROOT = Path(__file__).resolve().parents[2]
HIP_SOURCE = ROOT / "ratchet" / "backends" / "hip" / "__init__.py"
HIP_DOC = ROOT / "docs" / "backends" / "rocm-hip.md"


@dataclass
class FakeEvent:
    recorded: bool = False

    def record(self) -> None:
        self.recorded = True

    def elapsed_time(self, other: FakeEvent) -> float:
        assert self.recorded and other.recorded
        return 0.5


class FakeHipDevice:
    def __init__(self, *, available: bool = True, bf16_supported: bool = False) -> None:
        self.available = available
        self.bf16_supported = bf16_supported
        self.events: list[FakeEvent] = []
        self.synchronizations = 0
        self.reset_calls = 0
        self.peak_allocated = 128
        self.peak_reserved = 256

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.bf16_supported

    def Event(self, *, enable_timing: bool) -> FakeEvent:
        assert enable_timing is True
        event = FakeEvent()
        self.events.append(event)
        return event

    def synchronize(self) -> None:
        self.synchronizations += 1

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "Fake AMD"

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1
        self.peak_allocated = 0
        self.peak_reserved = 0

    def max_memory_allocated(self) -> int:
        return self.peak_allocated

    def max_memory_reserved(self) -> int:
        return self.peak_reserved


class FakeTorch:
    def __init__(self, device: object, *, hip: str | None) -> None:
        self.cuda = device
        self.version = SimpleNamespace(cuda="not-public", hip=hip)
        self.__version__ = "2.8.0-fake"
        self.compile_calls: list[tuple[object, str]] = []

    def compile(self, model: object, *, mode: str) -> object:
        self.compile_calls.append((model, mode))
        return ("compiled", model, mode)


def test_hip_constructs_lazily_and_public_identity_is_hip_not_cuda():
    device = FakeHipDevice()
    runtime = FakeTorch(device, hip="6.2")
    loader_calls = 0

    def load_fake() -> object:
        nonlocal loader_calls
        loader_calls += 1
        return runtime

    backend = HipBackend(load_fake)

    assert loader_calls == 0
    capabilities = backend.capabilities()
    identity = backend.probe()
    assert capabilities.availability is AvailabilityState.AVAILABLE
    assert capabilities.validation is ValidationState.UNVALIDATED
    assert capabilities.supports_events is True
    assert capabilities.supports_compilation is True
    assert capabilities.supports_peak_memory is True
    assert capabilities.supported_dtypes == ("float32",)
    assert identity.backend is BackendKind.HIP
    assert identity.runtime_version == "6.2"
    assert "cuda" not in identity.runtime_version.lower()
    assert loader_calls > 0


def test_hip_reports_bfloat16_only_after_a_positive_device_probe():
    unsupported = HipBackend(
        lambda: FakeTorch(FakeHipDevice(bf16_supported=False), hip="6.2")
    )
    supported = HipBackend(
        lambda: FakeTorch(FakeHipDevice(bf16_supported=True), hip="6.2")
    )

    assert unsupported.capabilities().supported_dtypes == ("float32",)
    assert supported.capabilities().supported_dtypes == ("float32", "bfloat16")


def test_hip_event_timing_compilation_and_memory_lifecycle_are_device_only():
    device = FakeHipDevice()
    runtime = FakeTorch(device, hip="6.2")
    backend = HipBackend(lambda: runtime)
    calls: list[str] = []

    timing = backend.time(lambda: calls.append("run"), TimingConfiguration(1, 2))
    model = object()
    compiled = backend.compile_model(model, CompilationPolicy("default", True))
    backend.reset_memory_stats()
    device.peak_allocated, device.peak_reserved = 1024, 2048
    memory = backend.memory_stats()

    assert timing.method == "hip_device_event"
    assert timing.samples_ns == (500_000, 500_000)
    assert timing.synchronized is True
    assert calls == ["run", "run", "run"]
    assert len(device.events) == 4
    assert device.synchronizations == 3
    assert runtime.compile_calls == [(model, "default")]
    assert compiled.backend is BackendKind.HIP
    assert compiled.model == ("compiled", model, "default")
    assert device.reset_calls == 1
    assert memory.peak_allocated_bytes == 1024
    assert memory.peak_reserved_bytes == 2048


def test_hip_rejects_nvidia_and_unavailable_or_unsupported_runtime_paths():
    nvidia_backend = HipBackend(lambda: FakeTorch(FakeHipDevice(), hip=None))
    unavailable = HipBackend(
        lambda: FakeTorch(FakeHipDevice(available=False), hip="6.2")
    )
    rocm_runtime = FakeTorch(FakeHipDevice(), hip="6.2")

    assert nvidia_backend.capabilities().availability is AvailabilityState.UNAVAILABLE
    assert unavailable.capabilities().validation is ValidationState.UNAVAILABLE
    with pytest.raises(BackendUnavailableError, match="not using HIP"):
        nvidia_backend.probe()
    with pytest.raises(BackendUnavailableError, match="no compatible device"):
        unavailable.time(lambda: None, TimingConfiguration(0, 1))
    with pytest.raises(BackendUnavailableError, match="using HIP"):
        CudaBackend(lambda: rocm_runtime).probe()

    no_events = FakeHipDevice()
    no_events.synchronize = None  # type: ignore[method-assign]
    event_backend = HipBackend(lambda: FakeTorch(no_events, hip="6.2"))
    assert event_backend.capabilities().supports_events is False
    with pytest.raises(BackendUnavailableError, match="event timing"):
        event_backend.time(lambda: None, TimingConfiguration(0, 1))

    no_reserved = FakeHipDevice()
    no_reserved.max_memory_reserved = None  # type: ignore[method-assign]
    memory_backend = HipBackend(lambda: FakeTorch(no_reserved, hip="6.2"))
    assert memory_backend.capabilities().supports_peak_memory is True
    assert memory_backend.memory_stats().peak_reserved_bytes is None

    no_allocated = FakeHipDevice()
    no_allocated.max_memory_allocated = None  # type: ignore[method-assign]
    allocated_backend = HipBackend(lambda: FakeTorch(no_allocated, hip="6.2"))
    assert allocated_backend.capabilities().supports_peak_memory is False
    with pytest.raises(BackendUnavailableError, match="peak-memory"):
        allocated_backend.memory_stats()

    no_reset = FakeHipDevice()
    no_reset.reset_peak_memory_stats = None  # type: ignore[method-assign]
    reset_backend = HipBackend(lambda: FakeTorch(no_reset, hip="6.2"))
    assert reset_backend.capabilities().supports_peak_memory is False
    with pytest.raises(BackendUnavailableError, match="peak-memory reset"):
        reset_backend.reset_memory_stats()


@pytest.mark.parametrize(
    "capabilities",
    [
        BackendCapabilities(
            AvailabilityState.UNAVAILABLE,
            ValidationState.UNAVAILABLE,
            False,
            False,
            False,
            (),
        ),
        BackendCapabilities(
            AvailabilityState.AVAILABLE,
            ValidationState.UNVALIDATED,
            True,
            True,
            True,
            ("float32",),
        ),
    ],
)
def test_hip_unavailable_or_unvalidated_capabilities_always_dispatch_untuned(
    capabilities: BackendCapabilities,
):
    identity = BackendIdentity(
        BackendKind.HIP,
        "Fake AMD",
        "fake-driver",
        "6.2",
        "2.8.0-fake",
        "torch.compile",
    )
    request = DispatchRequest(
        DispatchProfile("default", "a" * 64, "b" * 64, "float32"),
        identity,
        capabilities,
    )
    decision = EvidenceDrivenDispatch(
        CatalogueProjection("c" * 64, 1, ("EVT-000001",))
    ).choose(request)

    assert isinstance(decision, UntunedFallback)
    assert decision.implementation_id == "ratchet.amd.hip.eager.v1"
    assert decision.is_tuned is False


def test_hip_import_is_fresh_and_vendor_source_has_no_runtime_or_vendor_leakage():
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import ratchet.backends.hip; assert 'torch' not in sys.modules",
        ),
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

    tree = ast.parse(HIP_SOURCE.read_text(encoding="utf-8"), filename=str(HIP_SOURCE))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "torch" not in imports
    assert not any(
        name.startswith("ratchet.backends.xpu")
        or name.startswith("ratchet.backends.cuda")
        for name in imports
    )


def test_hip_doctor_and_future_gate_docs_are_explicitly_unvalidated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    doctor = importlib.import_module("ratchet.backends.__main__")
    monkeypatch.setattr(
        doctor,
        "get_backend",
        lambda _: HipBackend(lambda: (_ for _ in ()).throw(ModuleNotFoundError())),
    )
    monkeypatch.setattr(sys, "argv", ["ratchet.backends", "--backend", "hip"])

    assert doctor.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["capabilities"]["availability"] == "unavailable"
    assert "identity" not in payload
    documentation = HIP_DOC.read_text(encoding="utf-8")
    assert ".venv/bin/python -m ratchet.backends --backend hip" in documentation
    assert "unvalidated" in documentation
