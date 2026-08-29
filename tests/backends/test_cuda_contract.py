"""CPU-only CUDA contract tests using fake runtimes; no CUDA runtime is imported."""

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
from ratchet.dispatch import (
    DispatchProfile,
    DispatchRequest,
    EvidenceDrivenDispatch,
    UntunedFallback,
)
from ratchet.experiments import CatalogueProjection

ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "ratchet" / "backends" / "cuda" / "__init__.py"
CUDA_DOC = ROOT / "docs" / "backends" / "cuda.md"


@dataclass
class FakeEvent:
    recorded: bool = False

    def record(self) -> None:
        self.recorded = True

    def elapsed_time(self, other: FakeEvent) -> float:
        assert self.recorded and other.recorded
        return 0.5


class FakeCuda:
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
        return "Fake NVIDIA"

    def reset_peak_memory_stats(self) -> None:
        self.reset_calls += 1
        self.peak_allocated = 0
        self.peak_reserved = 0

    def max_memory_allocated(self) -> int:
        return self.peak_allocated

    def max_memory_reserved(self) -> int:
        return self.peak_reserved


class FakeTorch:
    def __init__(self, cuda: object, *, hip: str | None = None) -> None:
        self.cuda = cuda
        self.version = SimpleNamespace(cuda="12.8", hip=hip)
        self.__version__ = "2.8.0-fake"
        self.compile_calls: list[tuple[object, str]] = []

    def compile(self, model: object, *, mode: str) -> object:
        self.compile_calls.append((model, mode))
        return ("compiled", model, mode)


def test_cuda_constructs_without_loading_torch_then_reports_available_unvalidated():
    cuda = FakeCuda()
    runtime = FakeTorch(cuda)
    loader_calls = 0

    def load_fake() -> object:
        nonlocal loader_calls
        loader_calls += 1
        return runtime

    backend = CudaBackend(load_fake)

    assert loader_calls == 0
    capabilities = backend.capabilities()
    assert capabilities.availability is AvailabilityState.AVAILABLE
    assert capabilities.validation is ValidationState.UNVALIDATED
    assert capabilities.supports_events is True
    assert capabilities.supports_compilation is True
    assert capabilities.supports_peak_memory is True
    assert capabilities.supported_dtypes == ("float32",)
    assert loader_calls > 0


def test_cuda_reports_bfloat16_only_after_a_positive_device_probe():
    unsupported = CudaBackend(lambda: FakeTorch(FakeCuda(bf16_supported=False)))
    supported = CudaBackend(lambda: FakeTorch(FakeCuda(bf16_supported=True)))

    assert unsupported.capabilities().supported_dtypes == ("float32",)
    assert supported.capabilities().supported_dtypes == ("float32", "bfloat16")


def test_cuda_event_timing_warmups_synchronization_compilation_and_memory_lifecycle():
    cuda = FakeCuda()
    runtime = FakeTorch(cuda)
    backend = CudaBackend(lambda: runtime)
    calls: list[str] = []

    timing = backend.time(lambda: calls.append("run"), TimingConfiguration(1, 2))
    model = object()
    compiled = backend.compile_model(model, CompilationPolicy("default", True))
    backend.reset_memory_stats()
    cuda.peak_allocated, cuda.peak_reserved = 1024, 2048
    memory = backend.memory_stats()

    assert timing.method == "cuda_device_event"
    assert timing.samples_ns == (500_000, 500_000)
    assert timing.synchronized is True
    assert calls == ["run", "run", "run"]
    assert len(cuda.events) == 4
    assert cuda.synchronizations == 3
    assert runtime.compile_calls == [(model, "default")]
    assert compiled.model == ("compiled", model, "default")
    assert cuda.reset_calls == 1
    assert memory.peak_allocated_bytes == 1024
    assert memory.peak_reserved_bytes == 2048


def test_cuda_rejects_hip_and_unavailable_or_unsupported_runtime_paths():
    hip_backend = CudaBackend(lambda: FakeTorch(FakeCuda(), hip="6.2"))
    unavailable = CudaBackend(lambda: FakeTorch(FakeCuda(available=False)))

    assert hip_backend.capabilities().availability is AvailabilityState.UNAVAILABLE
    assert unavailable.capabilities().validation is ValidationState.UNAVAILABLE
    with pytest.raises(BackendUnavailableError, match="using HIP"):
        hip_backend.probe()
    with pytest.raises(BackendUnavailableError, match="no compatible device"):
        unavailable.time(lambda: None, TimingConfiguration(0, 1))

    no_events = FakeCuda()
    no_events.synchronize = None  # type: ignore[method-assign]
    backend = CudaBackend(lambda: FakeTorch(no_events))
    assert backend.capabilities().supports_events is False
    with pytest.raises(BackendUnavailableError, match="event timing"):
        backend.time(lambda: None, TimingConfiguration(0, 1))

    no_peaks = FakeCuda()
    no_peaks.max_memory_reserved = None  # type: ignore[method-assign]
    peak_backend = CudaBackend(lambda: FakeTorch(no_peaks))
    assert peak_backend.capabilities().supports_peak_memory is True
    assert peak_backend.memory_stats().peak_reserved_bytes is None

    no_allocated = FakeCuda()
    no_allocated.max_memory_allocated = None  # type: ignore[method-assign]
    allocated_backend = CudaBackend(lambda: FakeTorch(no_allocated))
    assert allocated_backend.capabilities().supports_peak_memory is False
    with pytest.raises(BackendUnavailableError, match="peak-memory"):
        allocated_backend.memory_stats()

    no_reset = FakeCuda()
    no_reset.reset_peak_memory_stats = None  # type: ignore[method-assign]
    reset_backend = CudaBackend(lambda: FakeTorch(no_reset))
    assert reset_backend.capabilities().supports_peak_memory is False
    with pytest.raises(BackendUnavailableError, match="peak-memory reset"):
        reset_backend.reset_memory_stats()

    no_compile = FakeTorch(FakeCuda())
    no_compile.compile = None  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="does not support compilation"):
        CudaBackend(lambda: no_compile).compile_model(
            object(), CompilationPolicy("default", True)
        )


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
def test_cuda_unavailable_or_unvalidated_capabilities_always_dispatch_untuned(
    capabilities: BackendCapabilities,
):
    identity = BackendIdentity(
        BackendKind.CUDA,
        "Fake NVIDIA",
        "fake-driver",
        "12.8",
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
    assert decision.implementation_id == "ratchet.nvidia.cuda.eager.v1"
    assert decision.is_tuned is False


def test_cuda_import_is_fresh_and_vendor_source_has_no_runtime_or_vendor_leakage():
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import ratchet.backends.cuda; assert 'torch' not in sys.modules",
        ),
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr

    tree = ast.parse(CUDA_SOURCE.read_text(encoding="utf-8"), filename=str(CUDA_SOURCE))
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
        or name.startswith("ratchet.backends.hip")
        for name in imports
    )


def test_cuda_doctor_and_future_gate_docs_are_explicitly_unvalidated(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    doctor = importlib.import_module("ratchet.backends.__main__")
    monkeypatch.setattr(
        doctor,
        "get_backend",
        lambda _: CudaBackend(lambda: (_ for _ in ()).throw(ModuleNotFoundError())),
    )
    monkeypatch.setattr(sys, "argv", ["ratchet.backends", "--backend", "cuda"])

    assert doctor.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["capabilities"]["availability"] == "unavailable"
    assert "identity" not in payload
    documentation = CUDA_DOC.read_text(encoding="utf-8")
    assert ".venv/bin/python -m ratchet.backends --backend cuda" in documentation
    assert "unvalidated" in documentation
