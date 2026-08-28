"""NVIDIA CUDA adapter; CUDA API selection stays internal to this package."""

from __future__ import annotations

from collections.abc import Callable

from ratchet.backends import BackendKind, BackendUnavailableError
from ratchet.backends._runtime import load_torch
from ratchet.backends._torch_backend import TorchAcceleratorBackend


class CudaBackend(TorchAcceleratorBackend):
    def __init__(self, torch_loader: Callable[[], object] = load_torch) -> None:
        super().__init__(BackendKind.CUDA, _cuda_api, _cuda_version, torch_loader)


def _cuda_api(runtime: object) -> object:
    version = getattr(runtime, "version", None)
    if getattr(version, "hip", None):
        raise BackendUnavailableError(BackendKind.CUDA, "PyTorch is using HIP")
    return getattr(runtime, "cuda")


def _cuda_version(runtime: object) -> str:
    version = getattr(runtime, "version", None)
    return str(getattr(version, "cuda", None) or "reported-by-pytorch")
