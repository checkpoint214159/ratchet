"""AMD HIP adapter; HIP detection stays internal to this package."""

from __future__ import annotations

from collections.abc import Callable

from ratchet.backends import BackendKind, BackendUnavailableError
from ratchet.backends._runtime import load_torch
from ratchet.backends._torch_backend import TorchAcceleratorBackend


class HipBackend(TorchAcceleratorBackend):
    def __init__(self, torch_loader: Callable[[], object] = load_torch) -> None:
        super().__init__(BackendKind.HIP, _hip_api, _hip_version, torch_loader)


def _hip_api(runtime: object) -> object:
    version = getattr(runtime, "version", None)
    if not getattr(version, "hip", None):
        raise BackendUnavailableError(BackendKind.HIP, "PyTorch is not using HIP")
    return getattr(runtime, "cuda")


def _hip_version(runtime: object) -> str:
    version = getattr(runtime, "version", None)
    return str(getattr(version, "hip", None) or "reported-by-pytorch")
