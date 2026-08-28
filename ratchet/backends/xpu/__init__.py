"""Intel XPU adapter; the XPU API stays internal to this package."""

from __future__ import annotations

from collections.abc import Callable

from ratchet.backends import BackendKind
from ratchet.backends._runtime import load_torch
from ratchet.backends._torch_backend import TorchAcceleratorBackend


class XpuBackend(TorchAcceleratorBackend):
    def __init__(self, torch_loader: Callable[[], object] = load_torch) -> None:
        super().__init__(BackendKind.XPU, _xpu_api, _xpu_version, torch_loader)


def _xpu_api(runtime: object) -> object:
    return getattr(runtime, "xpu")


def _xpu_version(runtime: object) -> str:
    version = getattr(runtime, "version", None)
    return str(getattr(version, "xpu", None) or "reported-by-pytorch")
