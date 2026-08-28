"""Lazy access to the optional PyTorch runtime used by vendor adapters."""

from __future__ import annotations

from importlib import import_module


def load_torch() -> object:
    """Import PyTorch only when an accelerator adapter is actually used."""

    return import_module("torch")
