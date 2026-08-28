"""Guard for the golden suite: everything in this directory needs a real CUDA device.

The golden tests pin the oracle's observable behavior -- locked constants, input bits,
gate mechanics, the fp64 error floor. They are the tripwire that fires when anyone
weakens the correctness gate. A tripwire that errors out with "no CUDA device" on a
CPU-only box looks like oracle breakage and trains people to ignore red, so instead the
whole directory is skipped when there is no GPU to run on.  If PyTorch itself is absent,
the test modules are not imported; when PyTorch is present but CUDA is unavailable, every
golden test is skipped loudly with its reason.

We add skip markers from pytest_collection_modifyitems rather than calling
pytest.skip(allow_module_level=True) at conftest import time: a module-level skip in a
conftest aborts pytest's configuration stage with a traceback instead of reporting a
clean skip.  `collect_ignore_glob` is set only for a genuinely missing `torch` module,
before pytest can import the GPU-only test modules.
"""

import pathlib

import pytest

try:
    import torch
except ModuleNotFoundError as exc:
    # Do not import any golden module when PyTorch is genuinely absent: each imports
    # either torch itself or the CUDA-only legacy oracle.  Ignoring collection keeps a
    # CPU-only environment green without changing a single GPU assertion.
    if exc.name != "torch":
        raise
    torch = None
    collect_ignore_glob = ["test_*.py"]

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    if torch is None:
        return

    if torch.cuda.is_available():
        skip = None
    else:
        skip = pytest.mark.skip(reason="golden oracle tests require a CUDA device")

    for item in items:
        # This hook sees the whole session's items; only mark the ones that live here.
        if _GOLDEN_DIR in pathlib.Path(item.path).resolve().parents:
            item.add_marker(pytest.mark.gpu)
            if skip is not None:
                item.add_marker(skip)
