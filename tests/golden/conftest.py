"""Guard for the golden suite: everything in this directory needs a real CUDA device.

The golden tests pin the oracle's observable behavior -- locked constants, input bits,
gate mechanics, the fp64 error floor. They are the tripwire that fires when anyone
weakens the correctness gate. A tripwire that errors out with "no CUDA device" on a
CPU-only box looks like oracle breakage and trains people to ignore red, so instead the
whole directory is skipped, loudly and with a reason, when there is no GPU to run on.

We add skip markers from pytest_collection_modifyitems rather than calling
pytest.skip(allow_module_level=True) at conftest import time: a module-level skip in a
conftest aborts pytest's configuration stage with a traceback instead of reporting a
clean skip.
"""

import pathlib

import pytest
import torch

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="golden oracle tests require a CUDA device")
    for item in items:
        # This hook sees the whole session's items; only mark the ones that live here.
        if _GOLDEN_DIR in pathlib.Path(item.path).resolve().parents:
            item.add_marker(skip)
