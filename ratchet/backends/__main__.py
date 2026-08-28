"""Small diagnostic CLI for explicit backend discovery state."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum

from ratchet.backends import BackendKind, BackendUnavailableError, get_backend


def _json_default(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a Ratchet backend")
    parser.add_argument(
        "--backend", choices=[kind.value for kind in BackendKind], required=True
    )
    arguments = parser.parse_args()
    backend = get_backend(arguments.backend)
    capabilities = backend.capabilities()
    payload: dict[str, object] = {"capabilities": asdict(capabilities)}
    try:
        payload["identity"] = asdict(backend.probe())
    except BackendUnavailableError as error:
        payload["unavailable_reason"] = error.reason
    print(json.dumps(payload, default=_json_default, sort_keys=True))
    return 0 if capabilities.availability.value == "available" else 2


if __name__ == "__main__":
    raise SystemExit(main())
