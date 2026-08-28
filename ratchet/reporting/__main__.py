"""Command-line entry point for deterministic reporting."""

from __future__ import annotations

import argparse

from .paper import PaperBuildError, build_paper


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m ratchet.reporting")
    parser.add_argument("command", choices=("build-paper",))
    parser.parse_args()
    try:
        artifact = build_paper()
    except PaperBuildError as error:
        parser.error(str(error))
    print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
