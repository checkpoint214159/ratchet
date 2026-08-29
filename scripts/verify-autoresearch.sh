#!/usr/bin/env bash
# Verify Ratchet's current autoresearch state without executing a GPU candidate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${RATCHET_PYTHON:-${ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'ERROR: create .venv with `uv venv --python 3.12 .venv` and `uv pip install --python .venv/bin/python -e ".[dev]"`.\n' >&2
  exit 2
fi

cd "${ROOT}"
"${PYTHON}" -c '
from pathlib import Path
from ratchet.experiments import FileExperimentArchive
archive = FileExperimentArchive(Path("research/archive"))
archive.verify()
projection = archive.projection()
print(f"verified archive: {projection.event_count} event(s), {projection.projection_id}")
'
"${PYTHON}" -m ratchet.reporting build-paper
./.beryl/scripts/check.sh
printf 'autoresearch verification: OK (no GPU experiment was run)\n'
