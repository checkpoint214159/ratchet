#!/usr/bin/env bash
# Zone A integrity. Detection, not prevention -- but it turns a silent catastrophe into
# a loud one, which is the realistic goal.
set -euo pipefail
cd "$(dirname "$0")/.."
MANIFEST="ratchet/oracle/.manifest.sha256"

fail=0

if [ ! -f "$MANIFEST" ]; then
  echo "no manifest; creating one (do this deliberately, on a clean tree)"
  find ratchet/oracle -name '*.py' -print0 | sort -z | xargs -0 sha256sum > "$MANIFEST"
  echo "created $MANIFEST"
else
  if ! sha256sum -c --quiet "$MANIFEST" 2>/dev/null; then
    echo "ORACLE MODIFIED. Zone A is immutable during optimization."
    echo "If a kernel only passes after this change, the kernel is wrong."
    echo "If the change is legitimate: separate branch, human approval, then"
    echo "  find ratchet/oracle -name '*.py' -print0 | sort -z | xargs -0 sha256sum > $MANIFEST"
    fail=1
  else
    echo "oracle: OK"
  fi
fi

# The ledger must never be opened for writing in anything but append mode.
if grep -rnE '(measurements\.jsonl["'"'"']?\s*,\s*["'"'"'](w|r\+))|open\([^)]*measurements[^)]*["'"'"']w' \
     --include='*.py' ratchet/ 2>/dev/null; then
  echo "LEDGER OPENED FOR WRITING. Zone C is append-only."
  fail=1
else
  echo "ledger append-only: OK"
fi

# Tolerances must not be reassigned anywhere outside their definition.
if grep -rnE '^\s*(REL_TOL|ABS_TOL)\s*=' --include='*.py' ratchet/ \
     | grep -v 'ratchet/oracle/correctness.py'; then
  echo "TOLERANCE REDEFINED OUTSIDE THE ORACLE. These are locked constants."
  fail=1
else
  echo "tolerances locked: OK"
fi

exit $fail
