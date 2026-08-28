#!/usr/bin/env bash
# Adapter between Beryl's affected-test engine and pytest-testmon.
#
# Beryl appends the changed files as positional arguments (jest --findRelatedTests
# style). pytest-testmon does not work that way: it selects affected tests from its
# own dependency DB, and passing non-test source files as positional args makes
# pytest collect nothing and exit 4/5, which reads as a failure. So: ignore the
# appended arguments, let testmon choose, and treat "no tests affected" (exit 5)
# as success -- that is testmon saying nothing needs to run, not an error.
set -uo pipefail
python3 -m pytest --testmon
rc=$?
[ "$rc" -eq 5 ] && exit 0
exit $rc
