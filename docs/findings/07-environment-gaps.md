# Finding 07 — Two environment gaps that are not code bugs

Recorded 2026-08-29. Neither was introduced by the bench work; both were verified against
commit `2db7266`, before any of it landed.

## 1. `pytest` could not collect three test files

Three of the teammate's suites failed collection with
`ModuleNotFoundError: No module named 'tests.fixtures'`.

Cause: a `tests` package installed in system `dist-packages` **shadows this repo's
`tests/`** under pytest's default `prepend` import mode, so `from tests.fixtures import
...` resolves to the wrong package entirely.

Fix, in `pyproject.toml` rather than in anyone's test code:

```toml
[tool.pytest.ini_options]
addopts = "--import-mode=importlib"
```

That converts 3 collection errors into runnable tests, taking the suite from "cannot
collect" to 353 passing.

## 2. `git merge-tree --write-tree` needs git >= 2.38; this machine has 2.34.1

18 tests fail, 14 of them in `tests/experiments/test_workspaces.py`, with:

```
WorkspaceLifecycleError: usage: git merge-tree <base-tree> <branch1> <branch2>
```

`ratchet/experiments/workspaces.py` calls the modern `--write-tree` form of
`git merge-tree`, which landed in **git 2.38** (Oct 2022). Ubuntu 22.04 ships
**2.34.1**, which only understands the older three-tree form and prints that usage line.

Verified pre-existing: checking out `2db7266` into a scratch worktree reproduces exactly
the same 14 failures.

**Not fixed here, deliberately.** The two options are upgrading git on this machine or
adding a fallback path to the workspace consolidator, and the second is a change to a
teammate's bounded context that they should make. Recorded so the next person does not
spend the time re-diagnosing it.

**Consequence worth knowing:** experiment-worktree consolidation — the mechanism intended
to let several people run candidates in parallel lanes and merge the results — cannot run
on this machine as configured. The `bench/` lane does not depend on it.
