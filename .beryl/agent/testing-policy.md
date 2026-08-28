# Testing Policy

## Command Matrix

| Check | Command | Status | Notes |
| --- | --- | --- | --- |
| Markdown sanity | `./.beryl/scripts/check-md.sh` | available | Unclosed fences and tabs |
| Test manifest immutability check | `./.beryl/scripts/check-tests-unchanged.sh` | available | Detects changes in configured test scope from `.beryl/agent/test-manifest.conf` |
| Affected test gate | `./.beryl/scripts/check-affected.sh --worktree` | available | Selects related tests from changed files and uses full-test fallback for broad changes |
| Aggregate deterministic gate | `./.beryl/scripts/check.sh` | available | Runs all deterministic checks |
| Format | `.venv/bin/ruff format --check .` | available | Ruff 0.16.5 is pinned in the `dev` extra; managed, immutable, and byte-preserved paths are excluded in `pyproject.toml`. |
| Lint | `.venv/bin/ruff check .` | available | Uses the same explicit Ruff surface and baseline rules. |
| Typecheck | `not available yet` | unavailable | Add the project typecheck command when configured |
| Unit tests | `not available yet` | unavailable | Add the project unit test command when configured |
| Integration tests | `not available yet` | unavailable | Add the project integration test command when configured |
| E2E smoke | `not available yet` | unavailable | When web runtime exists, use Microsoft Playwright MCP for deterministic browser feedback |

This repository is an installed Beryl target, not a Beryl source checkout. The broad
command is `./.beryl/scripts/check.sh`; `--development` is invalid without Beryl's
source-checkout marker.

## Authoritative Transformer Matrix

The protected evaluator's executable defaults are the primary compatibility case:

| Case | B | S | D | H | FFN | Layers | Causal | Padding | Dtype | Seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| default | 8 | 128 | 512 | 8 | 2048 | 6 | no | 0 | float32 | 1234 |
| causal | 2 | 257 | 512 | 8 | 2048 | 2 | yes | 0 | float32 | 2234 |
| padded | 4 | 127 | 256 | 8 | 1024 | 2 | no | 0.25 | float32 | 3234 |
| long | 1 | 512 | 512 | 8 | 2048 | 2 | no | 0 | float32 | 4234 |

When the backend probe declares BF16 or FP16 support, repeat the default and long cases
for each supported dtype. Unsupported dtypes are recorded as capability results, not
silently skipped. Correctness uses five trials beginning at the listed seed and the
evaluator's executable absolute-OR-relative thresholds (`atol=0.002`, `rtol=0.02`).

## Accelerator Performance Evidence

- Record compilation time and first-run latency separately from steady-state execution.
- Warm each model for 20 completed calls.
- Use ten alternating ABBA/BAAB blocks with 30 completed calls per model per block.
- Use backend events where supported and cross-check with a host timer bounded by an
  explicit synchronize before and after the measured region.
- Generate the timed input with `seed + 100000` and exclude input creation from timing.
- Record raw samples, median, mean, p90, minimum, standard error, a paired-bootstrap 95%
  speedup interval, peak allocated/reserved memory, clock/power status, and method.
- Candidate and baseline must share dtype, compiler policy, warm-up, process, input,
  and ordering blocks.
- Promotion requires full correctness, baseline and candidate 95% latency intervals not
  overlapping, the paired speedup interval lower bound above 1.02, and no unexplained
  peak-memory increase above 5%.
- If no accelerator is visible, hardware tests skip locally and empirical kernel
  iteration stops. Cached calibration is never accepted as a new result.

The reference evaluator's non-CUDA host-timer output is compatibility output only and
must not be entered as Intel performance evidence.

## Default Loop

1. Identify or add the failing behavior.
2. Select the smallest useful test level.
3. State success checks before implementation: expected artifact, narrow command, broader command, generated output or browser evidence when applicable, and one user-visible behavior.
4. Implement one internal feature slice.
5. Run narrow checks first, then broader checks.
6. Repair from actual tool output.
7. For web UI or HTML/CSS work, include a Playwright MCP browser verification step.

## Generated Output Verification

For static-site changes, source inspection is not enough. Always verify generated output that users, crawlers, or downstream tooling receive.

Check affected:

- Relevant `dist` HTML or equivalent built pages.
- Sitemap, robots, search index, feed, or structured data output.
- Copied assets when asset handling changed.
- Browser behavior when UI, routing, or layout changed.

If generated output is unavailable, explain why and run the closest deterministic build or inspection command.

## Affected Test Gate

Commit-time tests run through the affected test gate so developers get fast feedback without choosing test subsets manually.

- The pre-commit hook sets `CHECK_AFFECTED_MODE=staged` and runs `./.beryl/scripts/check.sh`.
- Manual `./.beryl/scripts/check.sh` uses worktree mode by default and selects from all changes relative to `HEAD`.
- `.beryl/scripts/check-affected.sh` reads `.beryl/agent/affected-tests.conf`.
- Changes to broad configuration, dependency, hook, or test-strategy files force `FULL_TEST_CMD` when configured.
- Source and test changes run `RELATED_TEST_CMD` with changed files appended when configured.
- If no project test runner is configured yet, the gate reports that no project tests are available and exits successfully.

Recommended project configurations:

```bash
# Jest
RELATED_TEST_CMD=(npx --no-install jest --findRelatedTests --passWithNoTests)
FULL_TEST_CMD=(npm test)

# pytest with testmon
RELATED_TEST_CMD=(pytest --testmon)
FULL_TEST_CMD=(pytest)
```

## Test Modification Rule

Existing tests may not be weakened to make implementation pass.

Intentional test changes are allowed only when all conditions are met:

1. The behavior change is explicit in the task or design artifact.
2. `./.beryl/scripts/update-test-manifest.sh` is run after the intentional change.
3. The manifest update is committed with the test change.
4. The final response explains why tests changed.
5. `.beryl/agent/test-manifest.conf` is updated if new test locations/patterns are introduced.

## Immutability Enforcement Scope

- The SHA manifest mechanism provides deterministic change detection, not cryptographic immutability guarantees against privileged users.
- Enforce stronger controls in CI/review policy, such as branch protection, required status checks, and code review.

## Mocking Rules

- Mock external systems such as network, clocks, randomness, payment providers, and email providers.
- Do not mock domain logic in the same bounded context.
