# Finding 18 — Three latent bugs found by asserting the obvious

Recorded 2026-08-29. `tests/bench/test_lineage_invariants.py`.

## What was swept, and why

Finding 17 exposed a bug our whole accuracy suite was structurally blind to: v12 could
replay an empty CUDA graph and return a stale buffer. The suite could not see it because
every accuracy check runs **one input per trial against a reference computed for that same
input** — and a stale buffer holding a correct *previous* answer matches the reference for
the *previous* input.

Any candidate that caches, captures, or holds mutable state across calls is exposed to the
same class of bug. So rather than fix one candidate, the whole registry was swept against
three invariants that need no tolerance reasoning and no reference implementation:

1. different inputs must not produce identical output;
2. the same input must produce the same output, even after intervening calls;
3. the tensor handed to the caller must not change underneath them on the next call.

## Result: 3 failures across 15 candidates, all of which passed the accuracy suite

| candidate | invariant broken | cause |
|---|---|---|
| v12_graph_over_compile | different inputs → identical output | the empty-graph staleness of finding 17, now reproducible in a test |
| v10b_no_fusedqkv | returned tensor mutated by the next call | returns Inductor's static output buffer |
| v10c_no_fp16 | returned tensor mutated by the next call | same |

**v13 passed all three**, confirming its verified-capture fix.

## The v10 failures are a separate bug, and a general one

`torch.compile(mode="reduce-overhead")` installs CUDA graphs internally and **returns a
tensor backed by a static buffer that the next call overwrites**. Handing that straight to
a caller who holds it across calls is a silent data race.

It never corrupted a measurement, because the harness computes both outputs and compares
them before calling anything again — so the hazard is invisible under our call pattern and
would surface under a different one. That is exactly the shape of L24: *correct because of
how the harness calls it, not on its own terms.*

Fixed by cloning the compiled output. The same reasoning applies to any candidate using
`reduce-overhead`, so it is worth stating as a rule: **if the compiler owns graph capture,
clone before returning.**

## v12 is retained but excluded, not hidden

v12 stays in the registry — it is real lineage, it produced finding 16, and its
measurements are sound under the harness. It is excluded from the invariant sweep with the
reason written into the test, alongside v5 (the known-incorrect stepping stone from
finding 08). Neither is a submission candidate.

Deleting them would destroy the record of why v13 and v6 exist.

## The methodological point

All three bugs were found by asserting things that sound too obvious to test: *a function
of its input should depend on its input.* The elaborate correctness machinery — locked
tolerances, nine input distributions, FP64 references, known-bad fixtures — could not see
any of them, because every one of those checks assumes the output corresponds to the input
it was just given.

**Invariance tests and equivalence tests catch disjoint bug classes.** A suite with only
the latter has a blind spot the size of every stateful component in the system.
