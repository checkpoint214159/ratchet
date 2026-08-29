# 24 — Four candidates carry a silent-wrong-answer bug, and the test suite was hiding it

**Date:** 2026-08-30. **Found by:** the lineage invariant sweep, when v16 was added.
**Affects:** `v9a_compiled_core`, `v9b_reduce_overhead`, `v11_lean`, `v15_lifted_veto`.
**Safe:** `v13_safe_capture`, `v16_ffn_megakernel` (both clone).

## The bug

Four candidates set `use_graph = False`, hand `_core` to `torch.compile` with a mode that
lets **Inductor** capture CUDA graphs (`reduce-overhead`, or `max-autotune`, which applies
graphs itself), and return the compiled callable's result directly:

    return self._compiled(x, valid_token_mask)

A CUDA graph replay writes into a **static output buffer**. Returning it hands the caller
a tensor that the next forward rewrites underneath them. This is finding 17's bug — the
one v13 was built to fix — arriving by a different route: v12 could replay an *empty*
graph, these replay a *correct* graph into a buffer they do not own.

## What makes this finding worth its own file

**The test written to catch exactly this reported all four as passing.**

`test_returned_tensor_survives_the_next_call` has existed since finding 18 and sweeps the
whole registry. Run over the whole file it reported 50 passed / 1 failed. Run one
candidate per process:

    v9a_compiled_core      FAILED
    v9b_reduce_overhead    FAILED
    v16_ffn_megakernel     FAILED   (fixed here)
    v13_safe_capture       passed

The cause is the hazard recorded earlier the same day: **Dynamo's `cache_size_limit` is 8
and shared per process.** Once exhausted, `torch.compile` silently falls back to eager.
Eager allocates a fresh output tensor every call, so the static-buffer test passes
vacuously — the candidate passes because the thing under test *was never compiled*.

So the suite was green for a week while four candidates carried a live defect, and it was
green **because** of a second defect. A test that passes because its subject was never
built is worse than no test: it converts an absence of checking into positive evidence.

The fix is one line in the fixture — `torch._dynamo.reset()` per candidate — and it should
have been there from the moment the cache limit was understood. It was understood at
14:xx today, written into a commit message as a deployment caveat, and not connected to
the suite that it invalidates.

## Why the four are not being retro-fixed

Precedent from v12: a defective stepping stone stays in the archive as lineage, excluded
from the sweep, documented. Adding `.clone()` to v9a/v9b/v11/v15 now would change their
timings — a clone is real work — while their ledger rows were measured *without* it. Those
rows would then describe code that no longer exists. Measurements are append-only and
keyed to a commit precisely so this cannot happen silently.

They are marked `known_unsafe` with the reason, and their recorded speeds should be read
as **slightly optimistic**: they omit a copy that a correct version must perform.

**This touches the frontier claim.** v13 (the frontier, 2.711x) clones and is safe. But
v9a and v9b sit at 2.678x and 2.655x in the scoreboard, and those numbers are for code
that returns a buffer it does not own. They are not shippable at that speed.

## L36 — A test can pass because its subject was never built

Green does not mean checked. This suite reported 113 passing while four candidates held a
silent-wrong-answer bug, because an unrelated resource limit quietly replaced the code
under test with a different implementation that trivially satisfies the assertion.

The general shape: **whenever a test's subject is produced by a lazy, budgeted, or
fallback-capable mechanism — a JIT, a compiler, a cache, a feature flag — the test must
assert that the mechanism actually ran.** v15's mechanism test already does this (it
asserts a `triton_tem` kernel appears); the invariant sweep did not, and that asymmetry is
what let this survive.

Related: L24 said "correct because of how the harness calls it is not correct." This is
its twin — *green because of how the test was run* is not green.
