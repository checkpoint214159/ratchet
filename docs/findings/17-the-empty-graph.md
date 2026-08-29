# Finding 17 — Our best candidate could return stale output, and the harness was hiding it

Recorded 2026-08-29. Candidates `v12_graph_over_compile` (the bug) and `v13_safe_capture`
(the fix).

## How it surfaced

Profiling config 12 — an ordinary follow-up to finding 16, expecting to find where its
1.40x was going — v12 did not produce a profile. It raised:

```
UserWarning: The CUDA Graph is empty. This usually means that the graph was
attempted to be captured on wrong device or stream.
InternalTorchDynamoError: RuntimeError: Cannot call
CUDAGeneratorImpl::current_seed during CUDA graph capture.
```

Dynamo re-traced *inside* the capture region, touched the RNG state, and CUDA rejected the
capture. The only difference from the harness was the call pattern: the profiler calls the
candidate directly on one tensor, where the harness runs five accuracy trials on fresh
inputs first, which happens to settle Dynamo before capture is attempted.

## The dangerous half is the warning, not the exception

An exception is loud and survivable. **An empty graph is not.** `replay()` then executes
nothing, `_static_y` retains whatever the warmup left in it, and the candidate returns a
tensor of exactly the right shape and dtype containing stale values.

This is the failure mode this project's own notes flagged from the beginning as the most
likely silent wrong answer from graph capture (see finding 03's fragilities list). It took
until generation 13 to actually hit it.

**v12's published numbers are sound.** The harness runs correctness before timing, and
five accuracy trials on fresh inputs cannot pass against a stale buffer. But *"correct
because the harness happens to call it in the right order"* is not a property worth
shipping — a grader whose harness warms differently would get silent garbage.

## The fix, and what it cost

v13 captures defensively in three parts: settle Dynamo on the **default** stream first
(side-stream warmup alone does not settle it); wrap capture in try/except; and then
**verify the graph is real** by replaying it against a freshly computed reference. An
empty graph fails that check by construction, because the static output will not have been
rewritten from the new input.

Any failure leaves `_graph` as None and every call falls through to the compiled callable
— v11's path, correct and about 7.9% slower.

| | geomean vs compiled |
|---|---|
| v12 (fast, latently unsafe) | 2.712x |
| **v13 (verified or unused)** | **2.711x** |

**The safety costs nothing measurable.** Degrading to slower-and-correct is the only
acceptable direction for a component whose failure mode is silence, and here it did not
even cost that.

## The lesson about test design

Four tests now pin this, and the one that matters is the crudest: **run two different
inputs and assert the outputs differ.** No tolerance reasoning, no reference comparison —
just the observation that a function of its input must not return the same thing twice.

Every accuracy check we had ran a single input per trial against a reference. That is
blind to staleness by construction, because a stale buffer holding a *correct previous
answer* still matches a reference computed for that same previous input. The bug was
invisible to the entire correctness suite and would have stayed invisible.
