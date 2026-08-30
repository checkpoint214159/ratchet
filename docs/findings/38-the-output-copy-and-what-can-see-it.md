# 38 — The graph's output copy, and the fact that nothing we run can see it

**Date:** 2026-08-30. **Candidate:** `v29_copy_elimination`, branch `cand/g29/copy-elimination`,
parent `v26_causal_correct`. **Ports:** `cand/g21/double-buffered` (v21), which built the
same mechanism against v18.

## The defect

The frontier's steady state is three operations, not one:

    self._static_x.copy_(x)        # DtoD copy of the INPUT
    self._graph.replay()
    return self._static_y.clone()  # DtoD copy of the OUTPUT

A profile of v26 at config 6's shape bills `Memcpy DtoD` at **7.2% of forward**, two calls
per forward. The input copy is not removable — a captured graph reads a fixed address. The
output copy is, but not by deleting it: finding 24 is four candidates that returned a
static buffer and had the caller's tensor rewritten underneath them.

v29 asks before clobbering. Nothing refers to the previous handout → replay, no copy. The
caller holds it un-aliased → rebind that tensor onto a fresh clone, which costs exactly
what the parent costs. The caller holds an alias that cannot be rebound → stop handing the
buffer out. Both timing loops hit the first case; both accuracy loops hit the second.

## What changed from g21, and why

g21 measured against v18 at 76.9 ms; the frontier is v26 at 69.2 ms, so the same absolute
copy is a larger share of a smaller number, and v23's single-tile attention changed the
buffer the graph carries. Four changes, each with a reason that is not "the frontier moved".

**1. Double buffering is dropped.** g21's own docstring says the sweep does not measure it:
both timing loops discard the result, so N=1 collects the whole win, and the `out =
model(x)` idiom is already covered by the liveness check at the parent's cost. Against
that, a second capture allocates into its own pool, so its memory gate declines on
**configs 6 and 13 — the only two shapes where the copy is worth anything**. The mechanism
is structurally absent where the prize is and present only where the prize is zero. It
also weakened the safety tests: with two buffers a held-output test at depth 1 passes by
rotation, and g21's own test had to special-case that. L17 says only ablation subtracts.

The cheap variant — sharing the graph pool so only the output is duplicated — stays dead.
g21 MEASURED it wrong on this workload: the allocator gave capture 1's output an address
capture 0 uses for an intermediate, and replaying graph 0 destroyed graph 1's result. That
is why the expensive variant was the only one on offer, and why dropping it costs nothing.

**2. The alias sensor is calibrated, not hardcoded.** g21 wrote `_storage_use_count(buf) >
2`, reasoning that the buffer's TensorImpl plus the handout's makes 2. True on this torch,
guaranteed by nothing; a build where the baseline were 3 would report "free" forever and
the candidate would silently become finding 24. v29 measures the baseline on the actual
buffer at arm time and proves the sensor can fire before trusting it (L38): make a view,
assert the count rises; drop it, assert it returns. Either failure refuses zero-copy and
leaves the parent's clone standing, with the reason in `zero_copy_reason`.

**3. An alias event no longer costs the graph permanently.** g21 retired the buffer forever
and served every later call from the compiled callable — giving up the whole +7.9% that
owning the graph bought (L20/L21) because one caller took one slice. v29 keeps zero-copy
off but brings the graph back the moment the alias is released: while the storage is shared
the compiled callable serves; when the count returns to baseline the buffer is replayed
into and cloned out, which is the parent exactly. Worst case equals g21; typical case
equals v26.

**4. The per-call bookkeeping was costed.** The saving is a device-side copy; the price is
Python in `forward`, and config 2 is a ~60 µs call that L20 measured as CPU-dispatch-bound.
Measured on CPU (no GPU timing, so no contention):

| operation | µs/call |
|---|---|
| `untyped_storage()` + `_storage_Use_Count` | 0.209 |
| cached `_cdata` + `_storage_Use_Count` | 0.098 |
| `buf.detach()` | 0.464 |
| `sys.getrefcount` | 0.034 |

The detach is removable: on the branch where the check has just proved nobody holds the
previous handout, that TensorImpl is ours to hand out again. Bookkeeping drops from
~0.71 µs to ~0.25 µs, against a ~2.2 µs launch and an allocation saved by not cloning.

## Two things that were measured rather than assumed

**g21's stated hole is not a hole.** Its finding says a caller retaining
`untyped_storage()` rather than a tensor is invisible to the check. Measured: a held
`UntypedStorage` raises the storage use count exactly like a view does, so the guard fires
and the buffer is never clobbered. A test pins it. What remains genuinely invisible is a
caller holding a raw integer from `data_ptr()`, which nothing in Python can observe.

**And the 0.11 µs in the table above was taken, then given back, because taking it broke
the guard.** Caching the storage handle at arm time turns the per-call check into one C
call — and `tensor.untyped_storage()` returns the *same Python object* every time, so the
handle we hold is the one the caller receives. Holding it permanently folded the caller's
reference into our own calibrated baseline, and the guard went blind to exactly the case
that had just been closed. `test_a_caller_who_kept_only_the_storage_is_seen` failed on the
first run after the change.

That is the whole shape of this candidate in one line: **a performance optimization
silently disabled the safety check that the same commit had just built**, and the only
reason it was caught in ten seconds rather than in a ledger row is that the test for the
hole existed *before* the optimization did.

## The measurement problem, which is the important part

**Predicted before measuring (L29):** ~3.6% on config 6 (half of a measured 7.2% `Memcpy
DtoD`), ~2.5% on 13, ~1.3% on 8, ~0 on the other eleven.

`bench/screen.py` was run twice, on configs 2, 7, 8, 10. Both said REJECT. **They do not
agree with each other, and neither is about the candidate.**

| config | v26 (ledger) | screen run 1 | screen run 2 |
|---|---|---|---|
| 2 | 0.0614 ms | 0.0594 | **0.1155** |
| 7 | 0.0870 ms | 0.0840 | 0.0850 |
| 8 | 6.5485 ms | 6.4614 | 6.4942 |
| 10 | 0.2447 ms | **0.8735** | 0.2509 |
| verdict | — | REJECT −25.7% | REJECT −14.4% |

Configs 7 and 8 land within 1% of the parent in both runs. Each run has exactly **one**
wild row, and it is a different row each time — both of them sub-millisecond. Run 1's
config 10 at 3.6× the parent coincided with another agent's ad-hoc CUDA probe on the same
GPU (finding 26 / L38: `run_matrix` takes the lock, an ad-hoc probe does not). Run 2 was
taken after waiting for the device to go idle, and config 10 came back to 0.2509 — but
config 2 then swung 1.9× against its own run-1 value **on identical code**.

Diagnosis, not excuse: a direct state probe at config 10's shape shows the mechanism fully
engaged — `graph_verified True`, `zero_copy on`, 24 zero-copy returns, **0 output copies**.
And the entire per-call CPU cost is 0.25 µs, so a 1.9× swing on a 60 µs call is not
something this candidate is capable of causing.

**The screen set contains no shape where this candidate can win.** Config 8 is the only one
with any of the prize (~1.3%, inside the floor), and configs 2, 7 and 10 are sub-millisecond
rows whose observed run-to-run spread here is up to 1.9×. A screen verdict on this
candidate is a coin flip dressed as a decision — L39's shape, and a stronger case than v18's
because here the *screen's own resolution*, not just its coverage, is the problem.

## The experiment that would resolve it

Not a sweep, not a geomean.

1. **Configs 6 and 13 only** (add 8 if there is budget). Everything else is predicted flat
   and only adds variance to an aggregate.
2. **Paired and interleaved across processes**: v26, v29, v26, v29, … at least 5 replicates
   per arm, each arm alone on the device (finding 05 forbids co-residency inside one
   process, but alternating *subprocesses* restores the thermal-drift defence that
   `run_matrix` currently gives up).
3. **Hold the GPU lock for the whole sequence**, and make sure no other agent is probing —
   run 1 above is what that costs.
4. **Report per-config paired deltas with the replicate spread**, never a geomean. L42
   already noted the geomean weights a 0.06 ms config equally with a 57 ms one, and this
   candidate is the case where that matters most.
5. Feasibility: L42 measured that configs above a millisecond reproduce within 0.6%. A 3.6%
   effect against a 0.6% spread is resolvable at 5 replicates. **The effect is inside the
   global noise floor but well outside config 6's own floor** — which is exactly why the
   experiment must be per-config.

**And one confirmation that needs no timing at all:** profile one forward of each arm at
config 6's shape and count `Memcpy DtoD` calls. v26 must show 2 per forward, v29 must show
1. That is a mechanism check, it is cheap, it cannot be contended into a wrong answer, and
if it fails then no amount of timing matters.

## Proposed lessons

Offered for `00-learnings.md`; the loop appends, not the executor (L41).

* **A screen's resolution is a property of its configs, not of its protocol.** The screen
  set was derived to be cheap and to span regimes (finding on `bench/screen.py`), and both
  are still true — but three of its four rows are sub-millisecond, and a sub-millisecond row
  on an unlockable-clock card can move 1.9× between runs of identical code. For any
  candidate whose predicted effect is single-digit percent, the screen cannot reject; it can
  only detect a catastrophe. `decide()` should distinguish "clearly worse" from "the screen
  cannot see this", the way finding 26's addendum made it distinguish "not measured".
* **An optimization can silently disable the safety check that justifies it.** Caching a
  storage handle to save 0.11 µs blinded the alias guard, because the object it caches is
  the same object the caller gets. Whenever a change makes a *guard* cheaper, re-run the
  test that proves the guard can fire — not the tests that prove it passes.
* **"I could not close this hole" is a claim that deserves a measurement.** g21 recorded the
  retained-storage case as unreachable. Six lines of probe showed the existing check already
  caught it. Inherited limitations rot the same way inherited defaults do (L27, L42).
