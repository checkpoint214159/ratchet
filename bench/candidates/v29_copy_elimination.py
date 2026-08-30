"""Candidate v29 -- stop paying for the graph's output copy on every call.

Generation 29. Parent: v26_causal_correct (the frontier). Branch: cand/g29/copy-elimination.

This is a PORT of `cand/g21/double-buffered` (v21_double_buffered) onto the current
frontier, with three deliberate changes. g21 built against v18 at 76.9 ms wall; the
frontier is now v26 at 69.2 ms, so the same absolute copy is a larger fraction of a
smaller number, and v23's single-tile attention changed the buffer the graph carries.

THE DEFECT
----------
The frontier's steady state is three operations, not one::

    self._static_x.copy_(x)        # DtoD copy of the INPUT
    self._graph.replay()
    return self._static_y.clone()  # DtoD copy of the OUTPUT

A fresh profile of v26 at config 6's shape bills `Memcpy DtoD` at **7.2% of forward
time**, two calls per forward. Roughly half of that is the output clone.

The input copy is NOT removable: a captured graph reads from a fixed address and the
caller's tensor is somewhere else every call. Only the output copy has an alternative.

WHY THE CLONE IS THERE, AND WHY DELETING IT IS A BUG
----------------------------------------------------
`_static_y` is overwritten by the next `replay()`. Finding 24 records four candidates
(v9a, v9b, v11, v15) that returned a static buffer and had the caller's tensor rewritten
underneath them -- a silent wrong answer that sat in the archive for a week with the test
written to catch it reporting green. Deleting the clone is not an optimization; it is
finding 24 again.

WHAT THIS DOES INSTEAD: ASK BEFORE CLOBBERING
---------------------------------------------
Before replaying into the buffer, ask whether anything outside this object still refers to
what was handed out of it last time.

  * nothing does  -> replay. **No copy at all.** This is what both timing loops hit: the
    reference benchmark (lines 494 and 504) and `run_matrix.median_ms` both call
    `model(x, mask)` as a bare statement and discard the result before the next call.
  * the caller still holds the tensor, un-aliased -> copy the buffer's contents into fresh
    memory and rebind the caller's tensor onto it (`Tensor.set_`), preserving its value
    exactly. Cost: one clone -- **identical to the parent**, never worse. This is what the
    accuracy loops hit: `candidate = optimized(x, mask)` leaves the previous result bound
    to `candidate` while the next call runs.
  * the caller holds an alias we cannot rebind -> we cannot preserve it, so we do not
    destroy it. Zero-copy is switched off and this call is served from the compiled
    callable. Slower and correct.

The result is faster when nobody is holding the output, exactly as expensive as the parent
when somebody is, and never silently wrong.

CHANGE 1 vs g21 -- MECHANISM 1 (DOUBLE BUFFERING) IS DROPPED, ON g21's OWN EVIDENCE
------------------------------------------------------------------------------------
g21 shipped a second graphExec with a second static output buffer, replays alternating, so
a handout survives N-1 calls rather than 0. Its own docstring says the sweep does not
measure it: both timing loops discard the result, so N=1 already collects the entire win,
and N=2 only covers the `out = model(x)` idiom -- which the liveness check ALREADY covers,
at exactly the parent's cost. Against that, the second capture costs:

  * a whole second working set. A capture allocates into its own pool, so the memory gate
    declines on the largest shapes -- **configs 6 and 13, which are the only two where the
    copy is worth anything at all.** The mechanism is structurally absent where the prize
    is, and present only where the prize is ~0.
  * a second capture is a second chance to fail, on the mechanism (findings 17 and 23)
    whose failure mode is a stale buffer.
  * it weakens the safety tests: with two buffers, a held-output test at depth 1 passes by
    rotation and never exercises the liveness check. g21's own test had to special-case
    `depth >= len(cand._bufs)` for this reason.
  * the cheap variant -- sharing the graph pool so only the output is duplicated -- is
    MEASURED WRONG on this workload. g21 tried it: the allocator handed capture 1's output
    an address capture 0 uses for an intermediate, and replaying graph 0 destroyed
    graph 1's result. That is why the expensive variant is the only one on offer.

L17 says an evolutionary loop only ever adds, and that ablation is the only mechanism that
subtracts. This subtracts. The double-buffered variant is preserved on
`cand/g21/double-buffered` with its tests; carrying it into the frontier would be inert
complexity with a memory bill on exactly the two configs that matter.

CHANGE 2 vs g21 -- THE LIVENESS SENSOR IS CALIBRATED AND PROVEN ABLE TO FIRE
----------------------------------------------------------------------------
g21 hardcoded the alias threshold: `_storage_use_count(buf) > 2`, on the reasoning that
buf's TensorImpl plus the handout's makes 2. That is true on this build of torch and is
guaranteed by nothing; if a future build made the baseline 3, the check would report
"free" forever and this candidate would silently become finding 24.

Here the baseline is MEASURED at arm time, on the actual buffer, and the sensor is proven
capable of firing before it is trusted (L38): make a view of the buffer, assert the count
rises; drop it, assert the count returns. If either fails, zero-copy is refused and the
parent's clone stands. `zero_copy_reason` records why.

Measuring it also closed the hole g21 said it could not: a caller who keeps
`out.untyped_storage()` and drops `out` DOES raise the count, and is therefore seen. What
remains genuinely invisible is a caller holding a raw integer from `data_ptr()`, which no
design in Python can observe.

CHANGE 3 vs g21 -- AN ALIAS EVENT NO LONGER COSTS THE GRAPH PERMANENTLY
------------------------------------------------------------------------
g21 retired the buffer forever on an un-rebindable alias and served every subsequent call
from the compiled callable -- v13's fallback, which is worth -7.9% (L20/L21: owning the
graph instead of Dynamo is what bought that). A caller who took one slice, once, paid for
it on every call thereafter.

Here the degradation is proportional. Zero-copy is off for good, but the graph comes back
as soon as the alias is released: while anything still shares the buffer's storage the
call is served from the compiled callable, and the moment the count returns to its
measured baseline the buffer is replayed into and cloned out -- **the parent's exact
behaviour at the parent's exact cost.** Worst case equals g21; typical case equals v26.

WHAT THIS DOES NOT DO, AND WHY
-------------------------------
The input copy could also be elided, by remembering `x.data_ptr()` and `x._version` and
skipping `_static_x.copy_(x)` when both are unchanged -- the graded timing loop passes the
same tensor every iteration, so it would fire on every call. Rejected: the version counter
does not see a write performed through a raw pointer, or through a storage aliased outside
autograd's view, so the failure mode is a silently stale INPUT -- finding 17's shape, on
the half of the copy that is harder to reason about. Not worth 3% of one config.

HONEST CEILING, STATED BEFORE MEASURING (L29)
----------------------------------------------
The output copy is ~3.6% of config 6 (half of a measured 7.2% `Memcpy DtoD`), ~2.5% of 13,
~1.3% of 8, ~0 on the other eleven. **That is inside the +/-7% noise floor and inside
config 6's own observed replicate spread.** A geomean verdict cannot resolve it, and three
of the four screen configs (2, 7, 10) are shapes where it is worth nothing at all. This is
L39's shape: the measurement that ranks candidates cannot see this fix. What CAN see it is
a replicated, interleaved, two-arm comparison restricted to configs 6 and 13.

NUMERICS ARE UNCHANGED. The same compiled callable is replayed over the same buffers; no
arithmetic, dtype, algorithm or tile is touched. This candidate cannot be more or less
accurate than its parent, and a test asserts the answer is bit-identical.

CAUSALITY IS INHERITED, NOT ASSUMED. v26 delegates a non-causal config to the unmodified
baseline (finding 32). Because this candidate takes over `forward` on the graph path it
would bypass that check, so the check is re-stated here as the first thing `forward` does,
and a test pins both settings.
"""

from __future__ import annotations

import sys

import torch

from .v26_causal_correct import build as build_v26


def _storage_use_count(t: torch.Tensor) -> int:
    """How many things share `t`'s storage: TensorImpls, and held UntypedStorage objects.

    MEASURED on this build (and pinned by a test): a fresh tensor reads 1; `t.detach()`
    makes it 2; a view of that makes it 3; and -- the case g21 believed it could not see --
    a caller who keeps `out.untyped_storage()` and drops `out` ALSO reads 3. The temporary
    storage object this function itself creates does not count itself.

    What it genuinely cannot see is a caller holding a raw integer address from
    `data_ptr()`. Nothing in Python can see that.
    """
    return torch._C._storage_Use_Count(t.untyped_storage()._cdata)


def build(baseline_cls):
    v26_cls = build_v26(baseline_cls)

    class CandidateV29(v26_cls):
        # "unbuilt" -> no graph yet | "on" -> handing out the buffer
        # "clone"   -> an alias was seen; parent behaviour, permanently
        # "refused" -> the sensor could not be trusted; parent behaviour, permanently
        zero_copy: str = "unbuilt"
        zero_copy_reason: str = "not built"

        # Observable counters. These are what the tests assert on, and what keeps the
        # mechanism from rotting into dead code: reinstate the clone and `output_copies`
        # stops being zero on the discard pattern.
        output_copies: int = 0
        zero_copy_returns: int = 0
        preserve_rebinds: int = 0
        alias_events: int = 0
        fallback_calls: int = 0

        # ------------------------------------------------------------------- arming

        def _arm_zero_copy(self) -> None:
            """Calibrate the alias sensor on the real buffer, and prove it can fire.

            Called once, after the parent chain has captured AND verified the graph. Any
            failure here leaves `zero_copy == 'refused'` and the parent's clone standing --
            the direction a component whose failure mode is silence must fail in.
            """
            self._zc_armed = True
            self._handout = None
            buf = getattr(self, "_static_y", None)
            if buf is None:                          # pragma: no cover
                self.zero_copy = "refused"
                self.zero_copy_reason = "refused: no static output buffer to hand out"
                return
            try:
                n0 = _storage_use_count(buf)
                probe = buf.reshape(-1)[:1]          # an alias we make ourselves
                n1 = _storage_use_count(buf)
                del probe
                n2 = _storage_use_count(buf)
            except Exception as exc:                 # pragma: no cover
                self.zero_copy = "refused"
                self.zero_copy_reason = (
                    f"refused: the alias sensor raised {type(exc).__name__}: {exc}")
                return
            # L38: a guard is only evidence if it is capable of firing. If the count does
            # not rise for an alias we deliberately created, it will not rise for the
            # caller's either, and "free" would mean nothing.
            if n1 <= n0:
                self.zero_copy = "refused"
                self.zero_copy_reason = (
                    f"refused: the alias sensor did not respond to a deliberate view "
                    f"({n0} -> {n1}); it cannot be trusted to report the caller's")
                return
            if n2 != n0:
                self.zero_copy = "refused"
                self.zero_copy_reason = (
                    f"refused: the alias sensor did not return to baseline after the probe "
                    f"was dropped ({n0} -> {n1} -> {n2}); it is not a live count")
                return
            self._base_use = n0
            self.zero_copy = "on"
            self.zero_copy_reason = (
                f"on: alias sensor calibrated at {n0} on the static output buffer and "
                f"verified able to fire; the output clone is now paid only when the caller "
                f"kept the previous one")

        # ----------------------------------------------------------------- liveness

        def _verdict(self) -> str:
            """May we overwrite the buffer? 'free' | 'rebind' | 'aliased'."""
            if self._handout is None:
                return "free"
            try:
                # An alias we cannot rebind -- a view, a slice, or a retained
                # `untyped_storage()` -- pushes the storage count past baseline + 1.
                if _storage_use_count(self._static_y) > self._base_use + 1:
                    return "aliased"
            except Exception:                        # pragma: no cover
                return "aliased"                     # conservative
            # NB: never bind the handout to a local before this call. `sys.getrefcount`
            # counts every live reference, so a convenience local makes the count read one
            # too high and the check reports "the caller is holding it" on EVERY call --
            # a candidate that measures as its own parent, in silence. g21 shipped that
            # bug once; `test_no_copy_on_the_pattern_both_harnesses_use` is what caught it.
            if sys.getrefcount(self._handout) <= 2:
                return "free"
            return "rebind"

        def _to_clone_mode(self, why: str) -> None:
            """Give up zero-copy for good, but not the graph.

            g21 retired to the compiled callable permanently. Here the graph returns as
            soon as the alias is released: `forward` re-checks the storage count each call
            and replays the moment it is back at baseline, which is exactly the parent.
            """
            self.alias_events += 1
            self._handout = None
            self.zero_copy = "clone"
            self.zero_copy_reason = (
                f"clone-on-return: {why}. The graph is still replayed and cloned out "
                f"(the parent's cost) on every call the buffer is unaliased.")

        # ------------------------------------------------------------------ forward

        def _replay(self, x, valid_token_mask) -> None:
            self._static_x.copy_(x)
            if self._static_m is not None:
                self._static_m.copy_(valid_token_mask)
            self._graph.replay()

        def forward(self, x, valid_token_mask=None):
            # Finding 32, restated because this method bypasses v26's own copy of it.
            # The reference benchmark's DEFAULT is causal=False; the optimized path is
            # only correct under causality and delegates otherwise.
            if not getattr(self.config, "causal", True):
                self.causal_path = "baseline (non-causal input)"
                return baseline_cls.forward(self, x, valid_token_mask)
            self.causal_path = "optimized (causal input)"

            if getattr(self, "_graph", None) is None:
                # Priming, tile selection, compilation, capture, verification and the
                # no-graph fallback all live in the parent chain and are unchanged.
                # Whatever it returns is already safe: it clones, or it is a fresh
                # allocation from the compiled callable.
                out = super().forward(x, valid_token_mask)
                if (getattr(self, "_graph", None) is not None
                        and not getattr(self, "_zc_armed", False)):
                    try:
                        self._arm_zero_copy()
                    except Exception as exc:         # pragma: no cover
                        self._zc_armed = True
                        self._handout = None
                        self.zero_copy = "refused"
                        self.zero_copy_reason = (
                            f"refused: arming raised {type(exc).__name__}: {exc}")
                return out

            buf = self._static_y

            if self.zero_copy == "on":
                verdict = self._verdict()
                if verdict == "aliased":
                    self._to_clone_mode(
                        "a caller aliased a returned tensor and an alias cannot be rebound")
                    self.fallback_calls += 1
                    return self._compiled_core(x, valid_token_mask)
                if verdict == "rebind":
                    held = self._handout
                    self._handout = None
                    try:
                        held.set_(held.clone())
                    except Exception:
                        # Cannot preserve it; do not destroy it either.
                        self._to_clone_mode(
                            "a returned tensor is still held and could not be rebound")
                        self.fallback_calls += 1
                        return self._compiled_core(x, valid_token_mask)
                    self.preserve_rebinds += 1
                    self.output_copies += 1
                else:
                    self._handout = None

                self._replay(x, valid_token_mask)
                handout = buf.detach()   # a distinct TensorImpl, so aliases are countable
                self._handout = handout
                self.zero_copy_returns += 1
                return handout

            # "clone" or "refused": the parent's behaviour, whenever the buffer is ours.
            if self.zero_copy == "clone":
                try:
                    still_aliased = _storage_use_count(buf) > self._base_use
                except Exception:                    # pragma: no cover
                    still_aliased = True
                if still_aliased:
                    self.fallback_calls += 1
                    return self._compiled_core(x, valid_token_mask)

            self._replay(x, valid_token_mask)
            self.output_copies += 1
            return buf.clone()

    return CandidateV29
