"""Candidate v37 -- the two lines out of v26 rejoin: v36's projection GEMMs on top of
v35's shape-latch fix, streaming, and reset discipline.

Generation 37. Parent: `v36_gemm_gelu`. Branch: `cand/g37/recombined2`.
RECOMBINATION CONTRIBUTOR: `v35_recombined` (the relationship v17 declared to v16, and
v35 declared to v34).

WHY THESE TWO
-------------
Both descend from `v26_causal_correct`, through different parents, so neither is an
ancestor of the other -- `git merge-base cand/g35/recombined cand/g36/gemm-gelu` is
`cand/g32/persistent-layernorm`. They are orthogonal in the strict sense:

  v35  = v33's SHAPE-LATCHING CORRECTNESS FIX + batch streaming + v34's kernel-count cut
         (36 -> 20). It answers *which shapes are computable, how many times a model may
         be called, and how many kernels one call costs*.
  v36  = a Triton GEMM with an exact-erf GELU epilogue for the narrow-K projections,
         decided per site by TIMING the vendor against 18 swept tiles. It answers *what
         one of those kernels costs*.

And they have complementary wins on the two rows that matter. Under the graded harness,
by CANDIDATE TIME -- never by the reported speedup, whose baseline arm spreads 33-39% on
sub-millisecond configs (finding 42's addendum):

    cfg      v26      v34      v35
      2   0.1440   0.0481   0.0481
      3   0.0743   0.0886   0.0553     <- v35 wins config 3 outright, where v34 is
      4   0.1106   0.0922   0.0922        WORSE than the v26 it descends from
     12   0.1792   0.0922   0.0922

while v36 measured **+0.082 of weighted_score (2.489 -> 2.571) with no regression on any
config**, concentrated on 1, 4, 5, 9, 10, 12 -- rows v35 does not touch at all.

THE MERGE'S OWN CONTENT: v35's RESET, GENERALIZED FROM FIVE NAMES TO A DERIVED SET
----------------------------------------------------------------------------------
v33's `_invalidate_shape_state` enumerates the state latched to an input shape: the CUDA
graph, its three static buffers, v23's attention tile, v17's FFN gate. v35's entire
content is that **v34 adds five more attributes v33 cannot know about** --
`launch_reason`, `launch_fused_used`, `launch_bm`, `launch_warps`, `mask_capture` -- so
v35 names them in `SHAPE_LATCHED_BY_V34` and resets them.

**v36 adds nine more.** Five flags (`gemm_used`, `gemm_reason`, `gemm_sites`,
`gemm_engaged`, `gemm_stats`) and four tile tuples (`_tile_qkv`, `_tile_out`,
`_tile_ffn_in`, `_tile_ffn_out`), every one of them computed in `_decide_gemm` from
`m = b * s`. That is exactly the "generation 36 that latches a sixth attribute" v35's own
test was written to catch, arriving one generation late and nine attributes wide.

So this candidate does not name them either. `shape_latched_over` walks the MRO and
returns **every class-body attribute introduced above `v26`, with its class-body
default**, and the reset restores all of them -- so a generation 38 that latches a
fifteenth is covered by construction rather than by remembering. The declared list is
still asserted, from the other direction: `tests/bench/test_v37_recombined2.py` computes
the set from the classes and requires it to be a strict superset of v35's five, so a
mechanism that is dropped from the reset fails a test instead of shipping ([L14]).

`_proj_t` is deleted rather than reset: it is an INSTANCE attribute built by
`_decide_gemm` from `self._cache`, so it has no class-body default to restore, and after
`_prime` re-runs it would otherwise hold transposes of the previous prime's tensors.

WHAT THE COMBINATION MAKES REACHABLE THAT NEITHER PARENT DOES
--------------------------------------------------------------
v35 recorded one wrong answer that exists ONLY in the combination it made: v34's
`_try_capture` elides the mask buffer when `_nomask`, and that is unreachable in v34
alone because v34 raises on the second shape -- v33 removes the raise. Measured there:
69407 / 262144 elements past the locked tolerance. v35 closes it by re-deriving the mask
state (`_prime` is idempotent), and that fix is inherited here unchanged.

The v36 half of this merge does **not** add a second wrong answer, and saying so
precisely matters more than claiming one. A stale `_tile_*` is a tile sized for a batch
that is no longer there: `proj_gemm` masks its M edge and `legal()`'s only M-dependent
rule is a padding-waste rule, so the arithmetic stays right and the kernel merely runs on
the wrong shape's plan. A stale `_proj_t` holds transposes of the same weights. What the
reset buys on this half is therefore **a correct plan rather than a correct answer** --
and, on the streamed path, a plan at all:

THE STREAMED PATH HAS TO SETTLE THREE DECISIONS NOW, IN ORDER
---------------------------------------------------------------
v33's `forward` returns before ever reaching v36's, so `_settle_slice_decisions` is where
every shape-latched decision has to be made -- v33 settles the attention tile and the FFN
gate, v35 added the launch decision, and this adds the GEMM plan. The order is not
cosmetic: `_decide_gemm` READS `launch_fused_used` and `fused_ffn_used` to know whether a
megakernel already owns the FFN, so planning it before them would plan `ffn_in` and
`ffn_out` sites that are never launched -- the exact "reason string that overstates what
engaged" v36's own test pins. Settled after both, and against the SLICE, not the batch
that never runs.

NO NEW MECHANISM, AND THEREFORE NO NEW SPEED ARGUMENT ([L33])
--------------------------------------------------------------
Every kernel here is v35's or v36's. The honest prediction is that the two win sets are
disjoint and simply add: v36's +0.082 lands on 1, 4, 5, 9, 10, 12 and v35's config-3
advantage over v34 lands on 3, which v36 does not touch (v36 takes only `qkv` there, and
config 3 is 512 tokens where the sweep reads a tie). Anything larger than the sum should
be disbelieved before it is celebrated.
"""

from __future__ import annotations

import copy

from .v26_causal_correct import build as build_v26
from .v33_streamed_long import build_on as build_streaming_on
from .v36_gemm_gelu import build as build_v36

# v35's own reset list (`CandidateV35.SHAPE_LATCHED_BY_V34`) is a class attribute created
# inside its `build`, so it cannot be imported here -- the test builds v35 and reads it
# off the class, which is the direction that matters: the assertion is that this
# candidate's DERIVED set covers v35's DECLARED one.


def _class_body_state(cls) -> dict:
    """Non-callable, non-dunder attributes a class introduces in its OWN body.

    Underscore-prefixed names are deliberately INCLUDED. v35's test helper excludes them
    and that was sound for v34, which latches nothing private; v36 latches four private
    tile tuples, and a reset that skipped them would leave the model launching a plan
    sized for a batch that is no longer there.
    """
    out = {}
    for name, value in vars(cls).items():
        if name.startswith("__"):
            continue
        if callable(value) or isinstance(value, (classmethod, staticmethod, property)):
            continue
        out[name] = value
    return out


def shape_latched_over(top_cls, base_cls) -> dict:
    """`{name: class-body default}` for every attribute introduced ABOVE `base_cls`.

    Computed from the classes, never typed out. `base_cls` is a separately built v26, so
    the comparison is by NAME rather than by identity -- `build_v26` returns a fresh class
    object each call and the one inside `top_cls.__mro__` is a different object with the
    same body.
    """
    known = set()
    for klass in base_cls.__mro__:
        known |= set(vars(klass))
    latched: dict = {}
    for klass in top_cls.__mro__:                 # most derived first
        for name, value in _class_body_state(klass).items():
            if name not in known:
                latched.setdefault(name, value)
    return latched


def build(baseline_cls):
    # v36 on top of v34 on top of v26, then v33's streaming layer on top of THAT --
    # the same layering v35 uses, one rung higher. One copy of every mechanism; nothing
    # was retyped, so nothing can drift from its sibling ([L14]).
    v36_cls = build_v36(baseline_cls)
    streamed_cls = build_streaming_on(v36_cls)

    # Resolved ONCE, at build time, from the class hierarchy that will actually run.
    latched = shape_latched_over(v36_cls, build_v26(baseline_cls))

    class CandidateV37(streamed_cls):
        # `{name: class-body default}` for everything v34 and v36 latch to an input
        # shape. Derived, not declared -- see the module docstring. Exposed as a class
        # attribute so the test can read it off the built class.
        SHAPE_LATCHED: dict = latched

        def _invalidate_shape_state(self, mask=None):
            """v33's reset, plus everything v34 and v36 latched that v33 cannot name.

            Order matters and is v35's: the superclass drops the graph and its static
            buffers FIRST, so re-priming below cannot repopulate caches that a live graph
            still points at.
            """
            super()._invalidate_shape_state(mask)

            for name, default in self.SHAPE_LATCHED.items():
                # `copy` so a mutable class-body default (v36's `gemm_stats` is typed
                # `dict | None`) can never be aliased into an instance and mutated there.
                setattr(self, name, copy.copy(default))

            # An INSTANCE attribute with no class-body default: v36 builds `_proj_t` in
            # `_decide_gemm` out of `self._cache`, and `_prime` below rebuilds `_cache`.
            # Dropping it means a stale plan cannot be read; `_decide_gemm` rebuilds it
            # whenever it takes a site.
            if hasattr(self, "_proj_t"):
                del self._proj_t

            # Re-derive the MASK-dependent state. `_nomask` gates v8's fast path, v17's
            # fused FFN, v36's `_decide_gemm` shortening, and -- the load-bearing one --
            # whether the captured graph has a mask buffer at all. A mask's validity is
            # not a function of its shape, so a shape change is simply the only moment we
            # get to notice the mask changed with it. `_prime` is idempotent.
            self._prime(mask)

        def _settle_slice_decisions(self, head):
            """v33's hook. The streamed path never reaches v36's `forward`, so all three
            of the decisions that `forward` would have made have to be made here --
            against the slice that runs, and in the order they read each other.

            `_decide_gemm` reads `launch_fused_used` and `fused_ffn_used`: where a
            megakernel already owns the FFN there is no `F.linear` left to replace, and
            planning one anyway produces a tile that is never launched and a
            `gemm_reason` that overstates what engaged.
            """
            super()._settle_slice_decisions(head)          # v23's tile, v17's FFN gate
            if self.launch_reason == "undecided":
                self._decide_launch(head)
            if self.gemm_reason == "undecided":
                self._decide_gemm(head)

    return CandidateV37
