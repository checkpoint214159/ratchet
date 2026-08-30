"""Candidate v35 -- the g33 shape fix and the g34 launch-bound fusion, in one model.

Generation 35. Parent: v33_streamed_long. Branch: cand/g35/recombined.
RECOMBINATION CONTRIBUTOR: v34_launch_bound (the same relationship v17 declared to v16).

WHY THESE TWO AND NOT SOME OTHER PAIR
-------------------------------------
They are the two live children of v26 and they are orthogonal in the strict sense: one
changes WHICH shapes are computable and how many times the model may be called, the other
changes HOW MANY KERNELS one call costs. Neither reads any state the other writes -- until
the merge, at which point exactly one thing binds them, and it is the reason this file
exists rather than a two-line import.

  v33  fixes a SILENT WRONG ANSWER. Everything from v13 up latches to the first input
       shape it ever sees; called again at a smaller batch, `_static_x.copy_(x)`
       BROADCASTS and the model returns the first sequence B times. Measured on the
       frontier: input (1,128,128) -> output (8,128,128). v33 returns (1,128,128).
       It also restores the batch streaming that fell out of the line at g17.

  v34  cuts 36 kernels per forward to 20, on a card where a graph node costs 0.798 us
       whatever it computes. Configs 9, 4, 12, 2, 8, 10 and 1 -- all the remaining
       headroom, six of them sub-millisecond -- are launch-bound.

THE ONE THING THAT BINDS THEM, AND IT IS THE WHOLE MERGE
--------------------------------------------------------
v33's `_invalidate_shape_state` enumerates the state that is latched to an input shape:
the CUDA graph, its three static buffers, v23's attention tile, v17's FFN gate. **v34 adds
five more latched attributes to that list and v33 cannot know about them** --
`launch_reason`, `launch_fused_used`, `launch_bm`, `launch_warps`, `mask_capture`. All
five are computed in `_decide_launch` from `(b*s, d_model, ffn_dim)` and from the SM count.

Compose the two naively and the bug v33 exists to kill comes straight back in a new
costume: a model warmed at batch 8 re-decides its graph and its tile at batch 1, and then
runs them through a megakernel whose `launch_bm` was sized for eight times as many tokens.
`one_wave` was true for 1024 tokens; nobody re-asked it for 128. That is [L14]'s warning
in its literal form -- when you merge, assert nothing was dropped -- and it is the reason
`test_v35_recombined.py` opens by asserting that the reset list COVERS the attribute set,
computed from the class rather than typed out, so a generation 36 that latches a sixth
attribute fails a test instead of shipping a wrong answer.

A SECOND BINDING, WHICH THE MERGE MAKES LOAD-BEARING
-----------------------------------------------------
`_nomask` is derived from the mask in `_prime`, which runs ONCE per model. v33 re-decides
shape state on a shape change but never re-primes, so `_nomask` survives it. In v33 alone
that is a latent hole (v8's `_fastpath` and `_needs_zeroing` go stale). In the merge it is
worse, because **v34's `_try_capture` elides the mask copy entirely when `_nomask` is
True** -- so a model warmed all-True and re-called at a new shape with a padded mask would
re-capture a graph that structurally cannot see the mask. A mask's validity is not a
function of its shape, so the only sound answer is to re-derive it: `_prime` is idempotent
(every implementation in the chain reassigns its lists rather than appending), so this
re-runs it. It costs one pass of weight casts on a shape change, which happens at most
once per distinct shape and never inside a timed loop.

THE THIRD PREDICATE, AND WHERE IT IS NOW SETTLED
-------------------------------------------------
Three shape predicates now coexist and they answer three different questions:

  choose()      (v14/v33)  does the working set fit the memory the device REPORTS FREE?
  amortizes()   (v17)      do enough tokens stream past the weights to pay for hoisting?
  one_wave()    (v34)      does the whole fused segment fit the device at once?

`amortizes` and `one_wave` are disjoint by construction and a test asserts it. `choose` is
on a different axis -- capacity, not occupancy -- so it does not contradict either; it
decides whether `_core` is called once or N times, and each call then asks the other two.

The merge has to place that ordering by hand. On the RESIDENT path v34's `forward` settles
the launch decision against the whole input, which is what runs. On the STREAMED path
v34's `forward` is never reached, so without this file `launch_reason` would still read
"undecided" at the moment `_core` runs -- not wrong (the flag defaults to False and the
parent's path is taken) but SILENT, which is the failure mode [L36] and finding 18 are
about. v33's `_settle_slice_decisions` hook is extended here so the launch decision is
made, and made against the SLICE that actually runs rather than the batch that never does.
On config 14's slice `amortizes` is true by four orders of magnitude, so the honest
outcome is a recorded decline, not a fusion.

WHAT THIS CANDIDATE DOES NOT CLAIM
-----------------------------------
No new mechanism and no new speed argument. The kernel-count win is v34's, bounded by v34
at 12.8 us per forward -- 0.71 us per node removed x 16 nodes -- which against each
config's own wall is 21.0% of config 2, 12.4% of config 12, 11.5% of config 4 and 0.45%
of config 8. Config 14 remains at 1.0 and is not a source of score (finding 33/40: the
reference materialises an 18.63 TiB tensor, so no speedup is claimable).

And the warning v34 left, restated because it applies unchanged here: removing 16 nodes
pushed config 2 from GPU-bound to CPU-bound, so its MIN is decisive (0.0440-0.0471 vs the
parent's 0.0604-0.0614, no overlap, 10/10) while its MEDIAN is not (0.0451-0.0666 vs
0.0604-0.0676, overlapping). `run_matrix` scores on a median. **The harness may not be
able to see this win on config 2, and that is a property of the statistic, not of the
candidate.** See the deliverable for the measurement that would resolve it.
"""

from __future__ import annotations

from .v33_streamed_long import build_on as build_streaming_on
from .v34_launch_bound import DERIVED_WARPS, build as build_v34


def build(baseline_cls):
    # v34 on top of v26, then v33's streaming layer on top of THAT. One copy of each
    # mechanism; neither was retyped, so neither can drift from its sibling ([L14]).
    v34_cls = build_v34(baseline_cls)
    streamed_cls = build_streaming_on(v34_cls)

    class CandidateV35(streamed_cls):
        # The attributes v34 latches to an input shape. Named here rather than inlined so
        # the test can read them off the class and check the reset covers every one.
        SHAPE_LATCHED_BY_V34: tuple[str, ...] = (
            "launch_reason", "launch_fused_used", "launch_bm", "launch_warps",
            "mask_capture",
        )

        def _invalidate_shape_state(self, mask=None):
            """v33's reset, plus the five attributes v33 could not know about.

            Order matters: the superclass drops the graph and its static buffers first, so
            re-priming below cannot repopulate caches that a live graph still points at.
            """
            super()._invalidate_shape_state(mask)

            self.launch_reason = "undecided"
            self.launch_fused_used = False
            self.launch_bm = 0
            self.launch_warps = DERIVED_WARPS
            self.mask_capture = "undecided"

            # Re-derive the MASK-dependent state as well. `_nomask` gates v8's fast path,
            # v17's fused FFN, and -- new in this merge -- whether the captured graph has
            # a mask buffer at all. It is not a function of the shape, so a shape change
            # is simply the only moment we get to notice the mask changed with it.
            self._prime(mask)

        def _settle_slice_decisions(self, head):
            """v33's hook. The streamed path never reaches v34's `forward`, so the launch
            decision has to be settled here -- against the slice that runs."""
            super()._settle_slice_decisions(head)
            if self.launch_reason == "undecided":
                self._decide_launch(head)

    return CandidateV35
