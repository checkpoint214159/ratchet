"""Candidate v33 -- restore batch streaming to the frontier, so a shape too large to hold
resident is computed instead of refused.

Generation 33. Parent: v26_causal_correct. Branch: cand/g33/config14.

THE GAP THIS CLOSES
-------------------
v14 built exactly this: a dispatch that streams the batch when the working set would not
fit in the memory the device reports free. It was never inherited. The frontier's lineage
is v26 <- v23 <- v18 <- v17 <- v13, and v17 branched from v13, not from v14 -- so the
streaming path fell out of the line at generation 17 and nothing since has had it.

Nobody noticed because it makes no difference on thirteen of the fourteen announced
configs, where `choose()` always returns "resident". It makes the difference on the
fourteenth. And the loss is not confined to this card: at config 14 the resident path
plans to hold roughly 6 x 12.21 GiB = 73 GiB of activations and v13's capture then
CLONES the 12.21 GiB input for its static buffer, so the frontier would fail this shape
on an 80 GiB accelerator too -- on a shape whose actual arithmetic fits in about 4 GiB.

A SECOND DEFECT, FOUND BY BUILDING THE FIRST
-------------------------------------------
Everything from v13 up latches to the shape of the FIRST input it ever sees -- the CUDA
graph and its static buffers, v17's FFN gate, v23's attention tile. Called again at a
different batch size the frontier raises from `_static_x.copy_(x)` when the new input is
larger, and BROADCASTS when it is smaller: the first sequence computed B times and
returned as B identical rows. No test saw it because every sweep builds a fresh model per
config, so no model in this project had ever been called at two shapes ([L24]).
`_invalidate_shape_state` drops all of it when the shape changes.

WHAT IT DOES
------------
One predicate, taken unchanged from v14 so the two cannot drift apart: compare the
estimated working set against `torch.cuda.mem_get_info()` at the first forward. Shapes
and one measured device property; no config id (CLAUDE.md rule 2), and evaluable on a
card nobody here has seen.

  RESIDENT  -> v26 unchanged, graph capture and all. This is what all 13 other configs get.
  STREAMED  -> slice the batch, run `_core` per slice, no compile and no graph capture.

Skipping capture on the streamed path is not an oversight. v13's `_try_capture` allocates
`x.clone()` as its static input; on a shape that is streamed *because* the input barely
fits, that clone is the largest single allocation in the process and buys nothing, since
a streamed forward runs its slices in a Python loop that no single graph spans.

WHAT IT DOES NOT DO, AND CANNOT
-------------------------------
It does not make config 14 runnable on this card. `forward(x) -> y` needs a 12.21 GiB
input and a 12.21 GiB output resident simultaneously -- 24.42 GiB of tensors against
15.99 GiB of VRAM, a floor no implementation removes (`bench/feasibility.py`,
impossibility 2). Streaming shrinks the working set between those two tensors from ~73 GiB
to about 4 GiB; it cannot shrink the two tensors themselves.

So this is a portability fix and a measurement enabler, not a rescue. What it buys here
is that the compute can be run and checked one sequence at a time at the true S=100000
(`bench/run_matrix.py`'s capability path), instead of the shape being unreachable and the
row reading `oom` with a truncated traceback.

CORRECTNESS
-----------
Slicing the batch is exact IN EXACT ARITHMETIC: attention is within-sequence, the token
mask is per-sequence, and every other operation is position-wise, so batch elements never
interact. It is NOT bitwise exact in floating point, and the reason is not ours -- the
batch axis is a GEMM's M dimension, so cuBLAS picks a different tiling and reduces in a
different order. Measured, the REFERENCE ITSELF moves 3.46e-4 when its own batch is
sliced the same way, against the candidate's 6.63e-4.

So `tests/bench/test_v33_streaming.py` asserts the thing that matters instead of a
threshold on that gap: streaming must not move the candidate FURTHER FROM THE REFERENCE,
which is the only distance a grader measures (measured 8.57e-4 whole, 8.25e-4 sliced --
slicing helped, fractionally). Every equivalence assertion is preceded by one that the
mechanism engaged (L36) and paired with a shape where it must NOT engage, because a
streaming test that quietly ran the resident path asserts only that v26 equals v26.
"""

from __future__ import annotations

import torch

from .v26_causal_correct import build as build_v26
from .v14_dispatch import RESIDENT_BUDGET, choose, estimate_working_set_bytes


def build_on(base_cls):
    """The streaming layer, applied to whatever class is handed in.

    Factored out at generation 35 so a recombination can stack it on top of a SIBLING of
    v26 rather than on v26 itself, WITHOUT a second copy of this dispatch existing. That
    is the same argument this file already makes for importing v14's predicate instead of
    restating it: two copies drift, one copy cannot ([L14]). `build` below is unchanged --
    it is `build_on(build_v26(...))` and nothing about v33's own behaviour moves.
    """

    class CandidateV33(base_cls):
        stream_path: str = "unset"
        stream_is_tuned: bool = False
        stream_slice: int = 0
        stream_reason: str = "undecided"
        _decided_for: tuple | None = None

        def _decide_stream(self, x):
            """Decided once, from the memory the device reports free right now -- not
            from `total_memory`, because by this point the harness is already holding the
            baseline's weights and a 12.21 GiB input."""
            free, total = torch.cuda.mem_get_info(x.device)
            b, s, d = x.shape
            heads = self.layers[0].attention.num_heads
            self.stream_path, self.stream_is_tuned = choose(
                b, s, d, heads, len(self.layers), x.element_size(), free)
            per_row = max(1, estimate_working_set_bytes(
                1, s, d, heads, len(self.layers), x.element_size()))
            self.stream_slice = max(1, min(b, int(free * RESIDENT_BUDGET // per_row)))
            self._decided_for = (b, s, d)
            need = estimate_working_set_bytes(b, s, d, heads, len(self.layers),
                                              x.element_size())
            self.stream_reason = (
                f"{self.stream_path}: working set {need / 2**30:.2f} GiB vs "
                f"{RESIDENT_BUDGET:.2f} x {free / 2**30:.2f} GiB free; "
                f"slice={self.stream_slice}")

        def _invalidate_shape_state(self, mask=None):
            """Drop everything that was decided against the previous input shape.

            `mask` is accepted and ignored here; it exists so a subclass that latches
            MASK-derived state as well can re-derive it (generation 35).

            v13 captures a CUDA graph on the first forward and keeps `_static_x` sized to
            that input for the life of the model; v17 and v23 latch their kernel choices
            the same way. Called again at a different batch size, `_static_x.copy_(x)`
            either raises (a bigger x) or BROADCASTS a smaller one -- computing the first
            sequence B times and returning B copies of it. That is the silent-wrong-answer
            shape of [L25], reached through the door [L24] describes: nothing caught it
            because every sweep builds a fresh model per config, so no model in this
            project's history had ever been called at two shapes.
            """
            self._graph = None
            self._capture_attempted = False
            self.graph_verified = False
            for attr in ("_static_x", "_static_m", "_static_y"):
                if hasattr(self, attr):
                    delattr(self, attr)
            # The tile and the FFN gate are functions of (batch, seq); re-open them.
            self.attn_reason = "undecided"
            self.fused_ffn_reason = "undecided"

        def _settle_slice_decisions(self, head):
            """Settle every shape-latched kernel decision against the SLICE that will
            actually run, not against the whole batch that never will.

            An extension point, not a convenience: a subclass that adds a fourth
            shape-latched decision has to settle it here too, or the streamed path runs
            with that decision still reading "undecided" and silently takes a default.
            """
            if self.attn_reason == "undecided":
                self._decide_attn(head)
            if self.fused_ffn_reason == "undecided":
                self._decide_ffn(head)

        def forward(self, x, valid_token_mask=None):
            # Non-causal input goes to v26's own guard, unchanged: it delegates to the
            # unmodified baseline, and that decision is not this candidate's to revisit.
            if not getattr(self.config, "causal", True):
                return super().forward(x, valid_token_mask)

            # RE-DECIDE when the input shape changes. v14 latched this on the first
            # forward, and v13's graph capture latches to the first shape as well, so a
            # model warmed at one batch size and then called at another raised
            #   "output with shape [1, S, D] doesn't match the broadcast shape [B, S, D]"
            # inside `_static_x.copy_(x)`. Found by the config-14 capability path calling
            # a per-sequence-warmed model once with the whole batch. Raising is better
            # than answering, but a decision cached against the wrong shape is [L24] --
            # correct only under the call pattern we happen to use.
            if self.stream_path == "unset" or self._decided_for != tuple(x.shape):
                if self._decided_for is not None:
                    self._invalidate_shape_state(valid_token_mask)
                self._decide_stream(x)
            if self.stream_path == "resident":
                return self._resident_forward(x, valid_token_mask)
            return self._streamed_forward(x, valid_token_mask)

        def _resident_forward(self, x, valid_token_mask):
            """The whole stack below this layer, unchanged.

            v23's and v17's forwards re-run their own decisions when the reason is
            "undecided", so delegating is enough -- but only because
            `_invalidate_shape_state` re-opened them.

            A METHOD rather than an inline `super().forward(...)` so that a descendant
            can wrap the attempt without restating the dispatch or the streaming loop
            around it (generation 38 catches `OutOfMemoryError` here). Extraction only:
            `forward` above behaves exactly as it did.
            """
            return super().forward(x, valid_token_mask)

        def _streamed_forward(self, x, valid_token_mask):
            """Prime the per-layer caches and settle every shape-latched kernel decision
            on a SLICE-shaped input, so the tile choice matches what actually runs."""
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            head = x[: self.stream_slice]
            self._settle_slice_decisions(head)

            out = torch.empty_like(x)
            for start in range(0, x.shape[0], self.stream_slice):
                stop = min(start + self.stream_slice, x.shape[0])
                ms = None if valid_token_mask is None else valid_token_mask[start:stop]
                out[start:stop] = self._core(x[start:stop], ms)
            return out

    return CandidateV33


def build(baseline_cls):
    return build_on(build_v26(baseline_cls))
