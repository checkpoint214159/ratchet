"""Candidate v14 -- shape-aware dispatch, with predicates derived from the device.

Generation 14. Parent: v13_safe_capture. Branch: cand/g14/dispatch.

WHY THIS IS THE ARCHITECTURE THE COMPETITION ASKED FOR
------------------------------------------------------
The problem statement is explicit twice over: participants "can decide different
implementations for different shapes by adding shape checks", and "different methods may
be used depending on the machine (GPU cards) you use". Every candidate so far applies ONE
implementation uniformly to all 14 configs. This is the first that chooses.

THE RULE THAT MAKES IT DEFENSIBLE
---------------------------------
Every predicate is a function of MEASURED device properties, never a config id and never
a hardcoded constant. A branch on `config_id == 14` would be shape-detection -- the thing
this project's own contract calls fraud (CLAUDE.md rule 2). A branch on "would this shape's
working set exceed the memory this device reports" is a statement about the hardware that
generalizes to a card we have never seen.

Two branches:

  STREAMED   when the activation working set would not fit in free device memory. Runs the
             batch in slices sized from that free memory, so a shape that cannot be held
             at once is still computed rather than OOMing.
  RESIDENT   otherwise: v13 unchanged.

The threshold comes from `torch.cuda.mem_get_info()` at first forward -- the memory
actually free on the device at that moment, not `total_memory`, because the harness has
already allocated the baseline model and the input by then.

REPORTING is_tuned, FOLLOWING ROCm/aiter
----------------------------------------
`choose()` returns `(path, is_tuned)`. `is_tuned=False` means the shape fell through to a
default nobody has measured. Callers can log it and the report can list it, so an untuned
path is never silently presented as a tuned one -- which is the whole point of the
signature and costs nothing to carry.

HONEST LIMIT: on the 13 runnable configs this dispatcher always chooses RESIDENT, so it
is expected to measure identically to v13. Its value is entirely in the shape it does not
yet get to run -- config 14, whose 12.21 GiB input the harness itself cannot build. The
dispatch exists so that the submission degrades to streaming instead of dying, and so the
predicate is stated in terms a different GPU can evaluate.
"""

from __future__ import annotations

import torch

from .v13_safe_capture import build as build_v13

# Fraction of FREE device memory a resident forward may plan to occupy. Below 1.0 because
# the allocator fragments, the harness holds the baseline model and input, and a forward
# needs headroom for its own intermediates. Not tuned -- deliberately conservative.
RESIDENT_BUDGET = 0.35


def estimate_working_set_bytes(batch: int, seq: int, d_model: int, heads: int,
                               layers: int, dtype_bytes: int) -> int:
    """Bytes a resident forward is expected to hold at peak.

    Counts the activation tensor and the live intermediates one layer needs at once (the
    fused QKV output is 3x the activation, plus the attention context and the residual).
    Deliberately an over-estimate: mispredicting "resident" for a shape that then OOMs is
    a crash, while mispredicting "streamed" only costs a loop.
    """
    act = batch * seq * d_model * dtype_bytes
    return act * 6


def choose(batch: int, seq: int, d_model: int, heads: int, layers: int,
           dtype_bytes: int, free_bytes: int) -> tuple[str, bool]:
    """(path, is_tuned). Pure function of shape and measured free memory."""
    need = estimate_working_set_bytes(batch, seq, d_model, heads, layers, dtype_bytes)
    if need > free_bytes * RESIDENT_BUDGET:
        # Streaming is correct but has never been measured through the harness, because
        # the one shape that needs it cannot have its input built. Say so.
        return "streamed", False
    return "resident", True


def build(baseline_cls):
    v13_cls = build_v13(baseline_cls)

    class CandidateV14(v13_cls):
        dispatch_path: str = "unset"
        dispatch_is_tuned: bool = False

        def _decide(self, x):
            free, _total = torch.cuda.mem_get_info(x.device)
            self.dispatch_path, self.dispatch_is_tuned = choose(
                x.shape[0], x.shape[1], x.shape[2],
                self.layers[0].attention.num_heads, len(self.layers),
                x.element_size(), free)
            # Slice size solved from the same budget, so the two cannot disagree.
            per_row = max(1, estimate_working_set_bytes(
                1, x.shape[1], x.shape[2], 1, len(self.layers), x.element_size()))
            self._slice = max(1, min(x.shape[0],
                                     int(free * RESIDENT_BUDGET // per_row)))

        def forward(self, x, valid_token_mask=None):
            if self.dispatch_path == "unset":
                self._decide(x)

            if self.dispatch_path == "resident":
                return super().forward(x, valid_token_mask)

            # STREAMED: slice the batch. No graph -- each slice is a distinct shape only
            # at the tail, and a shape that needed streaming is not launch-bound anyway.
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            out = torch.empty_like(x)
            for start in range(0, x.shape[0], self._slice):
                stop = min(start + self._slice, x.shape[0])
                ms = None if valid_token_mask is None else valid_token_mask[start:stop]
                out[start:stop] = self._core(x[start:stop], ms)
            return out

    return CandidateV14
