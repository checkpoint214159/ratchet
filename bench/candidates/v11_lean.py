"""Candidate v11 -- the frontier with dead weight removed.

Generation 11. Parent: v9a_compiled_core. Branch: cand/g11/lean.

WHAT WAS REMOVED AND WHY
------------------------
The g10 ablation (finding 15) measured each inherited trick by deleting it:

    L2 batch chunking (v3)    +5.8% worst, -0.3% on config 6   -> SUBSUMED
    fused Q|K|V (v1)          +20% on the launch-bound configs -> keep
    fp16 weight cache (v1/v6) +395% on config 13               -> keep

Chunking is gone. It took config 6 from 3.21x to 5.72x when v3 introduced it -- real then
-- but under `torch.compile` the win has vanished, including on the very config it was
designed for, where removing it is marginally faster. Inductor manages the working set;
our Python chunk loop was contributing loop overhead, a `torch.empty_like` allocation, a
device-calibration constant and a tuning parameter, for nothing.

`reduce-overhead` rather than `max-autotune`, per findings L11 and L15: the two were
indistinguishable here because Inductor **disables GEMM autotuning on this 66-SM card**
("Not enough SMs to use max_autotune_gemm mode"). Same result for a fraction of the
compile cost -- but that equivalence is a property of this hardware and must be re-tested
on anything larger.

WHAT REMAINS, AND WHY EACH EARNS ITS PLACE
------------------------------------------
    fused Q|K|V         one GEMM launch instead of three; worth 20% where launch count
                        dominates and ~2% where it does not
    fp16 + fp32 accum   the single largest lever, worth up to 5x; hand-rolled mixed
                        precision beats leaving the choice to Inductor
    flash via SDPA      exact under causality, and the only reason config 14 runs at all
    right-padding proof the key mask is redundant for a valid prefix, so the fast path
                        survives padding (finding 11)
    torch.compile       fuses the elementwise chains we could not fuse by hand without
                        breaking precision (finding 10)

Five components, each with a measurement behind it and none inherited on faith.
"""

from __future__ import annotations

import torch

from .v8_padfast import build as build_v8


def build(baseline_cls):
    v8_cls = build_v8(baseline_cls)

    class CandidateV11(v8_cls):
        use_graph = False          # Inductor owns graph capture

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if not hasattr(self, "_compiled_core"):
                self._compiled_core = torch.compile(
                    self._core, mode="reduce-overhead", dynamic=False)
            # No chunk loop: straight to the compiled core.
            return self._compiled_core(x, valid_token_mask)

    return CandidateV11
