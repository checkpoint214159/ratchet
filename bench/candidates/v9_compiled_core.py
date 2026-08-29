"""Candidate v9a -- let Inductor fuse OUR decomposition.

Generation 9, sibling A. Parent: v8_padfast. Branch: cand/g9/compiled-core.

THE REASONING (from learning L8)
--------------------------------
Against the compiled baseline, v8 wins where an ALGORITHMIC choice matters and loses
where the win would be pure kernel fusion:

    config 13   7.89x   streaming attention beats what Inductor generates
    config 6    3.00x   L2-sized chunking Inductor does not do
    config 11   3.69x   head_dim=8, many heads
    config 9    0.94x   LOSS -- nothing algorithmic to exploit, Inductor just fuses better
    config 12   0.90x   LOSS -- same

That is not a contradiction, it is a division of labour. Inductor is better than we are at
fusing elementwise chains into single kernels; we are better than Inductor at choosing
flash attention, chunking to L2, and eliding a provably redundant mask.

So: keep our algorithm and hand the resulting op sequence to Inductor. `_core` still calls
SDPA with the padding proof, still chunks, still fuses Q|K|V -- and `torch.compile` then
fuses the LayerNorm/GELU/residual chains around it, which is exactly what v7 tried by hand
and failed on precision. Inductor keeps fp32 where it needs to.

WHY THE MANUAL CUDA GRAPH IS DISABLED HERE
------------------------------------------
`mode="max-autotune"` already applies CUDA graphs itself. Capturing a compiled callable
inside our own static-buffer graph stacks two graph mechanisms and is a known source of
silent staleness. Inductor's is the better-tested one, so we hand the job over rather than
nest them. Compilation is deferred to first forward for the same reason the fp16 cache is:
the harness builds on CPU in fp32 before moving to device.
"""

from __future__ import annotations

import torch

from .v8_padfast import build as build_v8


def build(baseline_cls):
    v8_cls = build_v8(baseline_cls)

    class CandidateV9A(v8_cls):
        use_graph = False          # Inductor owns graph capture; see the docstring

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if not hasattr(self, "_compiled"):
                # Compile the CORE, not the module: the module's forward carries the
                # priming branch, which must not be traced.
                self._compiled = torch.compile(
                    self._core, mode="max-autotune", dynamic=False)
            return self._compiled(x, valid_token_mask)

    return CandidateV9A
