"""Known-bad kernel: returns its input unchanged.

The canonical fraud. A ReLU "kernel" that returned its input unchanged passed a
published benchmark's correctness check on all-positive inputs and reported a 374x
speedup. Attention is harder to fool this way -- the output of softmax-weighted
averaging looks nothing like q even on friendly inputs -- but the fixture stays,
because it is the cheapest possible probe of the gate: if THIS passes anywhere, the
gate is not comparing against a reference at all.

Expected gate: "tolerance", on the first distribution tried, on every shape.
"""

from __future__ import annotations

import torch


def kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           causal: bool = False) -> torch.Tensor:
    # Deliberately return the input tensor itself, not a copy. A gate that mutates
    # its inputs in place would be exposed by this too, which is a bonus.
    return q
