"""Known-bad kernel: correct attention, then nan_to_num on the output.

The single most common defect in machine-generated kernels. An audit of 2,638
kernels already accepted by their own system found a majority carried at least
one contract violation under adversarial gates, the most common being silently
replacing NaN/Inf with ordinary numbers -- usually a reflexive
`torch.nan_to_num` slapped on the output to make a flaky test go green.

It is also the defect `torch.allclose` cannot see: allclose skips non-finite
entries, so a kernel that launders NaN into 0.0 compares equal to a reference
that correctly says NaN. That is why check_nonfinite exists as its own gate.

The attention itself IS the reference -- this fixture calls reference_fp32
directly -- so the laundering is the only defect present. On the four standard
distributions (and denormal, and noncontiguous) the output contains no
non-finites, nan_to_num is the identity, and the kernel is bitwise correct;
the gate must sail it through those and catch it the moment the reference
produces a non-finite.

Expected gate: "nonfinite". Which distribution trips it depends on the shape:
with_nan / with_inf plant non-finites explicitly, and near_overflow gets there
first whenever the fp32 score matmul overflows on its own.
"""

from __future__ import annotations

import torch

from ..reference import reference_fp32


def kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           causal: bool = False) -> torch.Tensor:
    out = reference_fp32(q, k, v, causal=causal)
    return torch.nan_to_num(out)   # THE BUG: NaN -> 0.0, +/-Inf -> finite max
