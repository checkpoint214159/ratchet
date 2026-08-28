"""Known-bad kernel: attention with scores scaled by 1/d instead of 1/sqrt(d).

The classic transcription bug -- `scale = 1.0 / head_dim` where the paper says
1/sqrt(d_k) -- and a favorite of machine-generated kernels because the output is
still a plausible attention output: correct shape, correct range, rows still sum
to one. Only the softmax temperature is wrong, so every attention weight is
subtly off. Nothing but an elementwise comparison against a real reference
catches it, which is exactly what this fixture is here to prove the gate does.

Everything except the scale mirrors reference_fp32's algebra bit for bit (same
fp32 compute, same GQA expansion, same causal mask), so the ONLY defect present
is the scale. Fixtures carry one bug each; a fixture with two bugs tells you
nothing about which one the gate caught.

Expected gate: "tolerance" on every shape with more than one KV token. On the
N=1 decode shape this bug is mathematically invisible: softmax over a single
score is 1.0 regardless of any monotone rescaling, so the kernel is exactly
correct there. That is not a weakness of the gate -- it is a documented fact
about single-token decode, and the reason N=1 alone could never anchor a
correctness suite.
"""

from __future__ import annotations

import torch

from ..reference import _expand_kv


def kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           causal: bool = False) -> torch.Tensor:
    q32, k32, v32 = (t.to(torch.float32) for t in (q, k, v))
    k32, v32 = _expand_kv(k32, v32, q32.shape[1])

    scale = 1.0 / q32.shape[-1]   # THE BUG: should be 1.0 / sqrt(d)
    s = (q32 @ k32.transpose(-2, -1)) * scale

    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        mask = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril(n_k - n_q)
        s = s.masked_fill(~mask, float("-inf"))

    p = torch.softmax(s, dim=-1)
    return (p @ v32).to(q.dtype)
