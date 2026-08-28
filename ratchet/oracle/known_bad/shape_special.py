"""Known-bad kernel: correct at exactly one benchmarked shape, broken elsewhere.

This is the overfit-to-the-benchmark failure mode: a kernel tuned (or trained,
or memorized) against the timed shape matrix that carries a tail bug everywhere
else. It is exactly correct when q.shape == (32, 16, 512, 64) -- the
B=32, N=512, H=16, D=64 BENCHMARK shape, which is deliberately NOT in the
correctness set -- and on every other shape it drops the last KV token before
attending, the signature of an off-by-one in tail masking or block-boundary
handling. When dropping the token leaves zero KV tokens (the N=1 decode case),
it returns zeros, which is what a kernel that never enters its inner loop does.

This fixture is why CORRECTNESS_SHAPES and BENCHMARK_SHAPES are disjoint by
construction: validate on the shapes you time and this kernel is certified
correct. Off-by-one correctness sizes (127/128/129) exist precisely to make the
dropped-token error visible.

Both branches route through reference_fp32 so the special shape is exact by
construction and the buggy branch differs from the reference in one way only:
the truncated KV. Note the causal case is extra wrong, deliberately so -- with
n_k = n_q - 1 the aligned causal mask shifts by one for every query row, which
is what real block-boundary bugs do to masks.

Expected gate: "tolerance" on every shape in the correctness set.
"""

from __future__ import annotations

import torch

from ..reference import reference_fp32

# (B, H, N, D) as q carries it. Shape(B=32, N=512, H=16, D=64) from BENCHMARK_SHAPES.
_SPECIAL_Q_SHAPE = (32, 16, 512, 64)


def kernel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           causal: bool = False) -> torch.Tensor:
    if tuple(q.shape) == _SPECIAL_Q_SHAPE:
        return reference_fp32(q, k, v, causal=causal)

    if k.shape[-2] <= 1:
        # Dropping the only KV token: the inner loop never runs, the accumulator
        # is returned as initialized. Output shape and dtype match q, so only the
        # tolerance gate can catch this -- which is the point.
        return torch.zeros_like(q)

    return reference_fp32(q, k[..., :-1, :], v[..., :-1, :], causal=causal)
