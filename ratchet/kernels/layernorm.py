"""LayerNorm Triton kernel (forward), one program per row.

Completes the "whole transformer layer in custom kernels" story for E5: with flash
attention, the TF32 linear, and this, every op in the block runs in a hand-written kernel
except the elementwise residual add. LayerNorm is memory-bound and a small fraction of the
layer cost, so this is about completeness and launch/traffic fusion, not raw FLOPs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd(X, W, B, Y, stride_m, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(Y + row * stride_m + cols, y.to(Y.dtype.element_ty), mask=mask)


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    *lead, N = x.shape
    M = 1
    for d in lead:
        M *= d
    x2 = x.reshape(M, N).contiguous()
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(N)
    _layernorm_fwd[(M,)](x2, weight, bias, y, x2.stride(0), N, eps,
                         BLOCK=BLOCK, num_warps=4)
    return y.reshape(*lead, N)
