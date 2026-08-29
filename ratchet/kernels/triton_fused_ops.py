"""Custom Triton kernels for fused LayerNorm + residual and fused activation epilogues."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_layernorm_residual_fwd_kernel(
        X_ptr,
        R_ptr,
        Y_ptr,
        OutR_ptr,
        W_ptr,
        B_ptr,
        stride_x_row,
        stride_r_row,
        stride_y_row,
        stride_outr_row,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused LayerNorm(X + R) forward kernel.
        
        Loads X and R, writes (X + R) to OutR (if not None), normalizes,
        applies affine weight W and bias B, and stores to Y.
        """
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x_row = X_ptr + row_idx * stride_x_row
        r_row = R_ptr + row_idx * stride_r_row if R_ptr is not None else None
        y_row = Y_ptr + row_idx * stride_y_row
        outr_row = OutR_ptr + row_idx * stride_outr_row if OutR_ptr is not None else None

        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        if r_row is not None:
            r = tl.load(r_row + cols, mask=mask, other=0.0).to(tl.float32)
            accum = x + r
        else:
            accum = x

        if outr_row is not None:
            tl.store(outr_row + cols, accum, mask=mask)

        # Online mean and variance reduction
        mean = tl.sum(accum, axis=0) / N
        diff = tl.where(mask, accum - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        # Affine transform
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32) if W_ptr is not None else 1.0
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32) if B_ptr is not None else 0.0

        y = diff * rstd * w + b
        tl.store(y_row + cols, y, mask=mask)


    @triton.jit
    def _fused_gelu_kernel(
        X_ptr,
        Y_ptr,
        N_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Exact GELU forward: 0.5 * x * (1.0 + erf(x / sqrt(2)))."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_elements

        x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        inv_sqrt2 = 0.7071067811865475
        y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
        tl.store(Y_ptr + offsets, y, mask=mask)


def triton_fused_layernorm_residual(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute fused LayerNorm + residual addition with Triton on GPU, fallback on CPU."""
    if not _TRITON_AVAILABLE or not x.is_cuda:
        accum = x if residual is None else (x + residual)
        y = F.layer_norm(accum, (accum.shape[-1],), weight, bias, eps)
        return y, accum

    shape = x.shape
    d_model = shape[-1]
    rows = x.numel() // d_model

    x_2d = x.view(rows, d_model)
    r_2d = residual.view(rows, d_model) if residual is not None else None

    y_2d = torch.empty_like(x_2d)
    out_res_2d = torch.empty_like(x_2d) if residual is not None else x_2d

    block_size = triton.next_power_of_2(d_model)

    _fused_layernorm_residual_fwd_kernel[(rows,)](
        x_2d,
        r_2d,
        y_2d,
        out_res_2d if residual is not None else None,
        weight,
        bias,
        x_2d.stride(0),
        r_2d.stride(0) if r_2d is not None else 0,
        y_2d.stride(0),
        out_res_2d.stride(0) if residual is not None else 0,
        d_model,
        eps,
        BLOCK_SIZE=block_size,
    )
    return y_2d.view(shape), out_res_2d.view(shape)


def triton_fused_gelu(x: torch.Tensor) -> torch.Tensor:
    """Execute fused exact GELU with Triton on GPU, fallback on CPU."""
    if not _TRITON_AVAILABLE or not x.is_cuda:
        return F.gelu(x, approximate="none")

    y = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 1024
    grid = ((n_elements + block_size - 1) // block_size,)

    _fused_gelu_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return y
