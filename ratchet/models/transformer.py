"""Optimized transformer layer implementation with shape-aware dispatch and fused operators."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedSelfAttention(nn.Module):
    """Shape-aware optimized self-attention with fast scaled dot-product paths and fused QKV projection."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self._packed_weight: Optional[torch.Tensor] = None
        self._packed_bias: Optional[torch.Tensor] = None
        self._pack_ptrs: Optional[Tuple[int, int, int]] = None

    def _get_packed_qkv(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        current_ptrs = (
            self.q_proj.weight.data_ptr(),
            self.k_proj.weight.data_ptr(),
            self.v_proj.weight.data_ptr(),
        )
        if (
            self._packed_weight is not None
            and self._packed_weight.device == x.device
            and self._packed_weight.dtype == x.dtype
            and self._pack_ptrs == current_ptrs
        ):
            return self._packed_weight, self._packed_bias

        packed_w = torch.cat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
        )
        packed_b = (
            torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim=0
            )
            if self.q_proj.bias is not None
            else None
        )
        self._packed_weight = packed_w
        self._packed_bias = packed_b
        self._pack_ptrs = current_ptrs
        return packed_w, packed_b

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Fused QKV linear projection: 1 single GEMM instead of 3
        packed_w, packed_b = self._get_packed_qkv(x)
        qkv = F.linear(x, packed_w, packed_b)
        qkv = qkv.view(batch, seq_len, 3, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv.unbind(0)

        # Dynamic shape dispatch and masking logic
        if valid_token_mask is None or valid_token_mask.all():
            # Fast-path: No invalid tokens
            # PyTorch SDPA automatically selects FlashAttention/MemoryEfficient on accelerator
            # with internal FP32 accumulation.
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=causal, scale=self.scale
            )
        else:
            # Masked path: Construct broadcastable additive attention mask
            # Shape: [B, 1, 1, S]
            attn_mask = torch.zeros(
                (batch, 1, 1, seq_len), device=x.device, dtype=q.dtype
            )
            attn_mask = attn_mask.masked_fill(
                ~valid_token_mask[:, None, None, :], float("-inf")
            )

            if causal:
                causal_mask = torch.zeros(
                    (seq_len, seq_len), device=x.device, dtype=q.dtype
                ).masked_fill(
                    torch.ones(
                        (seq_len, seq_len), device=x.device, dtype=torch.bool
                    ).triu(diagonal=1),
                    float("-inf"),
                )
                attn_mask = attn_mask + causal_mask

            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=False, scale=self.scale
            )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


from ratchet.kernels.triton_fused_ops import (
    triton_fused_layernorm_residual,
    triton_fused_gelu,
)


class OptimizedTransformerBlock(nn.Module):
    """Fused transformer block with pre-LayerNorm, fused QKV SDPA, and exact GELU FFN."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = OptimizedSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        # Pre-LN Self-Attention with fused LayerNorm
        norm1_out, _ = triton_fused_layernorm_residual(
            x,
            residual=None,
            weight=self.norm1.weight,
            bias=self.norm1.bias,
            eps=self.norm1.eps,
        )
        attn_out = self.attention(norm1_out, valid_token_mask, causal)
        x = x + attn_out

        # Pre-LN FFN with fused LayerNorm and fused GELU
        norm2_out, _ = triton_fused_layernorm_residual(
            x,
            residual=None,
            weight=self.norm2.weight,
            bias=self.norm2.bias,
            eps=self.norm2.eps,
        )
        h = self.ffn_in(norm2_out)
        h = triton_fused_gelu(h)
        x = x + self.ffn_out(h)

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class OptimizedTransformer(nn.Module):
    """Optimized full transformer model strictly weight-compatible with baseline."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.causal = causal

        self.layers = nn.ModuleList(
            [
                OptimizedTransformerBlock(d_model, num_heads, ffn_dim)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x
