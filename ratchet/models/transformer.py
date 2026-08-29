"""Optimized transformer layer implementation with shape-aware dispatch and fused operators."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimizedSelfAttention(nn.Module):
    """Shape-aware optimized self-attention with fast scaled dot-product paths."""

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

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

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


class OptimizedTransformerBlock(nn.Module):
    """Fused transformer block with pre-LayerNorm, SDPA, and exact GELU FFN."""

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
        # Pre-LN Self-Attention with residual connection
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)

        # Pre-LN FFN with exact GELU and residual connection
        h = self.ffn_in(self.norm2(x))
        h = F.gelu(h, approximate="none")
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
