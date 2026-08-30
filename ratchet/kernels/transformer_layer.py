"""Optimized transformer forward that delegates attention to the flash kernel.

This is the body for the authoritative evaluator's `UserOptimizedTransformer` seam. It is
injected at runtime (monkeypatch) so `torch_transformer_benchmark.py` stays byte-for-byte
identical -- its baseline, correctness rule, and input generation are never touched. Only
the attention core is replaced; projections, LayerNorm, FFN, and residuals reuse the
baseline modules and weights, so `copy_model_weights(strict=True)` stays valid.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ratchet.kernels.flash_attention import flash_attention
from ratchet.kernels.layernorm import layernorm
from ratchet.kernels.linear_tf32 import linear_tf32


def optimized_forward(self, x: torch.Tensor,
                      valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    causal = self.config.causal

    # The flash kernel handles causal + sequence-length masking, not arbitrary key
    # padding. With real padding, fall back to the exact baseline blocks so correctness
    # is never traded for speed.
    if valid_token_mask is not None and not bool(valid_token_mask.all()):
        for layer in self.layers:
            x = layer(x, valid_token_mask, causal)
        return self.final_norm(x)
    for layer in self.layers:
        attn = layer.attention
        h = layer.norm1(x)
        B, N, _ = h.shape
        q = attn._split_heads(attn.q_proj(h))     # [B, H, N, Dh]
        k = attn._split_heads(attn.k_proj(h))
        v = attn._split_heads(attn.v_proj(h))
        ctx = flash_attention(q, k, v, causal=causal)
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, attn.d_model)
        x = x + attn.out_proj(ctx)
        x = x + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none"))

    x = self.final_norm(x)
    if valid_token_mask is not None:
        x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x


def optimized_forward_tf32(self, x: torch.Tensor,
                           valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """E3: same math as the baseline, but every GEMM (q/k/v/out projections and both FFN
    layers) runs on the tensor cores via the TF32 linear kernel, and attention uses the
    flash kernel. Projections+FFN are the real cost at seq=128; the fp32 baseline runs
    them without tensor cores, so this is where the speedup is.
    """
    causal = self.config.causal

    if valid_token_mask is not None and not bool(valid_token_mask.all()):
        for layer in self.layers:
            x = layer(x, valid_token_mask, causal)
        return self.final_norm(x)

    for layer in self.layers:
        attn = layer.attention
        h = layer.norm1(x)
        B, N, _ = h.shape
        Dh = attn.head_dim

        def heads(t):
            return t.view(B, N, attn.num_heads, Dh).transpose(1, 2).contiguous()

        q = heads(linear_tf32(h, attn.q_proj.weight, attn.q_proj.bias))
        k = heads(linear_tf32(h, attn.k_proj.weight, attn.k_proj.bias))
        v = heads(linear_tf32(h, attn.v_proj.weight, attn.v_proj.bias))
        ctx = flash_attention(q, k, v, causal=causal)
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, attn.d_model)
        x = x + linear_tf32(ctx, attn.out_proj.weight, attn.out_proj.bias)

        g = layer.norm2(x)
        g = linear_tf32(g, layer.ffn_in.weight, layer.ffn_in.bias, gelu=True)
        x = x + linear_tf32(g, layer.ffn_out.weight, layer.ffn_out.bias)

    return self.final_norm(x)


def optimized_forward_full(self, x: torch.Tensor,
                           valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """E5: the whole block in custom kernels -- Triton LayerNorm, fused QKV, TF32 GEMMs,
    flash attention. Only the residual adds stay in torch. Completeness milestone for
    "a GPU kernel for a transformer layer"; not expected to beat cuBLAS on FLOPs.
    """
    causal = self.config.causal

    if valid_token_mask is not None and not bool(valid_token_mask.all()):
        for layer in self.layers:
            x = layer(x, valid_token_mask, causal)
        return self.final_norm(x)

    for layer in self.layers:
        attn = layer.attention
        if not hasattr(attn, "_qkv_w"):
            attn._qkv_w = torch.cat(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], 0).contiguous()
            attn._qkv_b = torch.cat(
                [attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], 0).contiguous()

        h = layernorm(x, layer.norm1.weight, layer.norm1.bias, layer.norm1.eps)
        B, N, _ = h.shape
        d = attn.d_model
        qkv = linear_tf32(h, attn._qkv_w, attn._qkv_b)

        def heads(t):
            return t.view(B, N, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()

        ctx = flash_attention(heads(qkv[..., :d]), heads(qkv[..., d:2 * d]),
                              heads(qkv[..., 2 * d:]), causal=causal)
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, d)
        x = x + linear_tf32(ctx, attn.out_proj.weight, attn.out_proj.bias)

        g = layernorm(x, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
        g = linear_tf32(g, layer.ffn_in.weight, layer.ffn_in.bias, gelu=True)
        x = x + linear_tf32(g, layer.ffn_out.weight, layer.ffn_out.bias)

    return layernorm(x, self.final_norm.weight, self.final_norm.bias, self.final_norm.eps)


def optimized_forward_qkv(self, x: torch.Tensor,
                          valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """E4: E3 + fused QKV. The three projection GEMMs share one input (norm1(x)), so they
    fuse into a single [3*d_model, d_model] GEMM -- one launch and one input read instead
    of three. The concatenated weight/bias is built once per module and cached.
    """
    causal = self.config.causal

    if valid_token_mask is not None and not bool(valid_token_mask.all()):
        for layer in self.layers:
            x = layer(x, valid_token_mask, causal)
        return self.final_norm(x)

    for layer in self.layers:
        attn = layer.attention
        if not hasattr(attn, "_qkv_w"):
            attn._qkv_w = torch.cat(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], 0).contiguous()
            attn._qkv_b = torch.cat(
                [attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], 0).contiguous()

        h = layer.norm1(x)
        B, N, _ = h.shape
        d = attn.d_model
        qkv = linear_tf32(h, attn._qkv_w, attn._qkv_b)          # [B, N, 3*d]

        def heads(t):
            return t.view(B, N, attn.num_heads, attn.head_dim).transpose(1, 2).contiguous()

        q = heads(qkv[..., :d])
        k = heads(qkv[..., d:2 * d])
        v = heads(qkv[..., 2 * d:])
        ctx = flash_attention(q, k, v, causal=causal)
        ctx = ctx.transpose(1, 2).contiguous().view(B, N, d)
        x = x + linear_tf32(ctx, attn.out_proj.weight, attn.out_proj.bias)

        g = layer.norm2(x)
        g = linear_tf32(g, layer.ffn_in.weight, layer.ffn_in.bias, gelu=True)
        x = x + linear_tf32(g, layer.ffn_out.weight, layer.ffn_out.bias)

    return self.final_norm(x)
