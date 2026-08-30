"""Speed-lever exploration seams (not yet committed candidates).

Each is injected at the UserOptimizedTransformer seam and judged by the authoritative
evaluator (correctness first) + the drift-robust timer. The point is to find whether ANY
path beats the cuBLAS-fp32 baseline under the fp32 accuracy gate, before investing in a
hand-tuned kernel. Negative results are kept, per the repo contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F


@contextmanager
def _tf32(enabled: bool):
    """Enable cuBLAS/cuDNN TF32 tensor cores for the optimized path only, then restore.

    The baseline runs outside this scope, so it keeps true fp32 -- the comparison stays
    honest even when baseline and candidate are interleaved in one process.
    """
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_dnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_dnn


def _block(layer, x, causal):
    attn = layer.attention
    h = layer.norm1(x)
    B, N, _ = h.shape
    q = attn._split_heads(attn.q_proj(h))
    k = attn._split_heads(attn.k_proj(h))
    v = attn._split_heads(attn.v_proj(h))
    ctx = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    ctx = ctx.transpose(1, 2).contiguous().view(B, N, attn.d_model)
    x = x + attn.out_proj(ctx)
    x = x + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(x)), approximate="none"))
    return x


def forward_cublas_tf32(self, x: torch.Tensor,
                        valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """SDPA attention + cuBLAS TF32 GEMMs (tensor cores) for every projection and the FFN.
    The performance ceiling under the fp32 gate using only library kernels."""
    causal = self.config.causal
    if valid_token_mask is not None and not bool(valid_token_mask.all()):
        for layer in self.layers:
            x = layer(x, valid_token_mask, causal)
        return self.final_norm(x)
    with _tf32(True):
        for layer in self.layers:
            x = _block(layer, x, causal)
        return self.final_norm(x)


def forward_sdpa_fp32(self, x: torch.Tensor,
                      valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """SDPA attention + plain fp32 GEMMs. Isolates the SDPA/fusion effect from TF32."""
    causal = self.config.causal
    if valid_token_mask is not None and not bool(valid_token_mask.all()):
        for layer in self.layers:
            x = layer(x, valid_token_mask, causal)
        return self.final_norm(x)
    for layer in self.layers:
        x = _block(layer, x, causal)
    return self.final_norm(x)
