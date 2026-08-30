"""CUDA-graph-captured fp16 transformer using this repo's own Triton kernels.

Ported from the `ben` branch's winning recipe (v1/v2: fp16 + fused QKV + real flash +
CUDA graph, ~3.1x geomean over torch.compile on RTX 4070), but every heavy op runs in a
kernel written in THIS repo -- `flash_attention` and `linear_tf32` (which becomes an fp16
tensor-core GEMM when handed fp16 operands) -- rather than SDPA/cuBLAS, so it is a valid
kernel submission. Tuned/measured on GB10 (sm_121).

Three levers, each independently worth a lot:
  * CUDA graph: the many-kernel forward is captured once and replayed as a single launch,
    which is decisive on the launch-bound configs (small batch / short seq) where per-op
    overhead otherwise sinks a hand-written kernel.
  * fp16 tensor cores: the baseline runs fp32; fp16 halves the traffic and hits the tensor
    cores. The residual stream stays fp32 so accuracy holds through the layers.
  * fused QKV: the three projections share one input and fuse into a single GEMM.

Every announced competition row is causal, so is_causal is always on.
"""

from __future__ import annotations

from typing import Optional

import torch

from ratchet.kernels.flash_attention import flash_attention
from ratchet.kernels.linear_tf32 import linear_tf32

_LP = torch.float16


def _prime(self):
    """Cache low-precision fused weights once. dtype is a dispatch knob (self._lp, default
    fp16). Held as plain attributes so strict load_state_dict still matches baseline keys."""
    lp = getattr(self, "_lp", _LP)
    self._cache = []
    for layer in self.layers:
        a = layer.attention
        self._cache.append((
            torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).to(lp).contiguous(),
            torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).to(lp).contiguous(),
            a.out_proj.weight.to(lp).contiguous(), a.out_proj.bias.to(lp).contiguous(),
            layer.ffn_in.weight.to(lp).contiguous(), layer.ffn_in.bias.to(lp).contiguous(),
            layer.ffn_out.weight.to(lp).contiguous(), layer.ffn_out.bias.to(lp).contiguous(),
        ))
    self._graph = None


def _core(self, x, mask):
    causal = self.config.causal
    lp = getattr(self, "_lp", _LP)
    for layer, cached in zip(self.layers, self._cache):
        a = layer.attention
        qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
        b, s, _ = x.shape
        d = a.d_model

        qkv = linear_tf32(layer.norm1(x).to(lp), qkv_w, qkv_b)      # low-precision fused QKV
        q, k, v = qkv.split(d, dim=-1)
        q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2).contiguous()
        k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2).contiguous()
        v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2).contiguous()
        ctx = flash_attention(q, k, v, causal=causal)               # fp16 flash
        ctx = ctx.transpose(1, 2).reshape(b, s, d)
        x = x + linear_tf32(ctx, out_w, out_b).float()              # residual in fp32

        h = linear_tf32(layer.norm2(x).to(lp), in_w, in_b, gelu=True)
        x = x + linear_tf32(h, ffn_w, ffn_b).float()

    return self.final_norm(x)


def graphed_forward(self, x: torch.Tensor,
                    valid_token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Padding masks are not handled by the fp16 flash path; fall back to the exact baseline.
    if valid_token_mask is not None and not bool(valid_token_mask.all()):
        return baseline_forward(self, x, valid_token_mask)

    if not hasattr(self, "_cache"):
        _prime(self)

    # Dispatch may turn the graph off for compute-bound / memory-heavy shapes, where the
    # capture is neutral or its static buffers are too large. The kernels are the same.
    if not getattr(self, "_use_graph", True):
        return _core(self, x, None)

    # Warm up on a side stream: this runs Triton autotune (which benchmarks, and so must
    # NOT happen during graph capture) and lets cuBLAS/allocator settle.
    if self._graph is None:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(5):
                _core(self, x, None)
        torch.cuda.current_stream().wait_stream(side)

        self._static_x = x.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._static_y = _core(self, self._static_x, None)
        self._graph = graph

    self._static_x.copy_(x)
    self._graph.replay()
    return self._static_y.clone()


def baseline_forward(self, x, valid_token_mask=None):
    causal = self.config.causal
    for layer in self.layers:
        x = layer(x, valid_token_mask, causal)
    return self.final_norm(x)
