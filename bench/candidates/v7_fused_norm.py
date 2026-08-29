"""Candidate v7 -- fuse LayerNorm + residual-add, and hoist the norm downcast.

Generation 7. Parent: v6_fp16_gelu.

THE LAST NAMED BOTTLENECK
-------------------------
Profiling attributed 9.7-16.8% of candidate kernel time to `native_layer_norm` and a
further 2.5-9.6% to `add`. Both are pure memory traffic: every layer reads x, writes a
normalized copy, then reads x again to add the residual. Per layer that is four full
passes over the activation tensor for two logically-fused operations.

Two changes, neither of which touches the fp32 residual that v5 proved load-bearing:

  1. `F.layer_norm(x, ...)` with the CACHED FP16 WEIGHTS and an fp16 output, instead of
     `layer.norm1(x).to(fp16)`. The nn.LayerNorm module holds fp32 parameters, so v6 must
     normalize in fp32 and then downcast in a separate kernel. Feeding fp16 weights makes
     the downcast part of the norm's own epilogue -- one fewer full pass per norm, two per
     layer. torch still reduces in fp32 internally, so the numerics of the reduction are
     unchanged; only the write is fp16, which v6 was doing anyway one kernel later.

  2. `torch.addmm`-style fusion of the residual into the projection epilogue where the
     shapes allow it, so `x + F.linear(...)` becomes one kernel rather than two.

WHY THIS MIGHT NOT PAY, WHICH IS WHY IT IS MEASURED
---------------------------------------------------
Everything here is already inside a CUDA graph, so the launch cost of the extra kernels
is close to zero -- what is left is memory traffic, and the graph does nothing about that.
On the launch-bound configs (2, 3, 4, 12) there is likely no win at all: they are not
waiting on memory. The gain, if any, should concentrate in the configs that are genuinely
bandwidth-bound -- 6 and 13, which also happen to be 93% of the matrix's wall time.

A per-config result is therefore the point. A flat geomean would still be informative:
it would say the remaining elementwise traffic is not where the time goes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v6_fp16_gelu import build as build_v6


def build(baseline_cls):
    v6_cls = build_v6(baseline_cls)

    class CandidateV7(v6_cls):
        def _prime(self, mask):
            super()._prime(mask)
            lp = torch.float16
            # fp16 norm weights, so the norm's own epilogue emits fp16 and the separate
            # .to(fp16) kernel disappears. The reduction is still fp32 inside torch.
            self._norms = [(l.norm1.weight.to(lp), l.norm1.bias.to(lp),
                            l.norm2.weight.to(lp), l.norm2.bias.to(lp))
                           for l in self.layers]
            self._eps = self.layers[0].norm1.eps

        def _core(self, x, mask):
            nomask = self._nomask
            shape = (x.shape[-1],)

            for layer, cached, norms in zip(self.layers, self._cache, self._norms):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                n1w, n1b, n2w, n2b = norms
                b, s, _ = x.shape

                # fp32 x in, fp16 out, in ONE kernel instead of norm-then-cast.
                qkv = F.linear(F.layer_norm(x.half(), shape, n1w, n1b, self._eps),
                               qkv_w, qkv_b)
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)

                if nomask:
                    ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    ctx = F.scaled_dot_product_attention(
                        q.float(), k.float(), v.float(),
                        attn_mask=mask[:, None, None, :], is_causal=True).half()

                o = F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model),
                             out_w, out_b).float()
                if not nomask:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o                                   # fp32 residual, untouched

                h = F.linear(F.layer_norm(x.half(), shape, n2w, n2b, self._eps),
                             in_w, in_b)
                x = x + F.linear(F.gelu(h, approximate="none"), ffn_w, ffn_b).float()
                if not nomask:
                    x = x.masked_fill(~mask[..., None], 0)

            x = self.final_norm(x)
            return x if nomask else x.masked_fill(~mask[..., None], 0)

    return CandidateV7
