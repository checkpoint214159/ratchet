"""Candidate v6 -- keep the fp32 residual, drop only the GELU round-trip.

Generation 6. Parent: v3_chunked.

v5 established that the fp32 residual stream is load-bearing: removing it was ~1.4x
faster and failed 11 of 13 configs by 3.3-5.3x of the tolerance budget
(docs/findings/08-the-fp32-residual-is-load-bearing.md).

But the ~6 conversions per layer are not equally load-bearing, and that finding named the
distinction: the residual ACCUMULATES across layers, while an elementwise op does not.
This candidate removes exactly one non-accumulating round-trip and changes nothing else:

    v3:  x = x + F.linear(F.gelu(h.float(), approximate="none").to(fp16), w, b).float()
    v6:  x = x + F.linear(F.gelu(h,         approximate="none"),          w, b).float()
                          ^^^ h is already fp16; the upcast and downcast are both dropped

Two conversions per layer, four layers, eight per forward. The residual stays fp32, so
nothing compounds.

WHAT THE PRECISION RISK ACTUALLY IS. `h` is already an fp16 tensor either way -- v3
upcasts a value that was rounded to fp16 by the preceding `F.linear`, so the upcast
recovers no information. The only new error is one fp16 rounding of GELU's OUTPUT, about
5e-4 relative, on the FFN branch alone. That is a real risk against a budget already
spending 1.88e-3 of 2.0e-3 on config 7, which is why it is measured rather than assumed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v3_chunked import build as build_v3


def build(baseline_cls):
    v3_cls = build_v3(baseline_cls)

    class CandidateV6(v3_cls):
        def _core(self, x, mask):
            lp = torch.float16
            nomask = self._nomask

            for layer, cached in zip(self.layers, self._cache):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                b, s, _ = x.shape

                qkv = F.linear(layer.norm1(x).to(lp), qkv_w, qkv_b)
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)

                if nomask:
                    ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    ctx = F.scaled_dot_product_attention(
                        q.float(), k.float(), v.float(),
                        attn_mask=mask[:, None, None, :], is_causal=True).to(lp)

                o = F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model),
                             out_w, out_b).float()
                if not nomask:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o                                   # residual stays fp32

                h = F.linear(layer.norm2(x).to(lp), in_w, in_b)
                # THE ONLY CHANGE: gelu in fp16, no upcast-downcast pair around it.
                x = x + F.linear(F.gelu(h, approximate="none"), ffn_w, ffn_b).float()
                if not nomask:
                    x = x.masked_fill(~mask[..., None], 0)

            x = self.final_norm(x)
            return x if nomask else x.masked_fill(~mask[..., None], 0)

    return CandidateV6
