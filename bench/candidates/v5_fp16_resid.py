"""Candidate v5 -- keep the residual stream in fp16 end to end.

Generation 5. Parent: v3_chunked.

THE HYPOTHESIS
--------------
Profiling put dtype conversion at 12.8-26.8% of candidate kernel time. v2/v3/v4 round-trip
through fp32 several times per layer:

    norm1(x_fp32).to(fp16) -> qkv -> attention -> out_proj().float()  -> x_fp32 + o
    norm2(x_fp32).to(fp16) -> ffn_in -> gelu(h.float()).to(fp16) -> ffn_out().float()

Six conversions per layer, four layers. This candidate keeps x in fp16 for the whole
stack, caches fp16 LayerNorm weights so the norms accept it, runs GELU in fp16, and
converts to fp32 exactly once at the end.

WHY THIS MIGHT FAIL, WHICH IS THE POINT OF MEASURING IT
-------------------------------------------------------
The residual stream accumulates. Four layers of fp16 addition compound representation
error that the fp32 stream absorbed, and the tolerance budget is already nearly spent:
config 7 sits at max_abs 1.88e-3 against a 2.0e-3 limit, a 6% margin. If this candidate
fails correctness anywhere, that is a RESULT -- it establishes where the precision floor
actually is, and it is recorded rather than discarded.

LayerNorm is the one place precision is preserved deliberately: torch computes the
reduction in fp32 internally even for fp16 inputs, which is what makes an fp16 residual
stream survivable at all rather than obviously hopeless.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build(baseline_cls):

    class CandidateV5(baseline_cls):
        use_graph = True

        def _prime(self, mask):
            lp = torch.float16
            self._cache = []
            for layer in self.layers:
                a = layer.attention
                self._cache.append((
                    torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).to(lp),
                    torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).to(lp),
                    a.out_proj.weight.to(lp), a.out_proj.bias.to(lp),
                    layer.ffn_in.weight.to(lp), layer.ffn_in.bias.to(lp),
                    layer.ffn_out.weight.to(lp), layer.ffn_out.bias.to(lp),
                    # fp16 norm weights: without these an fp16 activation cannot be fed
                    # to the fp32 nn.LayerNorm the harness constructed.
                    layer.norm1.weight.to(lp), layer.norm1.bias.to(lp),
                    layer.norm2.weight.to(lp), layer.norm2.bias.to(lp),
                ))
            self._fnorm = (self.final_norm.weight.to(lp), self.final_norm.bias.to(lp))
            self._eps = self.layers[0].norm1.eps
            self._nomask = mask is None or bool(mask.all().item())
            self._graph = None

        def _core(self, x, mask):
            lp = torch.float16
            nomask = self._nomask
            shape = (x.shape[-1],)
            h16 = x.to(lp)                      # the ONLY downcast

            for layer, c in zip(self.layers, self._cache):
                a = layer.attention
                (qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b,
                 n1w, n1b, n2w, n2b) = c
                b, s, _ = h16.shape

                qkv = F.linear(F.layer_norm(h16, shape, n1w, n1b, self._eps), qkv_w, qkv_b)
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

                o = F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model), out_w, out_b)
                if not nomask:
                    o = o.masked_fill(~mask[..., None], 0)
                h16 = h16 + o                    # residual stays fp16

                g = F.linear(F.layer_norm(h16, shape, n2w, n2b, self._eps), in_w, in_b)
                h16 = h16 + F.linear(F.gelu(g, approximate="none"), ffn_w, ffn_b)
                if not nomask:
                    h16 = h16.masked_fill(~mask[..., None], 0)

            out = F.layer_norm(h16, shape, *self._fnorm, self._eps).float()
            return out if nomask else out.masked_fill(~mask[..., None], 0)

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if not self.use_graph:
                return self._core(x, valid_token_mask)

            if self._graph is None:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._core(x, valid_token_mask)
                torch.cuda.current_stream().wait_stream(side)
                self._static_x = x.clone()
                self._static_m = None if valid_token_mask is None else valid_token_mask.clone()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._static_y = self._core(self._static_x, self._static_m)
                self._graph = graph

            self._static_x.copy_(x)
            if self._static_m is not None:
                self._static_m.copy_(valid_token_mask)
            self._graph.replay()
            return self._static_y.clone()

    return CandidateV5
