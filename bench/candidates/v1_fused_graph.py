"""Candidate v1 — fused Q|K|V, fp16 GEMMs, SDPA, CUDA graph.

Generation 1. The root of the tree. 3.11x geomean across the 13 runnable configs, every
row passing correctness.

KEPT AS A STEPPING STONE, NOT AS A RECOMMENDATION. v2 supersedes it everywhere. It is
preserved because clade metaproductivity scores a parent by its descendants' outcomes,
and because its specific defect is the most instructive thing measured so far:

    It looks like it uses FlashAttention. It never does, on any config.

Two independent disqualifications, neither of which raises:
  * q/k/v are cast back to fp32 before the SDPA call (the `.float()` on the fused QKV
    output, below);
  * the padding mask is forwarded even when it is all-True.

`F.scaled_dot_product_attention` selects the best backend that ACCEPTS the arguments it
is given, so a kernel that merely fails to qualify costs you the speed silently. The
backend actually selected here, confirmed in the profiler on every row, is the fp32
memory-efficient CUTLASS path `fmha_cutlassF_f32_aligned_64x64_rf_sm80`.

See `docs/findings/04-the-flash-attention-that-never-was.md`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build(baseline_cls):
    """Return a candidate class derived from the benchmark's BaselineTransformer."""

    class CandidateV1(baseline_cls):
        use_graph = True

        def _prime(self):
            # Lazy: the harness builds on CPU in fp32, strict-copies weights, and only
            # then moves to device/dtype. Precomputing in __init__ caches CPU garbage.
            # Held as a plain attribute so strict load_state_dict still matches keys.
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
                ))
            self._graph = None

        def _core(self, x, mask):
            lp = torch.float16
            for layer, cached in zip(self.layers, self._cache):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                b, s, _ = x.shape

                # THE DEFECT IS THIS .float(): it disqualifies flash, which requires
                # fp16/bf16 inputs. v2 removes it.
                qkv = F.linear(layer.norm1(x).to(lp), qkv_w, qkv_b).float()
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)

                # AND THIS MASK: forwarded even when all-True, which also disqualifies
                # flash. v2 elides it once at priming.
                attn_mask = mask[:, None, None, :] if mask is not None else None
                ctx = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=self.config.causal)

                o = F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model).to(lp),
                             out_w, out_b).float()
                if mask is not None:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o

                h = F.linear(layer.norm2(x).to(lp), in_w, in_b)
                x = x + F.linear(F.gelu(h.float(), approximate="none").to(lp),
                                 ffn_w, ffn_b).float()
                if mask is not None:
                    x = x.masked_fill(~mask[..., None], 0)

            x = self.final_norm(x)
            return x.masked_fill(~mask[..., None], 0) if mask is not None else x

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime()
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
            return self._static_y.clone()   # clone: the harness holds it across replays

    return CandidateV1
