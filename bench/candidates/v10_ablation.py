"""Generation 10 -- ablate v9a's inherited stack under compilation.

Branch: cand/g10/ablation. Parent of all three: v9a_compiled_core.

WHY THIS BEFORE MORE OPTIMIZATION
---------------------------------
v9a is nine generations of accumulated tricks: fused Q|K|V (v1), fp16 GEMM cache with an
fp32 residual (v1/v5/v6), flash attention via the elided mask (v2), L2-sized batch
chunking (v3), the right-padding proof (v8), and finally torch.compile (v9a). Each was
justified against the state of the world when it was added -- and every one of those
justifications predates Inductor being in the loop.

The dossier's skeptic paper (arXiv 2602.16805, *Simple Baselines are Competitive with Code
Evolution*) is explicit that this is the ablation you owe before crediting the machinery
for a win. Inductor fuses, picks its own kernels, and manages its own memory; some of what
we hand it may now be redundant, and some may actively constrain it.

Three siblings, each removing exactly one inherited trick from v9a:

  v10a_no_chunk   -- drop L2-sized batch chunking. Inductor and CUDA graphs may already
                     handle the working set; chunking then only costs a Python loop.
  v10b_no_fusedqkv-- drop the fused Q|K|V cat, use three separate projections. Inductor
                     may fuse them itself, and the cat costs a copy at prime time.
  v10c_no_fp16    -- drop the fp16 weight cache entirely, hand Inductor fp32 and let it
                     choose precision. Tests whether our hand-rolled mixed precision still
                     beats the compiler's own.

A regression on removal means the trick still pays. A tie means it is now dead weight and
the submission is carrying complexity it cannot justify.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v8_padfast import build as build_v8


def _compiled(cls):
    """Wrap a candidate class so its _core is compiled, matching v9a exactly."""
    class Compiled(cls):
        use_graph = False

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if not hasattr(self, "_compiled_core"):
                self._compiled_core = torch.compile(
                    self._core, mode="reduce-overhead", dynamic=False)
            return self._compiled_core(x, valid_token_mask)
    return Compiled


def build_no_chunk(baseline_cls):
    """v9a minus L2-sized batch chunking."""
    v8_cls = build_v8(baseline_cls)

    class NoChunk(v8_cls):
        def forward(self, x, valid_token_mask=None):   # bypass v3's chunk loop entirely
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            return self._core(x, valid_token_mask)
    return _compiled(NoChunk)


def build_no_fused_qkv(baseline_cls):
    """v9a minus the fused Q|K|V projection -- three separate linears instead."""
    v8_cls = build_v8(baseline_cls)

    class NoFusedQKV(v8_cls):
        def _prime(self, mask):
            super()._prime(mask)
            lp = torch.float16
            self._split = [(l.attention.q_proj.weight.to(lp), l.attention.q_proj.bias.to(lp),
                            l.attention.k_proj.weight.to(lp), l.attention.k_proj.bias.to(lp),
                            l.attention.v_proj.weight.to(lp), l.attention.v_proj.bias.to(lp))
                           for l in self.layers]

        def _core(self, x, mask):
            lp = torch.float16
            nomask = self._fastpath and self._nomask
            zero = not self._nomask
            if not self._fastpath:
                return super()._core(x, mask)
            for layer, cached, sp in zip(self.layers, self._cache, self._split):
                a = layer.attention
                _, _, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                qw, qb, kw, kb, vw, vb = sp
                b, s, _ = x.shape
                n = layer.norm1(x).to(lp)
                q = F.linear(n, qw, qb).view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = F.linear(n, kw, kb).view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = F.linear(n, vw, vb).view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                o = F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model), out_w, out_b).float()
                if zero:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o
                h = F.linear(layer.norm2(x).to(lp), in_w, in_b)
                x = x + F.linear(F.gelu(h, approximate="none"), ffn_w, ffn_b).float()
                if zero:
                    x = x.masked_fill(~mask[..., None], 0)
            x = self.final_norm(x)
            return x.masked_fill(~mask[..., None], 0) if zero else x
    return _compiled(NoFusedQKV)


def build_no_fp16(baseline_cls):
    """v9a minus the fp16 weight cache -- pure fp32, Inductor chooses precision."""
    v8_cls = build_v8(baseline_cls)

    class NoFp16(v8_cls):
        def _core(self, x, mask):
            nomask = self._fastpath and self._nomask
            zero = not self._nomask
            if not self._fastpath:
                return super()._core(x, mask)
            for layer in self.layers:
                a = layer.attention
                b, s, _ = x.shape
                n = layer.norm1(x)
                qkv = torch.cat([a.q_proj(n), a.k_proj(n), a.v_proj(n)], dim=-1)
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                o = a.out_proj(ctx.transpose(1, 2).reshape(b, s, a.d_model))
                if zero:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o
                x = x + layer.ffn_out(F.gelu(layer.ffn_in(layer.norm2(x)),
                                             approximate="none"))
                if zero:
                    x = x.masked_fill(~mask[..., None], 0)
            x = self.final_norm(x)
            return x.masked_fill(~mask[..., None], 0) if zero else x
    return _compiled(NoFp16)
