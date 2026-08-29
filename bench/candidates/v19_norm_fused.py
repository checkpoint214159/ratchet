"""Candidate v19 -- fold the residual add and BOTH LayerNorms into the megakernel.

Generation 19. Parent: v18_capture_insurance. Branch: cand/g19/norm-fused.
Idea from proposal D-01, measured by research agent D and re-measured here.
FIRST candidate created under the branching discipline of finding 28.

WHAT IS BEING DELETED, AND WHY IT CANNOT INSTEAD BE SPED UP
-----------------------------------------------------------
v17/v18's kernel takes an already-normalized `xn` and an already-added residual, both
produced by a SEPARATE Inductor kernel that reads the residual and the attention output,
adds them, normalizes, and writes two full-size tensors to HBM -- which the megakernel then
reads straight back. A third kernel reads the megakernel's output and normalizes it for the
next layer.

Those three kernels are 35.0% of config 6's time across nine calls, and every one runs at
661-672 GB/s against the device's measured 613.7 GB/s. **They are at the achievable
bandwidth roofline.** No tiling, no scheduling and no compiler flag makes a kernel faster
than the memory it must move. They can only be made not to exist.

At d_model = 128 a token's whole row is one tile, so the kernel already holds the complete
output row in registers and can do both reductions itself. Traffic per token falls from
28*D bytes to 12*D, with weight traffic unchanged -- so finding 25's amortization crossover
still governs the dispatch and is reused verbatim.

MEASURED at the op level, against exactly what v17 does today:

    cfg 1   M=8192       106.1 us ->    42.3 us   2.51x
    cfg 13  M=65536      731.9 us ->   229.3 us   3.19x
    cfg 6   M=1280000  14725.0 us ->  3839.0 us   3.84x

L33 SAYS THAT NUMBER WILL SHRINK, AND BY HOW MUCH
-------------------------------------------------
The segment replaced is ~35% of layer time, so a 3.8x on it bounds the end-to-end gain at
about 1/(0.65 + 0.35/3.84) = 1.35x on config 6, not 3.8x. Stated before measuring, per the
discipline that has now landed twice (L33's dilution factor on v16, and v17's own
prediction).

PRECISION
---------
The residual stays fp32 from the add, through the normalization, to the store, never
round-tripping HBM -- strictly FEWER rounding steps than the Inductor kernel it replaces,
which materializes the residual in fp32 and the normalized copy in fp16. Finding 08's fp32
residual is preserved verbatim, and GELU is the exact-erf form matching the reference's
`approximate="none"`.

The one place a downcast would be NEW is the model's output: `final_norm`'s result is
returned in fp32. So the last layer sets `store_next=False` and `final_norm` is applied
outside in fp32. Every other layer's normalized output was already fp16 in v17, so folding
it in adds no rounding that was not already being paid.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v18_capture_insurance import build as build_v18
from ..kernels.ffn_fused import amortizes, fits, fused_ffn_normed


def build(baseline_cls):
    v18_cls = build_v18(baseline_cls)

    class CandidateV19(v18_cls):
        def _prime(self, mask):
            super()._prime(mask)
            lp = torch.float16
            # norm weights in fp16, and the NEXT consumer's norm for each layer: the
            # following layer's norm1, or final_norm for the last.
            self._norms = []
            n = len(self.layers)
            for i, layer in enumerate(self.layers):
                nxt = self.layers[i + 1].norm1 if i + 1 < n else self.final_norm
                self._norms.append((
                    layer.norm2.weight.to(lp), layer.norm2.bias.to(lp),
                    nxt.weight.to(lp), nxt.bias.to(lp),
                    float(layer.norm2.eps), i == n - 1,
                ))

        def _core(self, x, mask):
            if not self.fused_ffn_used or not self._nomask:
                return super()._core(x, mask)

            lp = torch.float16
            d = x.shape[2]
            xn = self.layers[0].norm1(x).to(lp)          # only the FIRST norm1 is separate
            for layer, cached, ffn_t, nrm in zip(self.layers, self._cache,
                                                 self._ffn_t, self._norms):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b = cached[0], cached[1], cached[2], cached[3]
                b, s, _ = x.shape

                qkv = F.linear(xn, qkv_w, qkv_b)
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn = F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model),
                                out_w, out_b).float()

                w1t, b1, w2t, b2 = ffn_t
                n2w, n2b, nnw, nnb, eps, is_last = nrm
                y, yn = fused_ffn_normed(
                    x.view(-1, d), attn.view(-1, d), n2w, n2b, w1t, b1, w2t, b2,
                    nnw, nnb, eps, self.BLOCK_M, self.NUM_WARPS,
                    store_next=not is_last)
                x = y.view(b, s, d)
                if not is_last:
                    xn = yn.view(b, s, d)

            # The last layer deliberately did not emit its normalized output: this one is
            # the model's fp32 answer and must not be rounded on the way out.
            return self.final_norm(x)

    return CandidateV19
