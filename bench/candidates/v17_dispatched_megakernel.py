"""Candidate v17 -- the frontier, with the FFN megakernel where it actually pays.

Generation 17. Parent: v13_safe_capture. Branch: cand/g17/dispatched-megakernel.
Recombines the g16 kernel into the g13 frontier -- a MERGE in the evolutionary tree,
which is what the branch protocol calls recombination.

WHAT g16 ESTABLISHED
--------------------
`v16_ffn_megakernel` put the whole FFN block in one Triton kernel and measured
2.493x overall against v13's 2.711x -- not the frontier. But per config it was
-7.6% on config 6 (the largest in the matrix), -5.7% on 7, -2.5% on 13, and
+113% on config 2.

The kernel loads both weight matrices into shared memory once and streams activation
tiles past them, so its advantage is proportional to how many tokens reuse that load.
Speedup is monotone in weight-bytes-per-token, and the sign flips between 1.0 and 8.0:

    w/token  0.051   0.5    1.0    8.0    32     128     512
    v16/v13  -7.6%  -5.7%  -2.5%  ~+1.3%  +1.1%  +49%   +113%

So this candidate uses the kernel only where the hoist is paid for, and v13's path
everywhere else. The predicate (`kernels.ffn_fused.amortizes`) is a ratio of weight
traffic to activation traffic -- shapes and element sizes only, no config id, no
literal token count (rule 2). It selects configs 6, 7 and 13 on this matrix, which are
exactly the three where g16 measured a win, and it would evaluate correctly on a shape
nobody here has seen.

HONEST EXPECTATION, STATED BEFORE MEASURING
-------------------------------------------
Three configs improving 7.6% / 5.7% / 2.5% moves a 13-config geomean by about 1.3%,
to roughly 2.745x. **That is inside the +/-7% noise floor and must not be reported as
a win on the geomean.** What is defensible is the per-config claim: the largest config
in the matrix gets 7.6% faster, and that is a hand-written kernel beating the compiler
at something the compiler structurally cannot do.

Whether that matters depends on an objective nobody has published -- a geomean of
speedups says almost nothing changed, a total-time score says config 6 dominates.
See finding 25.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v13_safe_capture import build as build_v13
from ..kernels.ffn_fused import amortizes, fits, fused_ffn


def build(baseline_cls):
    v13_cls = build_v13(baseline_cls)

    class CandidateV17(v13_cls):
        BLOCK_M = 64
        NUM_WARPS = 8
        fused_ffn_used: bool = False
        fused_ffn_reason: str = "undecided"

        def _prime(self, mask):
            super()._prime(mask)
            self._ffn_t = []
            for _layer, cached in zip(self.layers, self._cache):
                in_w, in_b, ffn_w, ffn_b = cached[4], cached[5], cached[6], cached[7]
                self._ffn_t.append((in_w.t().contiguous(), in_b,
                                    ffn_w.t().contiguous(), ffn_b))

        def _decide_ffn(self, x):
            """Decided ONCE, before compilation and graph capture, so the choice is a
            Python constant by the time anything traces it."""
            b, s, d = x.shape
            f = self.layers[0].ffn_in.weight.shape[0]
            props = torch.cuda.get_device_properties(x.device)
            smem = props.shared_memory_per_block_optin
            if not fits(d, f, 2, self.BLOCK_M, smem):
                self.fused_ffn_used = False
                self.fused_ffn_reason = f"declined: weights exceed {smem} B opt-in smem"
            elif not amortizes(b * s, d, f, 2):
                self.fused_ffn_used = False
                self.fused_ffn_reason = (
                    f"declined: {2*d*f*2/(b*s):.3f} weight-bytes/token, "
                    f"below the amortization crossover")
            else:
                self.fused_ffn_used = True
                self.fused_ffn_reason = f"fused: {2*d*f*2/(b*s):.3f} weight-bytes/token"

        def _core(self, x, mask):
            if not self.fused_ffn_used or not self._nomask:
                return super()._core(x, mask)

            lp = torch.float16
            for layer, cached, ffn_t in zip(self.layers, self._cache, self._ffn_t):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b = cached[0], cached[1], cached[2], cached[3]
                b, s, d = x.shape

                qkv = F.linear(layer.norm1(x).to(lp), qkv_w, qkv_b)
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                x = x + F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model),
                                 out_w, out_b).float()

                w1t, b1, w2t, b2 = ffn_t
                xn = layer.norm2(x).to(lp).view(-1, d)
                x = fused_ffn(xn, x.view(-1, d), w1t, b1, w2t, b2,
                              self.BLOCK_M, self.NUM_WARPS).view(b, s, d)

            return self.final_norm(x)

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if self.fused_ffn_reason == "undecided":
                self._decide_ffn(x)
            return super().forward(x, valid_token_mask)

    return CandidateV17
