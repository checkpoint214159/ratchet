"""Candidate v20 -- a fused QKV GEMM that writes head-major, so flash reads contiguously.

Generation 20. Parent: v18_capture_insurance. Branch: cand/g20/headmajor-qkv.
SIBLING of v19_norm_fused: both fork from v18, neither is an ancestor of the other. First
genuine fork created under finding 28's branching discipline.

THE TAX, WHICH IS OURS
----------------------
Since v2 the candidate computes one fused QKV GEMM to [B, S, 3D], splits it, and takes
`.view(B, S, H, hd).transpose(1, 2)`. That transpose is a VIEW, so FlashAttention reads
q/k/v through a stride that jumps by 3*D on the innermost axis. Measured:

    config 6   flash on STRIDED 4055.8 us   on CONTIGUOUS 2277.4 us   1.78x
    config 13                    303.3 us                 284.4 us   1.07x
    config 1                      32.5 us                  32.8 us   0.99x

The tax is real, large, and specific to the high-token-count regime. It cannot be
collected by repacking: `.contiguous()` on q/k/v costs 3477.5 us at config 6, more than
the 1778 us it recovers. It can only be collected by a GEMM that owns its epilogue and
scatters each tile straight into head-major buffers, paying no extra pass over memory.

MEASURED, proj+attention segment at config 6, against exactly what v18 does today:

    incumbent (cuBLAS + strided flash)   6354.8 us
    hand-written, tuned                  5465.3 us      1.163x

The first tiling attempt LOST (0.88x, BN=64). BN=128 wins. The output axis is 3*D=384
wide and a 64-wide tile fragments it -- a mechanism argument cannot tell you that, only
autotuning can, which is the division of labour spec 03 describes.

WHAT L33 SAYS TO EXPECT, STATED BEFORE THE SWEEP
------------------------------------------------
The segment is ~38% of config 6 (flash 25.3% + the QKV GEMM 13.3% in a clean profile), so
1.163x on it bounds the end-to-end gain at about 1/(0.62 + 0.38/1.163) = 1.05x on config 6
-- roughly 4.5% of total matrix wall time, and well under 1% on the 13-config geomean,
since only config 6 clears the predicate. **Anything larger than that in the sweep would
be evidence of a measurement error, not of a bigger win.**

DISPATCH
--------
`worth_it` requires >= 500k tokens, the measured crossover between config 6 (1.28M, 1.78x
tax) and config 13 (65k, 1.07x). Total token count, not a config id. Below it the fallback
is v18's path unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v18_capture_insurance import build as build_v18
from ..kernels.qkv_headmajor import qkv_headmajor, worth_it


def build(baseline_cls):
    v18_cls = build_v18(baseline_cls)

    class CandidateV20(v18_cls):
        headmajor_used: bool = False
        headmajor_reason: str = "undecided"

        def _prime(self, mask):
            super()._prime(mask)
            # The fused QKV weight pre-transposed to [D, 3D]; the kernel contracts over
            # the leading axis. Done once -- per call it would cost more than it saves.
            self._qkv_t = [cached[0].t().contiguous() for cached in self._cache]

        def _decide_headmajor(self, x):
            b, s, d = x.shape
            h = self.layers[0].attention.num_heads
            if worth_it(b, s, d, h):
                self.headmajor_used = True
                self.headmajor_reason = f"head-major: {b*s} tokens"
            else:
                self.headmajor_used = False
                self.headmajor_reason = f"declined: {b*s} tokens below the stride-tax crossover"

        def _core(self, x, mask):
            if self.headmajor_reason == "undecided":
                self._decide_headmajor(x)
            if not self.headmajor_used or not self._nomask:
                return super()._core(x, mask)

            lp = torch.float16
            for layer, cached, ffn_t, qkv_t in zip(self.layers, self._cache,
                                                   self._ffn_t, self._qkv_t):
                a = layer.attention
                qkv_b, out_w, out_b = cached[1], cached[2], cached[3]
                b, s, d = x.shape

                xn = layer.norm1(x).to(lp).view(-1, d)
                q, k, v = qkv_headmajor(xn, qkv_t, qkv_b, b, s, a.num_heads)
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                x = x + F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model),
                                 out_w, out_b).float()

                if self.fused_ffn_used:
                    from ..kernels.ffn_fused import fused_ffn
                    w1t, b1, w2t, b2 = ffn_t
                    xn2 = layer.norm2(x).to(lp).view(-1, d)
                    x = fused_ffn(xn2, x.view(-1, d), w1t, b1, w2t, b2,
                                  self.BLOCK_M, self.NUM_WARPS).view(b, s, d)
                else:
                    h = F.linear(layer.norm2(x).to(lp), ffn_t[0].t(), ffn_t[1])
                    x = x + F.linear(F.gelu(h.float(), approximate="none").to(lp),
                                     ffn_t[2].t(), ffn_t[3]).float()

            return self.final_norm(x)

    return CandidateV20
