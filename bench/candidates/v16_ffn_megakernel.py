"""Candidate v16 -- replace the FFN block with one hand-written Triton kernel.

Generation 16. Parent: v9b_reduce_overhead. Branch: cand/g16/ffn-megakernel.
Node chosen by CMP + Thompson over clade metaproductivity (seed 30 -> 2e855f81).
Idea drawn from the scored proposal distribution (C-02, GPU MODE research agent).

THE FIRST HAND-WRITTEN KERNEL IN THIS PROJECT
---------------------------------------------
Every result before this one -- all 2.711x of it -- was won by ARRANGING existing kernels
so vendor fast paths qualify. `bench/kernels/ffn_fused.py` is the first line of Triton
anyone here has written, and it exists because of what generations 11-15 established:
the PyTorch-composition level is exhausted, and finding 22 showed that even unlocking a
compiler optimization we had been denied buys nothing, because the compiler was already
collecting that work elsewhere.

WHAT THE COMPILER STRUCTURALLY CANNOT DO
----------------------------------------
Inductor fuses elementwise into a GEMM epilogue; it does not fuse GEMM into GEMM. Doing so
requires the intermediate held in registers across two mma chains rather than round-tripped
through HBM. `ffn_dim == d_model` on all 14 rows makes both weight matrices 64 KB at
d_model=128 -- inside the measured 99 KB opt-in smem -- so one program can hold both and
never materialize `h`.

MEASURED AT THE OP LEVEL, against what the current candidate path does:

    cfg 7  (D=32)    26.3 us  ->    6.3 us   4.18x
    cfg 1  (D=128)   66.1 us  ->   29.7 us   2.22x
    cfg 13 (D=128)  543.3 us  ->  156.9 us   3.46x
    cfg 6  (D=128) 12408.7 us -> 2703.6 us   4.59x
    cfg 8  (D=1024)  DECLINED by the smem predicate; falls back

AND IT IS MORE ACCURATE, NOT LESS. Against the fp32 reference the fused kernel's max_abs
is 1.13e-04 where the current fp16 path is 2.35e-04, because `h` stays in fp32 registers
instead of being rounded to fp16 between the two GEMMs. This SPENDS no tolerance budget;
it returns some. That matters given L26 -- the margin is thinner than it looks.

THE HONEST CAVEAT, WHICH IS L33
-------------------------------
Those are op-level numbers, and L33 was written yesterday because exactly this kind of
number lied about v15: an isolated probe measures the isolation. The FFN is roughly
15-21% of unfused layer time, so a 4x on the FFN cannot be a 4x on the layer, and the
LayerNorm fusion that swallowed v15's win is still present here. The end-to-end sweep is
the only number that counts, and it is the one recorded in the ledger.

The LayerNorm-fused variant, which would attack that directly, measured max_abs
1.9e-3 against a 2.0e-3 locked budget in the proposal's own probe. It is NOT built here.
Shipping a candidate whose correctness margin is 5% would be trading the one thing this
project refuses to trade.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v8_padfast import build as build_v8
from ..kernels.ffn_fused import fits, fused_ffn


class _Decision:
    """Records why the kernel was or was not used, so an untuned path is never reported
    as a tuned one (the is_tuned discipline from v14_dispatch)."""

    __slots__ = ("used", "reason")

    def __init__(self, used: bool, reason: str):
        self.used, self.reason = used, reason


def build(baseline_cls):
    v8_cls = build_v8(baseline_cls)

    class CandidateV16(v8_cls):
        use_graph = False          # Inductor owns graph capture, as in v9a/v9b
        BLOCK_M = 64
        NUM_WARPS = 8              # measured best at every shape that fits

        def _prime(self, mask):
            super()._prime(mask)
            # Pre-transpose once. nn.Linear stores [out, in]; the kernel contracts over
            # the LEADING axis, so it wants [in, out]. Doing this per call would hand
            # back more than the fusion wins.
            self._ffn_t = []
            for _layer, cached in zip(self.layers, self._cache):
                in_w, in_b, ffn_w, ffn_b = cached[4], cached[5], cached[6], cached[7]
                self._ffn_t.append((in_w.t().contiguous(), in_b,
                                    ffn_w.t().contiguous(), ffn_b))

            props = torch.cuda.get_device_properties(
                self._cache[0][0].device if self._cache else "cuda")
            d = self.layers[0].ffn_in.weight.shape[1]
            f = self.layers[0].ffn_in.weight.shape[0]
            if fits(d, f, 2, self.BLOCK_M, props.shared_memory_per_block_optin):
                self._ffn_decision = _Decision(True, f"fused: d_model={d} ffn_dim={f}")
            else:
                self._ffn_decision = _Decision(
                    False, f"declined: d_model={d} ffn_dim={f} exceeds "
                           f"{props.shared_memory_per_block_optin} B opt-in smem")

        def _core(self, x, mask):
            if not self._ffn_decision.used or not self._nomask:
                # Padded inputs still need the masked_fill the parent applies per layer;
                # rather than duplicate that logic in the kernel, fall back wholesale.
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

                # The whole FFN block -- both GEMMs, GELU, and the fp32 residual add --
                # in one launch, with the intermediate never leaving registers.
                w1t, b1, w2t, b2 = ffn_t
                xn = layer.norm2(x).to(lp).view(-1, d)
                x = fused_ffn(xn, x.view(-1, d), w1t, b1, w2t, b2,
                              self.BLOCK_M, self.NUM_WARPS).view(b, s, d)

            return self.final_norm(x)

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if not hasattr(self, "_compiled"):
                self._compiled = torch.compile(
                    self._core, mode="reduce-overhead", dynamic=False)
            return self._compiled(x, valid_token_mask)

    return CandidateV16
