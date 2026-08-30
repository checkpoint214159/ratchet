"""Candidate v27 -- the Q/K/V projection fused INTO the attention kernel.

Generation 27. Parent: v26_causal_correct (the frontier). Branch: cand/g27/qkv-fused-attn.

WHAT IT CHANGES, AND WHAT IT DOES NOT
-------------------------------------
v23 gave the attention kernel a fused `[B, S, 3*d_model]` buffer to read from, and v26
made it honour `config.causal`. That buffer is still produced by a separate cuBLAS GEMM,
which the fresh profile of config 6 -- 84% of matrix wall time -- puts at **15.4%**, sitting
immediately before our own kernel at 15.5%. It writes 983 MB per layer and the next
kernel reads it straight back.

v27 replaces the pair

    qkv = F.linear(norm1(x).to(fp16), Wqkv, bqkv)      # cuBLAS
    ctx = single_tile_attention(qkv, ...)              # v23

with one launch that loads the normalized input tile, projects its own head's Q, K and V
in registers, and attends. `bench/kernels/attn_qkv_fused.py` holds the kernel and the
predicates. Nothing else in the stack moves: the fp32 residual, the fused FFN megakernel,
the graph capture, the padding proof and v26's causal delegation are all inherited
unchanged, and where the predicate declines, the parent's path runs untouched.

THE 96 KB QUESTION, ANSWERED
----------------------------
The proposal framed this as "the fused QKV weight is 128 x 384 fp16 = 96 KB against the
measured 99 KB opt-in shared memory -- does it fit?". It fits, with 3 KB to spare, and
**it is the wrong question**: a program that owns one head never needs the whole weight.
Its slice is `3 * d_model * pad16(head_dim) * 2` bytes -- 24 KB at head_dim 32, 12 KB at
head_dim 8, 48 KB at head_dim 64. The 96 KB figure is reached only at head_dim 128
(config 9), which the predicate declines anyway.

The binding constraint is the other operand: the **input tile**, `[next_pow2(S), d_model]`
fp16, which all three projections read. 32 KB at d_model 128, 256 KB at d_model 1024
(config 8), 280 KB at seq_len 1024 (config 13). That is what refuses the wide model and
the long sequence, and it is a shape fact evaluated against
`shared_memory_per_block_optin`, not a config id.

WHERE IT DECLINES -- INCLUDING ONE THE MECHANISM ARGUMENT DID NOT PREDICT
-------------------------------------------------------------------------
Each program owns one head and needs the FULL model width to project it, so `heads`
programs each re-read the same input tile. That is the fusion's structural cost and it is
invisible until counted. At config 11 (16 heads, d_model 128) the fused kernel reads
512 KB per sequence per layer against the 320 KB the GEMM-and-attention pair moves -- the
fusion moves MORE bytes than it deletes. The op-level probe measured 0.822x there, and
`moves_fewer_bytes` declines it on the byte count, with no fitted constant: it reduces to
roughly `heads <= 7` at this width, derived rather than tuned.

Declined: configs 8, 9, 11, 13, 14. Applied: 1-7, 10, 12.

WHAT TO EXPECT END TO END, STATED BEFORE THE SCREEN (L33)
---------------------------------------------------------
The op-level probe (INDICATIVE ONLY, L41 -- GPU lock held, min of 5 x do_bench, nothing
recorded to the ledger) measured the fused kernel against `F.linear` + v23's kernel:

    config 6's shape at B=800    1.293x        config 10   1.154x
    config 7                     1.500x        config 12   1.300x
    config 1  (B=64)             0.962x        config 11   0.822x -> now declined

and a batch sweep at config 6's shape shows the ratio is stable at 1.25-1.34x for
B >= 512 and pure noise below it (1.53x at B=16, 0.92x at B=256, on 10-90 us kernels
inside a +/-7% floor). **Do not read the middle of that sweep as signal.**

Diluting honestly: the GEMM and the attention kernel are 30.9% of config 6 between them,
so a 1.29x on that pair is `1 - 0.309*(1 - 1/1.293)` = **a 7.0% ceiling on config 6**, and
config 6 is 84% of matrix wall time but only 1 of 14 equally weighted rows in the geomean.
If configs 1-7, 10 and 12 each gained the full 7% the geomean would move
`1.07^(9/14) = +4.5%` -- INSIDE the noise floor. The defensible claim is per-config, on
the matrix's largest shape, and the geomean should be expected to look flat.

WHY IT IS NOT SIMPLY 15.4% OF FREE TIME
---------------------------------------
The GEMM being 15.4% does not mean 15.4% is available. Measured, that GEMM moves 5.24 GB
in 8.8 ms = 595 GB/s, which is 97% of this card's 613.7 GB/s roofline: it is at the wall,
not badly written. Fusing does not delete its FLOPs, it moves them into our kernel, and
the stage stops being bandwidth-bound (96 FLOP/B, below the 144 ridge) and becomes
compute-bound (~328 FLOP/B, a 9.5 ms floor against 17.7 ms today). The fused kernel wins
only if it beats the 47.5% of peak the pair currently averages. That was a coin flip when
it was written down; the op probe says it lands at about 1.29x on the shape that matters.

PRECISION
---------
The projection accumulates in fp32, adds the bias in fp32, and rounds to fp16 once --
structurally identical to `F.linear` on fp16 operands with an fp32-accumulating cuBLAS
epilogue, so no rounding step is added or removed on that path. The softmax stays fp32
throughout (finding 08). Padded head_dim lanes load exactly zero from both weight and
bias, so they contribute exactly zero to the contraction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v26_causal_correct import build as build_v26
from ..kernels.attn_qkv_fused import applies, autotune_tile, fused_qkv_attention
from ..kernels.ffn_fused import fused_ffn


def build(baseline_cls):
    v26_cls = build_v26(baseline_cls)

    class CandidateV27(v26_cls):
        qkv_fused_used: bool = False
        qkv_fused_reason: str = "undecided"
        qkv_fused_tile: tuple[int, int, int] | None = None

        def _prime(self, mask):
            super()._prime(mask)
            # The kernel contracts over the leading axis, so the [3*d_model, d_model]
            # nn.Linear layout has to be transposed. Once, here -- never in the hot loop
            # and never inside a traced region.
            self._qkv_t = [c[0].t().contiguous() for c in self._cache]

        def _decide_qkv(self, x):
            """Decided ONCE, before compilation and graph capture, so the tile is a Python
            constant by the time anything traces it.

            A sibling candidate measured -18.9% because plan resolution ran inside
            Dynamo's traced region, dropping the frame to eager -- and the CUDA graph
            faithfully captured the eager ops. Every lookup this path needs is resolved
            here and stored as a plain attribute.
            """
            if not getattr(self.config, "causal", True):
                self.qkv_fused_used = False
                self.qkv_fused_reason = "declined: non-causal input goes to the baseline"
                return
            a = self.layers[0].attention
            b, s, d = x.shape
            props = torch.cuda.get_device_properties(x.device)
            ok, why = applies(s, d, a.head_dim, a.num_heads, props)
            if not ok:
                self.qkv_fused_used = False
                self.qkv_fused_reason = why
                return
            try:
                tile, how = autotune_tile(s, d, a.head_dim, a.num_heads, b, x.device)
            except Exception as exc:                   # never fail closed on a tuner
                self.qkv_fused_used = False
                self.qkv_fused_reason = f"declined: tile selection failed ({exc})"
                return
            self.qkv_fused_tile = tile
            self.qkv_fused_used = True
            self.qkv_fused_reason = f"{why}; {how}"

        def _core(self, x, mask):
            # `_fastpath` is v8's proof that a right-padded causal key mask is redundant.
            # Without it the mask must be applied inside attention, which this kernel
            # deliberately does not do -- same condition v23 imposes on its own kernel.
            if not self.qkv_fused_used or not self._fastpath:
                return super()._core(x, mask)

            lp = torch.float16
            zero = self._needs_zeroing
            use_ffn = self.fused_ffn_used and self._nomask       # v17's own condition
            bm, warps, stages = self.qkv_fused_tile

            for layer, cached, ffn_t, w_t in zip(self.layers, self._cache, self._ffn_t,
                                                 self._qkv_t):
                a = layer.attention
                qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached[1:]
                b, s, d = x.shape

                # ONE launch: normalize, then project AND attend without the
                # [B, S, 3*d_model] buffer ever existing.
                xn = layer.norm1(x).to(lp)
                ctx = fused_qkv_attention(xn, w_t, qkv_b, a.num_heads, a.head_dim,
                                          a.scale, bm, warps, stages)
                o = F.linear(ctx, out_w, out_b).float()
                if zero:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o

                if use_ffn:
                    w1t, b1, w2t, b2 = ffn_t
                    xn2 = layer.norm2(x).to(lp).view(-1, d)
                    x = fused_ffn(xn2, x.view(-1, d), w1t, b1, w2t, b2,
                                  self.BLOCK_M, self.NUM_WARPS).view(b, s, d)
                else:
                    h = F.linear(layer.norm2(x).to(lp), in_w, in_b)
                    x = x + F.linear(F.gelu(h, approximate="none"),
                                     ffn_w, ffn_b).float()
                    if zero:
                        x = x.masked_fill(~mask[..., None], 0)

            x = self.final_norm(x)
            return x.masked_fill(~mask[..., None], 0) if zero else x

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if self.qkv_fused_reason == "undecided":
                self._decide_qkv(x)
            # v26's forward sends a non-causal input to the unmodified baseline, then
            # v23's decides its own fallback tile. Both still apply.
            return super().forward(x, valid_token_mask)

    return CandidateV27
