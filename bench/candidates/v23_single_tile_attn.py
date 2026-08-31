"""Candidate v23 -- hand-written single-tile causal attention where the score matrix fits.

Generation 23. Parent: v18_capture_insurance. Branch: cand/g23/single-block-attn.
From proposals C-01 (research agent C) and D-04 (research agent D), which are the same
idea found twice from disjoint sources. Where they disagree, see RECONCILIATION below.

THE MECHANISM
-------------
Every candidate from v2 to v18 calls `F.scaled_dot_product_attention`, which dispatches
to FlashAttention. Flash exists to avoid materializing an S x S score matrix that does not
fit on chip. Ten of the fourteen announced rows have `seq_len == 128` and one has 32, so
the whole score matrix for one (batch, head) is at most 128x128 -- 64 KB of fp32, which
fits in the register file of one thread block on this card. In that regime flash's
machinery (a running max, a running sum, an accumulator rescale per K block) is
bookkeeping for a loop that runs exactly once.

`bench/kernels/attn_single_tile.py` replaces it with one tile: load Q, K, V for one
(batch, head, query-block), one `tl.dot`, one `tl.where` for the causal triangle, ONE
ordinary softmax, one `tl.dot` for `P V`, store. It also reads Q/K/V straight out of the
fused `[B, S, 3*d_model]` projection buffer and writes `[B, S, d_model]` head-major, so
the `.split`, the three `.transpose(1, 2)` views and -- the part that actually costs a
kernel -- the `transpose(1, 2).reshape` repack all disappear with it.

RECONCILIATION OF C-01 AND D-04
-------------------------------
C-01 specifies **one program per (batch, head)**: `block_m == seq_len`, the literal
"single block". D-04 specifies a single *tile of K/V* without fixing the query block.
Swept on this card, `block_m = 64` beats `block_m = 128` on every shape big enough for
the timer to resolve (12.3 vs 13.3 us at head_dim 8; 20.5 vs 21.5 us at head_dim 32).
**We implement D-04's shape.** C-01's stricter claim is measurably wrong by ~5%: 128
query rows put the fp32 score tile at 64 KB, halving how many blocks stay resident per SM
for no reduction in work, and the K/V tiles are cheap enough to re-read once per block.
The rest of the two proposals -- single-pass softmax, in-kernel padding of head_dim to
the MMA width, exactness under causality -- is common to both and is what this builds.

WHERE IT DECLINES, AND WHY THAT IS THE INTERESTING PART
-------------------------------------------------------
The op-level sweep found two shapes where the kernel is legal, correct and LOSES:
head_dim 128 (0.94x) and head_dim 256 (0.84x). The mechanism is not arithmetic. This
kernel has NO LOOP, so there is nothing to software-pipeline: each program is one long
dependent chain and its only latency hiding is other resident blocks on the same SM. The
fp32 score tile plus the Q/K/V operands live in registers, so the register working set
caps residency directly, and the measured sign flip is at ~4 resident blocks per SM. The
predicate is that budget, evaluated against `regs_per_multiprocessor` and
`max_threads_per_multi_processor` read off the device -- shapes and measured properties
only, no config ids (CLAUDE.md rule 2). It declines head_dim 128 and 256, seq_len 1024
and seq_len 100000, and falls back to v18's SDPA path there, unchanged.

THE TILE IS AUTOTUNED, NOT ARGUED
---------------------------------
A mechanism argument cannot pick a tile size, and no formula fits this card: 64x4 warps
wins at head_dim 32 while 32 warps-8 wins at head_dim 64 at an identical register cost.
So the candidate times its own viable tiles once, at prime time, before compilation and
graph capture, on a probe batch capped from the measured SM count. A derived tile is the
fallback if that fails.

PRECISION
---------
The softmax accumulates in fp32 throughout (finding 08). With one K block, the online
softmax and the textbook softmax are the same arithmetic with the rescalings deleted, so
this is numerically no worse than the path it replaces; the causal mask is exact and
padding head_dim to 16 with zeros contributes exactly zero to the contraction. Margin is
a first-class metric here (L26): the tests record `max_abs` against the fp32 reference
rather than only pass/fail.

WHAT TO EXPECT END TO END, STATED BEFORE THE SCREEN (L33)
---------------------------------------------------------
Attention is 18-46% of layer time depending on config, so the op-level 1.1x-2.4x dilutes
hard. Taking the shares measured by agent D and this sweep's op speedups, the ceilings
are roughly 1.20x on config 7, 1.25x on config 11, 1.15x on config 12, 1.09x on config 1
and ~1.03x on config 10 -- and configs 8, 9, 13 and 14 get exactly nothing, by design.
Anything above those numbers should be disbelieved before it is celebrated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v18_capture_insurance import build as build_v18
from ..kernels.attn_single_tile import applies, autotune_tile, single_tile_attention
from ..kernels.ffn_fused import fused_ffn


def build(baseline_cls):
    v18_cls = build_v18(baseline_cls)

    class CandidateV23(v18_cls):
        attn_used: bool = False
        attn_reason: str = "undecided"
        attn_tile: tuple[int, int, int] | None = None

        # WHICH INSTRUMENT THE TILE SWEEP RANKS ON. `None` is `attn_single_tile`'s own
        # default, `flushed_time` -- what this class has shipped since generation 23.
        #
        # AN EXTENSION POINT, NOT A CONVENIENCE, and here for the same reason
        # `_attention` below is: generation 42's claim is that this sweep's timer picks
        # the wrong tile, and the only protocol that can measure that claim is two arms
        # resident in ONE process (finding 50 measured two candidates reading 46% apart
        # on config 3 when the arms were separated by a process boundary). A descendant
        # that flips this attribute therefore differs from its parent by exactly one
        # value, in one place, with the parent's path untouched -- which is what makes
        # the A/B attributable and supplies its byte-identical control configs.
        attn_tile_timer = None

        # HOW MANY TIMES THE SWEEP MUST AGREE WITH ITSELF BEFORE IT DISPLACES ANYTHING.
        # `1` is `autotune_tile`'s own default and is what this class has shipped since
        # generation 23. The second extension point, for the same reason as the first:
        # generation 43's claim is that a SHARPER timer needs its answer replicated
        # before it is acted on, and measuring that claim needs a parent arm that is
        # byte-identical to the parent. One value, in one place, both paths untouched.
        attn_tile_replicates = 1

        def _decide_attn(self, x):
            """Decided ONCE, before compilation and graph capture, so the tile is a
            Python constant by the time anything traces it."""
            a = self.layers[0].attention
            b, s, _ = x.shape
            props = torch.cuda.get_device_properties(x.device)
            ok, why = applies(s, a.head_dim, props)
            if not ok:
                self.attn_used = False
                self.attn_reason = why
                return
            try:
                tile, how = autotune_tile(s, a.head_dim, a.num_heads, b, x.device,
                                          timer=self.attn_tile_timer,
                                          replicates=self.attn_tile_replicates)
            except Exception as exc:                   # never fail closed on a tuner
                self.attn_used = False
                self.attn_reason = f"declined: tile selection failed ({exc})"
                return
            self.attn_tile = tile
            self.attn_used = True
            self.attn_reason = f"{why}; {how}"

        def _attention(self, qkv, a, b, s):
            """Compute attention over the fused `[B, S, 3*d_model]` buffer.

            AN EXTENSION POINT, NOT A CONVENIENCE -- the discipline v33's
            `_settle_slice_decisions` docstring states. This exact if/else was written
            out inline in v23's, v34's and v36's `_core` (four copies, byte-identical),
            so a descendant that wanted to compute attention differently had to override
            three long `_core` bodies and keep them in sync with their originals
            forever. That is the [L14] shape: a mechanical edit across duplicated code
            that produces something valid and wrong.

            The `else` branch is v8's path verbatim: fp16, no `attn_mask`, so flash
            qualifies and the key mask is provably redundant under causality (finding
            11). The `transpose(1, 2).reshape` is a real copy, and it is the second half
            of what `single_tile_attention` deletes (finding 31).
            """
            if self.attn_used:
                bm, warps, stages = self.attn_tile
                # No split, no view, no transpose, no repack: the kernel indexes the
                # fused buffer and writes the layout out_proj already wants.
                return single_tile_attention(qkv, a.num_heads, a.head_dim, a.scale,
                                             bm, warps, stages)
            q, k, v = qkv.split(a.d_model, dim=-1)
            q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
            k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
            v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
            return F.scaled_dot_product_attention(
                q, k, v, is_causal=True).transpose(1, 2).reshape(b, s, a.d_model)

        def _core(self, x, mask):
            # `_fastpath` is v8's proof that the key mask is redundant for a right-padded
            # causal input. Without it the mask must be applied inside attention, which
            # this kernel deliberately does not do.
            if not self.attn_used or not self._fastpath:
                return super()._core(x, mask)

            lp = torch.float16
            zero = self._needs_zeroing
            use_ffn = self.fused_ffn_used and self._nomask       # v17's own condition

            for layer, cached, ffn_t in zip(self.layers, self._cache, self._ffn_t):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                b, s, d = x.shape

                qkv = F.linear(layer.norm1(x).to(lp), qkv_w, qkv_b)
                ctx = self._attention(qkv, a, b, s)
                o = F.linear(ctx, out_w, out_b).float()
                if zero:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o

                if use_ffn:
                    w1t, b1, w2t, b2 = ffn_t
                    xn = layer.norm2(x).to(lp).view(-1, d)
                    x = fused_ffn(xn, x.view(-1, d), w1t, b1, w2t, b2,
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
            if self.attn_reason == "undecided":
                self._decide_attn(x)
            return super().forward(x, valid_token_mask)

    return CandidateV23
