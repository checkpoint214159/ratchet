"""Candidate v41 -- the attention chooser is allowed to STEP ASIDE for the vendor.

Generation 41. Parent: `v40_looped_attn` (`ac0ef1e`). Branch: `cand/g41/attn-audit`.
Answers the OPEN QUESTION written into `bench/kernels/attn_single_tile.py` at generation
23 and re-opened as a proposal by finding 50.

WHAT IT ADDS
------------
One call, after v40's decision has been taken: where the plan is still "run our loop-free
single-tile kernel", `attn_choice.autotune_vendor` times **that exact tile** against
`sdpa+repack`, hot, two arms with one trial budget each, and hands the shape to the vendor
if the vendor clears v23's inherited `DECISIVE` 10% margin. Every other outcome leaves the
plan exactly as v40 left it, so a shape the vendor does not win is byte-identical to the
parent by construction -- the same structural guarantee `autotune_looped` has, for the
same reason (an attributable A/B and a byte-identical control arm).

WHY IT IS NOT A DEFEAT
-----------------------
`attn_single_tile.pays()` is a RESIDENCY argument: it says whether a loop-free kernel has
enough co-resident blocks to hide its memory latency. That is a statement about our
kernel, and it is silent about how fast the vendor's is. The two were never the same
question, and the file said so at generation 23:

    "The screen measured config 10 (head_dim 64) at -7.1% end to end -- the marginal
     case, sitting at exactly MIN_RESIDENT_BLOCKS... It is deliberately NOT implemented
     until a full sweep confirms the regression is real."

A kernel that knows when to step aside is worth more than one that always fires.
"""

from __future__ import annotations

import torch

from .v40_looped_attn import build as build_v40
from ..kernels import attn_choice


def build(baseline_cls):
    v40_cls = build_v40(baseline_cls)

    class CandidateV41(v40_cls):
        # Why the vendor was, or was not, given the shape. Reported so [L36] can be
        # checked from outside: a candidate whose mechanism never engages is its parent
        # with extra build time, and this one is designed to engage on very few shapes.
        attn_vendor_reason: str = "not asked"

        def _decide_attn(self, x):
            """v40's decision, then ONE more question of it.

            Order matters and is deliberate. The looped form is asked first and already
            carries `sdpa+repack` as a hard floor inside `autotune_looped`, so a shape it
            wins has already beaten the vendor and is not re-litigated here. Only the
            fallback path -- v23's `autotune_tile`, which has never had a vendor floor at
            all -- reaches this check.
            """
            super()._decide_attn(x)
            if not self.attn_used or self.attn_form != "single_tile":
                self.attn_vendor_reason = (
                    f"not asked: plan is {self.attn_form}")
                return
            a = self.layers[0].attention
            b, s, _ = x.shape
            try:
                why = attn_choice.autotune_vendor(
                    s, a.head_dim, a.num_heads, b, self.attn_tile, x.device)
            except Exception as exc:          # never fail closed on a tuner (v23's rule)
                self.attn_vendor_reason = f"kept the kernel: {exc}"
                return
            # `attn_used = False` is v23's own switch, and every descendant's `_core`
            # routes it through `_attention`'s else-branch -- `qkv.split` -> three
            # views/transposes -> SDPA(is_causal=True) -> `transpose(1,2).reshape`. That
            # is v8's path verbatim: fp16, no `attn_mask`, so flash qualifies and the key
            # mask is provably redundant under causality (finding 11). Note v23's own
            # `_core` bails to the parent when `attn_used` is False, but v36's -- which is
            # what this lineage actually runs -- does not: it gates on `_fastpath` only
            # and calls `self._attention`, so the fused GEMMs and the megakernel survive
            # the switch. Finding 50's config-9 census is the standing evidence that this
            # path is live and fast on shapes the kernel declines.
            self.attn_used = False
            self.attn_form = "sdpa"
            self.attn_vendor_reason = why
            self.attn_reason = f"{self.attn_reason}; STEPPED ASIDE: {why}"

        def _invalidate_shape_state(self, mask=None):
            """Reset the vendor verdict with everything else latched to an input shape.

            `attn_vendor_reason` is introduced by THIS class, so v37's `SHAPE_LATCHED` --
            derived at v37's build time over the classes below it -- cannot name it.
            [L50]: the defect this lineage keeps rediscovering is a fix that makes a
            second, dormant defect reachable for the first time.
            """
            super()._invalidate_shape_state(mask)
            self.attn_vendor_reason = type(self).attn_vendor_reason

    return CandidateV41
