"""Candidate v28 -- attention and the FFN in ONE launch: the whole layer as one kernel.

Generation 28. Parent: v26_causal_correct (the frontier). Branch: cand/g28/layer-megakernel.
From proposals A-03, B-01 and B-07, reconciled below.

WHICH PROPOSAL THIS IS, AND WHY
===============================
Three proposals describe the same mechanism at three scopes:

    B-07   one kernel for the post-attention block          <- ALREADY BUILT (v17, v19)
    this   one kernel for a whole LAYER                     <- attention + FFN, one launch
    A-03   one kernel for a whole sequence-block, all layers, residual never leaving smem
    B-01   the same, framed as a persistent kernel, "35 kernels per forward become 1"

**B-07 is done and its answer is already in the record.** v17 built its FFN half and v19
its normalization half; v19 measured FLAT on config 6 (finding 29). So the remaining
question is not whether fusing pointwise work into a GEMM pays -- it does not, Inductor
already had it -- but whether fusing *across the GEMMs*, which no candidate has done,
pays. That is this candidate.

**A-03 and B-01 are the same idea one level up**, and they are deferred deliberately, not
skipped. B-01's own text calls a single fused layer "B-01's honest first milestone", and
the arithmetic agrees: per-layer fusion removes 4 x (read x + write y) = 5.2 GB of
config 6's 31.5 GB, whole-model fusion removes a further 3.9 GB. Three quarters of the
prize is in the first step, and the first step does not require holding four layers'
weights, four layers of register pressure, or a persistent grid. If a single fused layer
cannot pay, four of them fused together certainly cannot, and this is the cheap way to
learn that.

Where A-03 and B-01 are followed: the dispatch predicate is theirs (a shape compared
against a measured device property), and A-03's warning that "the design must be a plain
persistent kernel with software pipelining, not a producer/consumer warp-specialized one"
is respected -- sm_89 has `mma.sync` and nothing else.

Where they are contradicted: **both budget the working set against the 99 KB opt-in
shared memory. It is the 256 KB REGISTER FILE that binds.** Measured, not argued -- see
`bench/kernels/layer_fused.py`, where the compiler's own `n_spills` for these exact
shapes is tabulated. A-03's "one sequence's residual tile is 32 KB against 99 KB of smem,
room for a K/V staging buffer and a weight tile" is the wrong budget for a Triton kernel,
whose block values live in registers.

WHAT IT DELETES, ON CONFIG 6 (84% of the matrix's wall time)
============================================================
    LN1                     read x32, write xn16          983 MB / layer
    QKV gemm                read xn16, write qkv16*3     1311 MB / layer  <- nothing has
    attention               read qkv16*3, write ctx16    1311 MB / layer     touched these
    out-proj                read ctx16, write o16         655 MB / layer
    add + LN2 + cast                                     1966 MB / layer
    fused_ffn                                            1638 MB / layer
    -------------------------------------------------- -----------------
    frontier                                             7864 MB / layer
    this kernel                                         ~1966 MB / layer

31.5 GB -> ~7.9 GB over four layers; 51.3 ms of roofline -> 12.8 ms. The QKV buffer alone
is 25% of the layer's traffic and is the one piece neither v17 nor v19 could reach.

HONEST EXPECTATION, STATED BEFORE ANY MEASUREMENT (L33)
======================================================
The traffic argument is NOT the argument, because deleting that traffic moves config 6
across the roofline ridge: 1.174 TFLOP against 7.9 GB is 149 FLOP/B, past this card's
measured ridge of 144. **The fused layer is compute bound**, and its outcome is set by
what fraction of the 88.2 BF16-TFLOP/s peak one monolithic CTA reaches:

    15% of peak  ->  88.8 ms  ->  0.73x        30% of peak  ->  44.4 ms  ->  1.46x
    20% of peak  ->  66.6 ms  ->  0.98x        40% of peak  ->  33.3 ms  ->  1.95x

**Break-even is 20%.** Anything outside 0.5x-2.5x on config 6 should be disbelieved before
it is celebrated.

And the same caution the parent's own history demands: the identical traffic argument
predicted -23% for v19 on config 6 and delivered +0.4%, because a megakernel does not
delete work, it moves it into a kernel with worse occupancy. Every tile this kernel can
use runs at ONE block per SM.

THE DILUTED FIGURE, SINCE THE GEOMEAN IS WHAT GETS QUOTED (L33)
==============================================================
The predicate declines configs 2, 3, 8, 13 and 14 outright. On the 13-config geomean a
1.5x on config 6 alone is worth about +3%, which is INSIDE the +/-7% noise floor (L29).
**The defensible claim, whichever way this lands, is per-config and on total wall time,
not on the geomean.** The four-config screen is worse still: it contains configs 2 and 8,
where this candidate is byte-identical to its parent by construction, so a real win on
configs 7 and 10 arrives at the screen geomean halved.

WHAT IT DOES NOT CHANGE
=======================
Non-causal input still delegates to the unmodified baseline (v26, finding 32): the kernel
masks the causal triangle unconditionally and has no non-causal path. A masked input
still takes the parent's path -- the kernel writes whole rows and does not apply the
per-token zeroing the reference does, the same restriction v17's FFN megakernel carries.
Everything the predicate declines is v26's path, unchanged, and `layer_fused_reason`
records which and why, so a fallback is never presented as a tuned path.
"""

from __future__ import annotations

import torch

from .v26_causal_correct import build as build_v26
from ..kernels.layer_fused import (applies, fused_layer, next_pow2, padded_head_dim,
                                   select_tile)


def build(baseline_cls):
    v26_cls = build_v26(baseline_cls)

    class CandidateV28(v26_cls):
        layer_fused_used: bool = False
        layer_fused_reason: str = "undecided"
        layer_tile: tuple[int, int] | None = None

        # -- priming ---------------------------------------------------------------
        def _prime(self, mask):
            super()._prime(mask)
            # The kernel contracts over the LEADING axis, so every nn.Linear weight is
            # transposed once here rather than per call. LayerNorm weights stay fp32:
            # they are [d_model] and cost nothing, and the reference normalizes in fp32.
            self._layer_w = []
            for layer, cached in zip(self.layers, self._cache):
                qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                self._layer_w.append((
                    layer.norm1.weight.float(), layer.norm1.bias.float(),
                    layer.norm2.weight.float(), layer.norm2.bias.float(),
                    qkv_w.t().contiguous(), qkv_b,
                    out_w.t().contiguous(), out_b,
                    in_w.t().contiguous(), in_b,
                    ffn_w.t().contiguous(), ffn_b,
                ))

        # -- the decision, made ONCE and before anything traces it ------------------
        def _decide_layer(self, x):
            """Resolved at prime time so the tile, the grid and every derived constant
            are Python integers by the time `torch.compile` sees `_core`.

            Resolving them inside the traced region is the launch-wrapper defect a
            sibling candidate paid -18.9% for: a `bit_length()` or a ceil-divide on a
            shape is enough for Dynamo to break the graph and drop the frame to eager.
            """
            a = self.layers[0].attention
            b, s, d = x.shape
            if not getattr(self.config, "causal", True):
                # The kernel masks the causal triangle unconditionally. v26 delegates
                # non-causal input to the unmodified baseline before `_core` is ever
                # reached; declining here as well is belt and braces (finding 32).
                self.layer_fused_used = False
                self.layer_fused_reason = "declined: non-causal input"
                return
            f = self.layers[0].ffn_in.weight.shape[0]
            props = torch.cuda.get_device_properties(x.device)
            ok, why = applies(s, d, f, a.head_dim, a.num_heads, b, props)
            if not ok:
                self.layer_fused_used = False
                self.layer_fused_reason = why
                return
            try:
                tile, how = select_tile(s, d, f, a.head_dim, a.num_heads, b, x.device)
            except Exception as exc:                   # never fail closed on a tuner
                self.layer_fused_used = False
                self.layer_fused_reason = f"declined: tile selection failed ({exc})"
                return
            self.layer_tile = tile
            self._layer_grid = -(-s // tile[0])
            self._layer_bn = next_pow2(s)
            self._layer_dp = padded_head_dim(a.head_dim)
            self._layer_eps = float(self.layers[0].norm1.eps)
            self.layer_fused_used = True
            self.layer_fused_reason = f"{why}; {how}"

        # -- the computation -------------------------------------------------------
        def _core(self, x, mask):
            # `_nomask` is required for the same reason v17's FFN megakernel requires it:
            # the kernel writes whole rows and does not apply the reference's per-token
            # zeroing of invalid positions. A padded input takes the parent's path.
            if not self.layer_fused_used or not self._nomask:
                return super()._core(x, mask)

            a = self.layers[0].attention
            bm, warps = self.layer_tile
            for lw in self._layer_w:
                x = fused_layer(x, *lw, a.num_heads, a.head_dim, a.scale,
                                self._layer_eps, bm, warps,
                                self._layer_grid, self._layer_bn, self._layer_dp)
            return self.final_norm(x)

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if self.layer_fused_reason == "undecided":
                self._decide_layer(x)
            return super().forward(x, valid_token_mask)

    return CandidateV28
