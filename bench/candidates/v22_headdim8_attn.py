"""Candidate v22 -- a hand-written attention kernel where head_dim is below the mma floor.

Generation 22. Parent: v18_capture_insurance. Branch: cand/g22/headdim8-attn.
Proposals A-01 and B-04, which are the same mechanism found twice from the same source
file -- corroboration of the READING, not of the value (L34).

THE PREMISE THAT IS FALSE, RESTATED SO NOBODY REVIVES IT
--------------------------------------------------------
`matrix.py` used to claim vendor backends "may silently fall back" at head_dim=8. Finding
23 measured it: **flash, mem-efficient, cuDNN and math all accept head_dim=8.** The one
refusal in the whole matrix is cuDNN at head_dim=256 (config 8). Nothing here rests on a
fallback that does not happen.

What IS true is that sm_89's only tensor-core instruction is `mma.sync.m16n8k16`, and
PyTorch's bundled FlashAttention-2 has no head_dim=8 kernel -- `HEADDIM_SWITCH` rounds
anything <= 32 up to kHeadDim = 32. The vendor kernel is not refused, it is mis-tiled: it
contracts over 32 lanes where 8 carry data. Triton's floor is 16, so we pad half as far,
inside the kernel, where padding is free. See `bench/kernels/attn_smallhead.py`.

WHAT IT IS WORTH -- STATED BEFORE THE CONTROLLER SPENDS A SWEEP ON IT
---------------------------------------------------------------------
Op level, isolated `do_bench`, median of 5, strided q/k/v exactly as this lineage
produces them (INDICATIVE, L41 -- a probe may propose, it may never conclude):

    cfg 7   (64, 4,128,8)   SDPA + repack 23.4 us -> ours 16.8 us   1.40x
    cfg 11  (64,16,128,8)   SDPA + repack 59.0 us -> ours 42.2 us   1.40x

That is a genuine 1.4x over a vendor kernel, and it is much less than the 4x the
`HEADDIM_SWITCH` argument predicts, because at these shapes attention is nowhere near
mma-bound: the mma at DP=16 accounts for ~7 us of cfg 11's 42 us. The pad was never the
whole cost, and the proposals' 1.79x / 1.49x "reported potential" is not reachable by
this mechanism.

END TO END IT IS SMALL, AND THE ARITHMETIC IS HERE SO NOBODY HAS TO GUESS (L33).
Attention's in-graph share, from the v13 profiler table in proposal B-04: cfg 7 46.8 us of
124.9 us (37.5%); cfg 11 162.7 us of 362.5 us (44.9%). At 1.40x on that share:

    cfg 7    124.9 -> ~113 us    ~1.10x on the config
    cfg 11   362.5 -> ~321 us    ~1.13x on the config
    13-config geomean                     ~ +1.7%      inside the +/-7% noise floor (L29)
    matrix.weighted_score (cap 3.0)       ~ +0.7%      cfg 11 is already capped at 6.24x,
                                                       so its gain scores exactly ZERO

**So: the mechanism is real and the aggregate value is not measurable by one sweep.** The
defensible claim is per-config and op-level, in the same shape as v17's (finding 25). This
is a report artefact -- "we beat the vendor kernel in the one regime where the hardware
says we should" -- not a frontier move, and it should be judged as one.

WHERE THE REST OF THE HEADROOM ACTUALLY IS
------------------------------------------
Measured while tuning: the same kernel on CONTIGUOUS [Z,H,S,hd] inputs runs at 1.67-1.84x
instead of 1.40x. q/k/v here are views of one [Z,S,3D] GEMM output, so at head_dim=8 every
load is 16 useful bytes inside a 32-byte sector -- half of attention's DRAM traffic is
thrown away by the LAYOUT, not by the mma pad. `.contiguous()` does not collect it (~20 us
of repack to save ~10 us). v20's `qkv_headmajor` kernel does, and its `worth_it()`
currently declines `head_dim < 16`. That recombination is the follow-up worth queuing.

DISPATCH
--------
`attn_smallhead.applicable()` -- head_dim strictly below the `tl.dot` contraction floor
that the Triton backend reports for THIS device, plus divisibility and power-of-two
checks. No config id, no announced shape constant (CLAUDE.md rule 2). Everywhere else,
including every other config in the matrix, `_core` is v18's untouched.

The choice is made once in `_prime`, before compilation and graph capture, so by the time
Dynamo traces `_core` the branch is a Python constant and no guard is added.

NUMERICS
--------
Same online-softmax recurrence as flash, fp16 operands, fp32 accumulators and fp32 softmax
statistics, `head_dim**-0.5` applied in fp32 after the dot. The pad lanes hold literal 0.0
in K and V and are dropped by the output store, so they contribute exactly 0.0 -- the same
exactness argument the causal triangle already uses. This is an exact reordering of the
same sum, not an approximation, and it replaces a kernel that was doing the same
reordering with more padding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v18_capture_insurance import build as build_v18
from ..kernels.attn_smallhead import DEFAULT_TILE, applicable, clamp_tile, smallhead_attention
from ..kernels.ffn_fused import fused_ffn


def build(baseline_cls):
    v18_cls = build_v18(baseline_cls)

    class CandidateV22(v18_cls):
        smallhead_attn_used: bool = False
        smallhead_attn_reason: str = "undecided"
        attn_tile: tuple[int, int, int, int] = DEFAULT_TILE

        def _prime(self, mask):
            super()._prime(mask)
            a = self.layers[0].attention
            cfg = self.config
            ok = applicable(a.head_dim, cfg.seq_len, a.d_model, a.num_heads)
            # The kernel skips the causal triangle by construction, so it is only correct
            # for the causal case -- which every announced row is, but the check is on the
            # config rather than on that fact.
            ok = ok and bool(cfg.causal)
            self.smallhead_attn_used = ok
            self.attn_tile = clamp_tile(DEFAULT_TILE, cfg.seq_len)
            self.smallhead_attn_reason = (
                f"smallhead: head_dim {a.head_dim} below the mma contraction floor"
                if ok else
                f"declined: head_dim {a.head_dim} needs no in-kernel pad, or non-causal")

        def _core(self, x, mask):
            # v18's path, byte for byte, unless our predicate fired AND v8's
            # redundant-mask proof holds. Two guards, both inherited, neither weakened.
            if not self.smallhead_attn_used or not self._fastpath:
                return super()._core(x, mask)

            lp = torch.float16
            zero = self._needs_zeroing
            # v17 gates its FFN megakernel on `_nomask`; keep exactly that condition.
            use_ffn = bool(getattr(self, "fused_ffn_used", False)) and self._nomask
            tile = self.attn_tile

            for layer, cached, ffn_t in zip(self.layers, self._cache, self._ffn_t):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                b, s, d = x.shape

                qkv = F.linear(layer.norm1(x).to(lp), qkv_w, qkv_b)
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)

                # Returns [b, s, d_model] token-major already, so no transpose/reshape.
                ctx = smallhead_attention(q, k, v, tile)

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

    return CandidateV22
