"""Candidate v25 -- fp16 accumulation in the MMA, and the boundary it cannot cross.

Generation 25. Parent: v18_capture_insurance. Branch: cand/g25/fp16-accum.
Reconciles three independently-authored proposals for the same mechanism (C-03, A-05,
D-05). Per L34 that convergence corroborates the READING, not the value: all three were
reading the same sm_89 documentation, and none had run the model.

THE READING IS CORRECT
----------------------
Consumer Ada runs tensor-core FP16-with-FP32-accumulate at half rate. Confirmed here from
the generated PTX rather than the datasheet -- `tl.dot(out_dtype=tl.float16)` emits

    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

where `out_dtype=tl.float32` emits `...f32.f16.f16.f32`, and in a loop of 1024 mma
instructions with everything else held the f16 form measures **1.569x** faster
(36.6 us -> 23.4 us). The instruction really is ~1.6x. Every proposal's premise holds.

AND THE MECHANISM IS STILL WORTHLESS HERE, FOR TWO INDEPENDENT REASONS
---------------------------------------------------------------------
**Reason 1 -- we are nowhere near that instruction.** The fused FFN block moves
`d_model*(2+4+4)` bytes per token and does `4*d_model*ffn_dim` FLOP, an arithmetic
intensity of **51.2 FLOP/B** at d_model == ffn_dim == 128 against this device's MEASURED
ridge point of **143.7 FLOP/B**. The kernel is memory-bound by ~2.8x; the tensor cores
are already idle waiting on HBM, and making the instruction 1.6x faster shortens
something that is not on the critical path. Measured, per site, against the fp32
reference formulation:

    shape                 arm        us    speedup    max_abs   failing elements
    D=128 tok=8192     f32,f32     29.6     1.000x   8.18e-04          0
    D=128 tok=8192     f16,f32     27.9     1.060x   3.24e-03         11
    D=128 tok=8192     f32,f16     29.6     1.000x   5.18e-03         56
    D=128 tok=8192     f16,f16     28.9     1.026x   6.22e-03        272
    D=128 tok=1.28M    f32,f32   2749.2     1.000x   1.17e-03          0
    D=128 tok=1.28M    f16,f32   2748.1     1.000x   3.88e-03       2327
    D=128 tok=1.28M    f32,f16   2766.3     0.994x   7.03e-03       9343
    D=128 tok=1.28M    f16,f16   2742.1     1.003x   8.39e-03      38314

On **config 6 -- the largest shape in the matrix and the one this kernel exists for --
the speedup is 1.000x.** Not a small win inside the noise floor: no win. The 1.6x
instruction is entirely absorbed by the memory wall.

**Reason 2 -- the error is over budget everywhere it could pay.** Finding 08 established
the distinction that made this worth probing: the residual ACCUMULATES across layers, an
elementwise op does not, so an fp16 accumulator inside ONE GEMM over K=128 is a different
risk from an fp16 residual across 4 layers. The distinction is real and it is not enough.
A single GEMM's fp16 accumulator lands at 3.2e-3 to 8.4e-3 against a 2.0e-3 budget --
squarely inside finding 08's 3.3x-5.3x-over band, from one site in one layer.

THE SCISSORS, WHICH IS THE DURABLE PART
---------------------------------------
The two conditions are monotone in OPPOSITE directions in the contraction depth K, and in
this architecture K == d_model == ffn_dim, so one parameter drives both:

  * fp16-accumulate is FAST only above the ridge point. Intensity is linear in d_model,
    so `mma_bound` needs **d_model >= 359** on this device.
  * fp16-accumulate is ACCURATE only while `eps_fp16 * sqrt(K) <= atol`, which at the
    locked 2e-3 and unit output magnitude needs **K <= 16**.

There is no shape where both hold -- the regions are disjoint by a factor of ~22, and
they diverge further on any faster card, because a higher peak-FLOPs-to-bandwidth ratio
pushes the ridge point UP while fp16's mantissa stays 11 bits. `ffn_accum.no_shape_
satisfies_both()` computes the gap from measured device properties rather than asserting
it, and a test pins it.

This is not "needs tuning". It is the same shape of result as finding 08: a mechanism
whose premise is correct and whose window does not exist.

WHAT THIS CANDIDATE THEREFORE IS
--------------------------------
v18 exactly, plus the parameterized kernel and the two predicates, with the accumulator
width chosen by `_decide_accum` from shapes and measured device properties. The predicate
declines on every shape, so **v25 is numerically identical to v18 as shipped** and must
not be expected to measure differently. `accum_mode` and `accum_reason` are reported so
the decline is observable (L36/L38: a guard that cannot be seen to fire cannot be
trusted), and `RATCHET_FORCE_ACCUM` lets the tests force the fp16 arms so the boundary
stays measured rather than remembered.

L33, stated before anyone asks: the GEMMs this candidate can reach are the two inside the
fused FFN block, which runs on 3 of 14 configs (those where v17's amortization predicate
fires) and is ~15-21% of unfused layer time. Even a free 1.6x on 100% of that would be a
few percent. It is 1.000x on the config that matters.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .v18_capture_insurance import build as build_v18
from ..kernels.ffn_accum import (device_roofline, fused_ffn_accum, mma_bound,
                                 accumulator_affordable)
from ..kernels.ffn_fused import fits

# The locked tolerance, passed to the predicate rather than re-derived inside it.
LOCKED_ATOL = 2e-3
# Post-LayerNorm activations are unit-variance by construction, so the FFN output the
# accumulator must resolve is O(1). Not fitted -- it is what LayerNorm guarantees.
OUTPUT_MAGNITUDE = 1.0


def _forced_mode() -> tuple[int, int] | None:
    """Test/probe escape hatch. `RATCHET_FORCE_ACCUM=a,b` forces the accumulator arms so
    the falsifier can measure the path the predicate refuses to select. Never read on the
    shipped path -- absent means 'let the predicate decide'."""
    raw = os.environ.get("RATCHET_FORCE_ACCUM")
    if not raw:
        return None
    a, _, b = raw.partition(",")
    return int(a), int(b or a)


def build(baseline_cls):
    v18_cls = build_v18(baseline_cls)

    class CandidateV25(v18_cls):
        accum_mode: tuple[int, int] = (0, 0)     # (site A, site B); 0 = fp32, 1 = fp16
        accum_reason: str = "undecided"

        def _decide_accum(self, x):
            """Shapes and MEASURED device properties only (rule 2). Decided once, before
            compilation and capture, so the choice is a Python constant by trace time."""
            forced = _forced_mode()
            if forced is not None:
                self.accum_mode = forced
                self.accum_reason = f"forced by RATCHET_FORCE_ACCUM to {forced}"
                return

            d = x.shape[-1]
            f = self.layers[0].ffn_in.weight.shape[0]
            peak, bw = device_roofline(x.device)
            fast = mma_bound(d, f, peak, bw)
            accurate = accumulator_affordable(d, LOCKED_ATOL, OUTPUT_MAGNITUDE)

            if fast and accurate:
                self.accum_mode = (1, 1)
                self.accum_reason = (
                    f"fp16 accumulate: d_model={d} is compute-bound AND K={d} fits "
                    f"{LOCKED_ATOL:g}")
            elif not fast:
                self.accum_mode = (0, 0)
                self.accum_reason = (
                    f"declined: d_model={d} ffn_dim={f} is memory-bound "
                    f"({4.0*d*f/(d*10):.1f} FLOP/B below the {peak/bw:.1f} FLOP/B ridge), "
                    f"so a faster MMA is off the critical path")
            else:
                self.accum_mode = (0, 0)
                self.accum_reason = (
                    f"declined: K={d} exceeds what an fp16 accumulator carries inside "
                    f"{LOCKED_ATOL:g}")

        def _decide_ffn(self, x):
            """Forcing an accumulator arm must also force the kernel that CONTAINS it.

            Without this the falsifier is vacuous: v17's amortization predicate declines
            the fused kernel at small token counts, `_core` falls through to the parent,
            and every forced arm measures identically to the shipped one -- a green
            result produced by the mechanism never running (L36). Only the token-count
            gate is bypassed; the shared-memory gate is physical and still binds.
            """
            super()._decide_ffn(x)
            if _forced_mode() is None or self.fused_ffn_used:
                return
            d = x.shape[-1]
            f = self.layers[0].ffn_in.weight.shape[0]
            props = torch.cuda.get_device_properties(x.device)
            if fits(d, f, 2, self.BLOCK_M, props.shared_memory_per_block_optin):
                self.fused_ffn_used = True
                self.fused_ffn_reason = "forced by RATCHET_FORCE_ACCUM (amortization gate bypassed)"

        def _core(self, x, mask):
            if not self.fused_ffn_used or not self._nomask or self.accum_mode == (0, 0):
                # Nothing to change: either the fused kernel is not in play, or the
                # predicate chose fp32 and v18's own kernel already is fp32. Falling
                # through keeps the frontier's exact code path rather than a copy of it.
                return super()._core(x, mask)

            lp = torch.float16
            acc_a, acc_b = self.accum_mode
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
                x = fused_ffn_accum(xn, x.view(-1, d), w1t, b1, w2t, b2,
                                    self.BLOCK_M, self.NUM_WARPS,
                                    acc_a=acc_a, acc_b=acc_b).view(b, s, d)

            return self.final_norm(x)

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if self.accum_reason == "undecided":
                self._decide_accum(x)
            return super().forward(x, valid_token_mask)

    return CandidateV25
