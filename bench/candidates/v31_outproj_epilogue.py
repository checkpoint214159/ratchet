"""Candidate v31 -- the attention out-projection absorbed into the attention kernel's
own epilogue, with the fp32 widen and the fp32 residual add.

Generation 31. Parent: v26_causal_correct (the frontier). Branch: cand/g31/outproj-epilogue.
Ported from `cand/g24/outproj-prologue`, which built the same fusion against the OLD
frontier (v18, SDPA-based) as a standalone GEMM. This does not repeat that work; it moves
the fusion one kernel further in, which is only possible because v23 replaced SDPA with a
kernel we own.

WHAT THE FRONTIER DOES AND WHAT THIS DOES INSTEAD
--------------------------------------------------
v23/v26's attention half of a layer:

    ctx = single_tile_attention(qkv, ...)          # our Triton kernel, writes fp16 [B,S,D]
    o   = F.linear(ctx, out_w, out_b).float()      # cuBLAS GEMM, writes fp16 [B,S,D]
    x   = x + o                                    # Inductor's widen-and-add pointwise

Three kernels, and the middle one is reading back a tile that was in the first kernel's
REGISTERS a microsecond earlier. `bench/kernels/attn_outproj.py` finishes the projection
there: one program owns every head of a query block, projects each head's context through
its own `[head_dim, D]` slice of the weight, accumulates in one fp32 `[BM, D]` register
block, adds the bias, adds the fp32 residual, and stores once.

    v23    write ctx 2D | read ctx 2D + write o 2D | read o 2D + read x 4D + write y 4D
           = 16D bytes per token per layer, 3 launches
    v31    read x 4D + write y 4D
           = 8D bytes per token per layer, 1 launch

`g24` removed 4D of 14D and one launch of two, working on SDPA's output. Owning the
attention kernel removes 8D of 16D and two launches of three.

THE PREMISE `g24` HAD TO KILL, ASKED AGAIN OF OUR OWN CODE
-----------------------------------------------------------
g24's assignment claimed the out-projection reads through a head-major gather. Finding 30
falsified it in one script: SDPA returns a `[B, S, H, hd]`-CONTIGUOUS buffer wearing a
head-major view, so `transpose(1, 2).reshape` is a free view at every shape in the matrix.
The assignment for THIS candidate asked whether the same is true of v23's output, which is
our code rather than the vendor's. It is, more flatly: `single_tile_attention` allocates
`torch.empty((B, S, d_model))` and writes head `h` at column offset `h * head_dim`, so
`ctx` is token-major contiguous **by construction** -- no view, no transpose, no stride to
inspect. `bench/probe_outproj_epilogue.py` prints `ctx.stride()` and
`ctx.is_contiguous()` rather than arguing about them (L27's audit rule; the third time in
this project that reading the artifact beat reasoning about it).

Two consequences. There is no gather to absorb, so **the win here is materialization, not
layout** -- same correction g24 had to make. And g24's `CONTIG` constexpr, worth
1.185x -> 1.500x at head_dim 8 because Triton could otherwise prove only a 16-byte
contiguous run, has no analogue: this kernel never addresses `ctx` at all, so there is no
address form to get wrong. That 1.5x lesson is inherited as "there is nothing left to
vectorize here", not as code.

WHERE IT DECLINES, WHICH IS THE PART THAT COST THE THINKING
------------------------------------------------------------
The fusion is not free. Relative to v23 the program gains an fp32 `[BM, d_model]`
accumulator live across the whole head loop, and the grid LOSES a factor of `heads`
(v23 emits `ceil(S/BM) * heads * B` programs; this emits `ceil(S/BM) * B`). Both reduce
occupancy, which for a kernel of this shape is most of the latency hiding available.
So the predicate has three parts, all shapes and measured device properties:

  * legality        -- an `mma.sync`-shaped tile whose working set fits the register file
  * residency       -- >= 4 resident blocks per SM, v23's MEASURED crossover, reused
                       unchanged and therefore conservatively (this kernel has a head loop
                       whose iterations are independent, so its true crossover is lower)
  * saturation      -- `programs >= props.multi_processor_count`, which is exactly the
                       rule g24 MEASURED for the out-projection GEMM's tile crossover: the
                       sign flipped at 66 SMs, not at a token count

Evaluated on this card the fused path takes configs **1, 4, 5, 6, 7, 11, 12** and declines
2 and 3 (too few programs -- 8 and 32 against 66 SMs), 9, 10 and 13 (the accumulator plus
the score tile leave fewer than 4 blocks resident), and 8 and 14 (no legal tile, as for
v23). Every declined shape falls back to v23's split path, which is the frontier and is
already fast, so declining costs nothing that was being won.

Note that config 6 -- 1.28M tokens, 48.5s of a 112s full sweep, and the config the
profile in the assignment was taken on -- is in the accepted set, and configs 9, 10, 13
are shapes where v23's own numbers are weakest anyway.

CORRECTNESS
-----------
* **Causality** is inherited from v26: a non-causal config never reaches this code, it
  delegates to the unmodified baseline (finding 32 / L42). The kernel masks the causal
  triangle structurally and would be wrong without that gate.
* **The residual stays fp32** (finding 08: an fp16 residual failed 12 of 14 configs). The
  accumulator, the bias add, the residual load, the sum and the store are all fp32.
* **The masked path is supported rather than declined.** g24 declined it wholesale; here
  the mask is one `tl.where` on the projection output before the residual add, which is
  exactly `o.masked_fill(~mask[..., None], 0)` -- the operation v8's fast path performs at
  that point. Padding ratio is a benchmark-exposed knob that finding 11 exists to serve
  (L5), so keeping the fused path available there is worth four lines.
* **Precision improves.** The fp16 rounding of `ctx` is common to both paths and is
  unavoidable (tensor-core operands; FlashAttention rounds `P` the same way). What the
  fusion DELETES is the fp16 rounding of the projection OUTPUT, which `F.linear` on fp16
  performs before `.float()` widens it again. Same direction as the FFN megakernel and as
  g24, which measured ~600x tighter against fp64 for exactly this reason.

WHAT TO EXPECT END TO END, STATED BEFORE ANY MEASUREMENT (L33)
---------------------------------------------------------------
The fresh profile of the frontier at config 6's shape gives the out-projection's cutlass
GEMM at **6.8%** of forward time and our attention kernel at 15.5%. This candidate does
not delete the 6.8%: the projection's arithmetic still has to happen, now inside our
kernel instead of inside cuBLAS's, and cuBLAS is very good at it. What it deletes is the
`ctx` round trip and two launches -- call it the memory-bound half of that bucket plus a
share of the widen-and-add pointwise Inductor currently fuses into the LayerNorm bucket.

**So the honest ceiling is 3-5% on config 6, and less on the 13-config geomean**, because
six of the thirteen configs decline. That is INSIDE L29's +/-7% noise floor. A screen
cannot resolve it, and the screen set makes it worse: of configs (2, 7, 8, 10) the fused
path fires on **7 only**. Anything the screen prints above ~1.05x should be disbelieved
before it is celebrated (L33), and a PROMOTE here means "not clearly worse", nothing more.

The defensible claims are structural: half the epilogue's HBM traffic, two launches of
three removed per layer per forward, an fp16 rounding step deleted from a path that runs
`num_layers` times, and a predicate that declines rather than guesses on the six shapes
where the fusion's occupancy cost is not paid back.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v26_causal_correct import build as build_v26
from ..kernels.attn_outproj import applies, attn_outproj, autotune_tile
from ..kernels.ffn_fused import fused_ffn


def build(baseline_cls):
    v26_cls = build_v26(baseline_cls)

    class CandidateV31(v26_cls):
        outproj_used: bool = False
        outproj_reason: str = "undecided"
        outproj_tile: tuple[int, int, int] | None = None

        def _prime(self, mask):
            super()._prime(mask)
            # `nn.Linear` stores [out, in]; the kernel contracts over the LEADING axis, so
            # it wants [in, out]. Transposed ONCE, at prime time -- per call this would
            # cost more than the fusion wins (the v16/v17 lesson).
            self._out_t = [cached[2].t().contiguous() for cached in self._cache]

        def _decide_outproj(self, x):
            """Decided ONCE, before compilation and graph capture, so the tile is a Python
            constant by the time anything traces it.

            THE LAUNCH-WRAPPER RULE. A sibling's first screen read -18.9% because plan
            resolution ran inside Dynamo's traced region and dropped the frame to eager.
            Everything shape- or device-dependent is resolved here; `attn_outproj` itself
            contains no `.item()`, no `bool(tensor)` and no `is_contiguous()` branch.
            """
            a = self.layers[0].attention
            b, s, _ = x.shape
            props = torch.cuda.get_device_properties(x.device)
            ok, why = applies(s, a.head_dim, a.num_heads, b, props)
            if not ok:
                self.outproj_used = False
                self.outproj_reason = why
                return
            try:
                tile, how = autotune_tile(s, a.head_dim, a.num_heads, b, x.device)
            except Exception as exc:                   # never fail closed on a tuner
                self.outproj_used = False
                self.outproj_reason = f"declined: tile selection failed ({exc})"
                return
            self.outproj_tile = tile
            self.outproj_used = True
            self.outproj_reason = f"{why}; {how}"

        def _core(self, x, mask):
            # Three gates, all decided at prime time. `_fastpath` is v8's proof that a
            # right-padded causal key mask is redundant; `attn_used` is v23's own
            # predicate, and a shape it declined has no context tile for us to project.
            if not self.outproj_used or not self.attn_used or not self._fastpath:
                return super()._core(x, mask)

            lp = torch.float16
            zero = self._needs_zeroing
            use_ffn = self.fused_ffn_used and self._nomask      # v17's own condition
            bm, warps, stages = self.outproj_tile
            # A Python constant, not a tensor predicate: `HAS_MASK` must be a constexpr.
            attn_mask = mask if zero else None

            for layer, cached, out_t, ffn_t in zip(
                    self.layers, self._cache, self._out_t, self._ffn_t):
                a = layer.attention
                qkv_w, qkv_b, out_b = cached[0], cached[1], cached[3]
                b, s, d = x.shape

                qkv = F.linear(layer.norm1(x).to(lp), qkv_w, qkv_b)
                # THE CHANGE. Attention, the out-projection, the fp32 widen, the mask and
                # the fp32 residual add, in ONE launch. No fp16 context tile, no fp16
                # projection temporary, nothing between them touching HBM.
                x = attn_outproj(qkv, x.view(-1, d), out_t, out_b, attn_mask,
                                 a.num_heads, a.head_dim, a.scale,
                                 bm, warps, stages).view(b, s, d)

                if use_ffn:
                    w1t, b1, w2t, b2 = ffn_t
                    xn = layer.norm2(x).to(lp).view(-1, d)
                    x = fused_ffn(xn, x.view(-1, d), w1t, b1, w2t, b2,
                                  self.BLOCK_M, self.NUM_WARPS).view(b, s, d)
                else:
                    in_w, in_b, ffn_w, ffn_b = cached[4], cached[5], cached[6], cached[7]
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
            # A non-causal config never reaches `_core` (v26 delegates to the unmodified
            # baseline), so do not spend a tuner probe deciding a path it will not take.
            if (self.outproj_reason == "undecided"
                    and getattr(self.config, "causal", True)):
                self._decide_outproj(x)
            return super().forward(x, valid_token_mask)

    return CandidateV31
