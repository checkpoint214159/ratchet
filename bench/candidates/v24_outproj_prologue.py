"""Candidate v24 -- the attention out-projection, its fp32 cast and the residual add,
in one Triton kernel.

Generation 24. Parent: v18_capture_insurance. Branch: cand/g24/outproj-prologue.
Idea from proposal D-02; the proposal's headline mechanism did not survive contact and
is documented below rather than deleted.

WHAT D-02 CLAIMED, AND WHAT IS ACTUALLY THERE
---------------------------------------------
D-02 proposed a GEMM whose PROLOGUE does a head-major gather -- the mirror image of
`qkv_headmajor`'s epilogue scatter, which is worth 1.163x on its own segment. The premise
was that

    x = x + F.linear(ctx.transpose(1, 2).reshape(B, S, D), out_w, out_b).float()

feeds a GEMM from a transposed view, forcing a copy.

**It does not, and this was the first thing measured** (`bench/probe_outproj.py --layout-only`).
`F.scaled_dot_product_attention` on this card returns

    shape (B, H, S, hd)     stride (H*S*hd, hd, H*hd, 1)

which is a `[B, S, H, hd]`-CONTIGUOUS buffer wearing a head-major view. That holds at
every shape in the matrix and at every head_dim from 8 to 256. So `ctx.transpose(1, 2)`
is already contiguous, `.reshape` is a free view, there is no copy, and the `Memcpy DtoD`
in the config-6 profile is something else. The gather this candidate was commissioned to
build does not exist to be built.

WHAT SURVIVES IS THE EPILOGUE, AND IT IS SMALLER BUT REAL
---------------------------------------------------------
The two-kernel path still writes an fp16 `[M, D]` temporary that a second kernel
immediately reads back to widen and add to the fp32 residual. Per token:

    two kernels   read ctx 2D + write o 2D | read o 2D + read res 4D + write y 4D = 14D
    fused         read ctx 2D              | read res 4D + write y 4D             = 10D

29% of the segment's traffic and one launch of the two. `bench/kernels/outproj_resid.py`
does the GEMM with the widening and the residual add in its epilogue, in fp32 registers.

MEASURED ON THE SEGMENT at all thirteen runnable configs (`bench/probe_outproj.py`),
against the torch.compile'd two-kernel path on the same data, min-of-3:

    shape      D   tokens      compiled    fused     gain
    cfg2     128      128        0.0082   0.0052    1.579x
    cfg3     128      512        0.0102   0.0070    1.447x
    cfg12    128    2,048        0.0142   0.0100    1.429x
    cfg4     128    2,048        0.0140   0.0109    1.282x
    cfg1     128    8,192        0.0315   0.0213    1.483x
    cfg9     128    8,192        0.0290   0.0212    1.368x
    cfg10    128    8,192        0.0292   0.0212    1.377x
    cfg11    128    8,192        0.0328   0.0219    1.500x
    cfg7      32    8,192        0.0128   0.0088    1.446x
    cfg8   1,024    8,192        0.3280   0.2477    1.324x
    cfg5     128   16,384        0.0512   0.0393    1.302x
    cfg13    128   65,536        0.2028   0.1510    1.343x
    cfg6     128 1,280,000       3.7187   2.8336    1.312x

**No shape loses and nothing declines**, so unlike the FFN megakernel there is no
amortization gate -- inventing one that never fires would be theatre. L33/L41 apply in
full: these are isolated op-level numbers and the harness sweep is the one that counts.

FRACTION OF REAL FORWARD TIME (L33)
-----------------------------------
This is the honest ceiling and it is NOT the 1.4x above. At config 6 the non-flash GEMMs
are ~18.9% of forward time, and the out-projection is one of four GEMMs per layer
(Q|K|V fused counts as one, out-proj, and the FFN's two are already inside the g16
megakernel on this path). Taking the out-projection at roughly a third of that bucket
plus the pointwise add it absorbs, the segment is on the order of **8-10% of config 6's
forward time**, so a 1.4x on it is worth **2-3% end to end** -- INSIDE the +/-7% noise
floor of L29. On the 13-config geomean it is smaller still.

**So this candidate should not be expected to move the geomean, and a screen cannot
resolve it.** What it can defensibly claim is a per-segment 1.28-1.49x with no losing
shape, one fewer kernel launch per layer, and the accuracy result below. Whether that is
worth a full sweep is the controller's call, and the case for it is weak on speed alone.

IT RETURNS TOLERANCE MARGIN, WHICH IS THE STRONGER ARGUMENT (L26)
-----------------------------------------------------------------
The two-kernel path rounds the projection to fp16 before `.float()` widens it again. The
fused kernel deletes that rounding step. Against an fp64 reference on the same data:

    two-kernel max_abs    1.2e-04 .. 2.4e-04
    fused      max_abs    1.4e-07 .. 3.6e-07     ~600x tighter

L26 measured our worst config at 94% of the 2e-3 budget and showed a routine change in
the input distribution multiplies the error by 2.5. This hands back margin on a path that
runs `layers` times per forward, in the same direction as the FFN megakernel (v16 was
also more accurate than what it replaced). Margin is not scored by the geomean and is the
reason to keep this even if the speed is a wash.

DISPATCH
--------
`outproj_resid.fits` is shapes plus `props.shared_memory_per_block_optin`;
`outproj_resid.tiling_for` is shapes plus `props.multi_processor_count`. No config ids,
no announced constants. The masked path is declined wholesale (the parent's behaviour),
because the residual add the kernel absorbs is exactly where the padded path applies its
`masked_fill`, and duplicating that logic inside the kernel to serve a path this frontier
already declines would be spending correctness risk for nothing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v18_capture_insurance import build as build_v18
from ..kernels.ffn_fused import fused_ffn
from ..kernels.outproj_resid import fits, outproj_resid, tiling_for


def build(baseline_cls):
    v18_cls = build_v18(baseline_cls)

    class CandidateV24(v18_cls):
        outproj_used: bool = False
        outproj_reason: str = "undecided"
        outproj_tiling: tuple = ()

        def _prime(self, mask):
            super()._prime(mask)
            # nn.Linear stores [out, in]; the kernel contracts over the LEADING axis, so
            # it wants [in, out]. Once, at prime time -- per call this would cost more
            # than the fusion wins (the v16/v17 lesson).
            self._out_t = [cached[2].t().contiguous() for cached in self._cache]

        def _decide_outproj(self, x):
            """Decided ONCE, before compilation and capture, so the choice is a Python
            constant by the time anything traces it."""
            b, s, d = x.shape
            heads = self.layers[0].attention.num_heads
            props = torch.cuda.get_device_properties(x.device)
            tile = tiling_for(b * s, d, props.multi_processor_count)
            self.outproj_tiling = tile
            if not fits(d, heads, 2, tile[0], tile[1], tile[2],
                        props.shared_memory_per_block_optin):
                self.outproj_used = False
                self.outproj_reason = (
                    f"declined: d_model={d} heads={heads} tile={tile} does not fit "
                    f"{props.shared_memory_per_block_optin} B opt-in smem, or is not a "
                    f"legal tl.dot shape")
            else:
                self.outproj_used = True
                self.outproj_reason = f"fused: tile={tile} on {props.multi_processor_count} SMs"

        def _core(self, x, mask):
            # `_nomask` for the reason in the docstring; the parent's own fused path
            # requires it too, so declining here costs nothing that was being won.
            if not self.outproj_used or not self._nomask:
                return super()._core(x, mask)

            lp = torch.float16
            tile = self.outproj_tiling
            for layer, cached, out_t, ffn_t in zip(
                    self.layers, self._cache, self._out_t, self._ffn_t):
                a = layer.attention
                qkv_w, qkv_b, out_b = cached[0], cached[1], cached[3]
                b, s, d = x.shape

                qkv = F.linear(layer.norm1(x).to(lp), qkv_w, qkv_b)
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)

                # THE CHANGE. Projection, fp32 widening and fp32 residual add, one launch,
                # no fp16 temporary. `ctx` is read through its runtime strides, so this is
                # correct whatever layout the SDPA backend chose.
                x = outproj_resid(ctx, x.view(-1, d), out_t, out_b, *tile).view(b, s, d)

                if self.fused_ffn_used:
                    w1t, b1, w2t, b2 = ffn_t
                    xn = layer.norm2(x).to(lp).view(-1, d)
                    x = fused_ffn(xn, x.view(-1, d), w1t, b1, w2t, b2,
                                  self.BLOCK_M, self.NUM_WARPS).view(b, s, d)
                else:
                    in_w, in_b, ffn_w, ffn_b = cached[4], cached[5], cached[6], cached[7]
                    h = F.linear(layer.norm2(x).to(lp), in_w, in_b)
                    x = x + F.linear(F.gelu(h, approximate="none"), ffn_w, ffn_b).float()

            return self.final_norm(x)

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if self.outproj_reason == "undecided":
                self._decide_outproj(x)
            return super().forward(x, valid_token_mask)

    return CandidateV24
