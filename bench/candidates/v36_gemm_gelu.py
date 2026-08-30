"""Candidate v36 -- the d_model=128 projections on a hand-written GEMM, GELU in the epilogue.

Generation 36. Parent: v34_launch_bound. Branch: cand/g36/gemm-gelu. Proposal F-01.

THE CENSUS THAT MOTIVATES IT, ON CONFIGS NOBODY HAD EVER PROFILED
------------------------------------------------------------------
Every profile in the record before this one was config 6 or config 2. Profiled on the
real frontier at config 9 -- the #1 headroom row -- **the sixteen projection GEMMs are
55.0% of device time**, running at 47.3 TFLOP/s = **53.6% of this card's measured 88.2
BF16-TFLOP/s peak**. cuBLAS selects `ampere_fp16_s1688gemm...` on all twelve narrow-K
calls: `s1688` is `mma.sync.m16n8k8`, where sm_89 also issues `m16n8k16` and `tl.dot`
emits it. The full argument, the numbers and the three hardware reasons are in
`bench/kernels/proj_gemm.py`'s docstring; it is not restated here.

And the GELU is its own kernel on every layer of every config -- 12.14 us plus four of
config 9's 35 graph nodes -- purely because finding 22's 68-SM veto leaves every GEMM in
this stack on cuBLAS/CUTLASS, which takes no epilogue. Finding 39 already recorded that
correction.

WHAT WAS ALREADY IN THE LEDGER AND NOBODY READ
-----------------------------------------------
`v15_lifted_veto` lifted Inductor's `min_sms = 68` and so let it emit Triton GEMM
templates WITH pointwise epilogues. Finding 22 closed it on the geomean (-1.4%, later
-0.8%). The per-config rows, at two separate commits, say something the geomean hid:

              v9b      v15 (5cc0295a)   v15 (f7e70e9a)
      cfg  9    0.2488     0.2345 -5.8%     0.2355 -5.3%
      cfg 10    0.2529     0.2365 -6.5%     0.2365 -6.5%
      cfg 12    0.1485     0.1403 -5.5%     0.1413 -4.8%
      cfg  5    0.4383     0.4639 +5.8%     0.4669 +6.5%

Two commits, four configs, reproducing to 0.5%, and configs 9 and 10 are now the #1 and
#3 headroom rows. **Finding 22's headline is narrower than it was stated**: it is an
aggregate null over a matrix whose launch-bound rows had not yet been optimised. Config
5's +6% is exactly why the decision here must be per-shape and measured.

THE DECISION IS A MEASUREMENT OF BOTH PATHS, NOT A CLAIM ABOUT SHAPES
---------------------------------------------------------------------
`proj_gemm.plan` times the vendor call and every legal swept tile once, at prime time, on
the real operand shapes, and **keeps the vendor unless Triton wins by more than 10%**.
Four sites (qkv, out_proj, ffn_in+GELU, ffn_out) are decided independently, so the
candidate can take three and decline the fourth. On this card that is expected to decline
config 8 outright, where cuBLAS selects `cutlass_80_tensorop_f16_s16816gemm` and already
reaches 100.4% of measured peak (F-05: config 8 is closed).

This is CLAUDE.md rule 2 satisfied by construction rather than by assertion: there is no
config id and no announced shape constant anywhere in the predicate, and a card where
cuBLAS picks the right kernel declines everywhere without being retuned.

WHERE IT ATTACHES
-----------------
v34 dispatches on `launch_fused_used`. Both of its branches are covered:

  * **declined (configs 1, 5, 8, 9, 10, 11 and the fallback rows)** -- v23's `_core`, with
    four separate `F.linear` calls per layer and a free-standing `F.gelu`. All four sites
    are candidates, and the GELU folds into `ffn_in`'s epilogue.
  * **fused (configs 2, 3, 4, 12)** -- v34's megakernel has already absorbed `ffn_in`,
    the GELU and `ffn_out`, so only `qkv` and `out_proj` remain. Both are candidates.

The two branches are reproduced here rather than hooked, because there is no seam in the
parents to hook: they call `F.linear` directly. Every other line is the parent's.

THE TILE IS SWEPT, AND `n_spills` IS RECORDED
----------------------------------------------
v20 lost at 0.88x on a guessed tile and won at 1.163x on a swept one; the g28 megakernel
measured 1.52x spill-free against 2.28x SLOWER once it spilled. So the tile is timed over
18 candidates, filtered by an accumulator-register budget before anything is compiled, and
the winner's `n_regs`/`n_spills`/`shared` are read off the `CompiledKernel` and reported
on `gemm_stats` (ncu is unavailable under WSL2 -- it denies GPU counters).

THE LAUNCH WRAPPER
------------------
Every tile parameter is resolved to a Python int in `_decide_gemm`, which runs in
`forward` BEFORE `torch.compile` and before graph capture -- the same discipline v34 uses
for `launch_bm`/`launch_warps`. Nothing about the plan is resolved inside a traced region.
A sibling candidate that got this wrong dropped the frame to eager and screened at -18.9%.

PRECISION
---------
Same fp16 operands, same fp32 accumulate, bias added in fp32, one rounding to fp16 --
identical to `F.linear`. The GELU epilogue is strictly BETTER than the split path, which
rounds `h` to fp16, writes it to HBM, reads it back and applies GELU to the rounded value;
the epilogue applies the exact `erf` form to the fp32 accumulator and rounds once. Finding
08's fp32 residual is untouched: every `.float()` in the parents' code stays where it was.

STATED BEFORE MEASURING, AS THE DILUTED FIGURE (L33)
-----------------------------------------------------
The GEMM bucket is 55% of config 9, so an op-level 1.5x caps config 9 at ~1.36x and the
probe's cold 1.26x/1.5x ratios must be halved for L2-hot operands. Against v34's clean
sweep the optimistic total is **+0.12 of the 3.000 weighted score** and the realistic one
**+0.05 to +0.06**; configs 3, 6, 7, 11 and 13 are past the cap and can only regress,
config 8 is expected to decline outright. Anything above +0.12 should be disbelieved
before it is celebrated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v34_launch_bound import build as build_v34
from ..kernels.attn_single_tile import single_tile_attention
from ..kernels.ffn_fused import fused_ffn, fused_ffn_normed
from ..kernels.proj_gemm import plan, proj_gemm

# The four call sites, in the order a layer executes them. `gelu` says whether that
# site's epilogue absorbs the activation.
SITES: tuple[str, ...] = ("qkv", "out", "ffn_in", "ffn_out")


def build(baseline_cls):
    v34_cls = build_v34(baseline_cls)

    class CandidateV36(v34_cls):
        gemm_used: bool = False
        gemm_reason: str = "undecided"
        gemm_sites: tuple[str, ...] = ()
        gemm_engaged: bool = False
        gemm_stats: dict | None = None

        # Each is a (BM, BN, BK, warps, stages) tuple of Python ints, or None for "the
        # vendor won this site". Plain attributes rather than a dict: `_core` is traced by
        # Dynamo and a None-check on an attribute is the least it can possibly guard on.
        _tile_qkv = None
        _tile_out = None
        _tile_ffn_in = None
        _tile_ffn_out = None

        # ------------------------------------------------------------- decision
        def _decide_gemm(self, x):
            """Resolved ONCE, before compilation and graph capture, so every tile
            parameter is a Python constant by the time anything traces `_core`.

            Each site is decided on its own timing, and the vendor holds the ground.
            """
            self.gemm_stats = {}
            reasons, taken = [], []
            try:
                b, s, d = x.shape
                a = self.layers[0].attention
                f = self.layers[0].ffn_in.weight.shape[0]
                m = b * s
                # (site, K, N, gelu). qkv is the fused Q|K|V matrix, so N = 3*d_model.
                shapes = [("qkv", d, 3 * a.d_model, False),
                          ("out", a.d_model, d, False),
                          ("ffn_in", d, f, True),
                          ("ffn_out", f, d, False)]
                # Where a megakernel already owns the FFN there is no `F.linear` left to
                # replace, and planning one produces a tile that is never launched and a
                # `gemm_reason` that overstates what engaged. Two predicates put it
                # there: v34's `one_wave` and v17's `amortizes`, and between them they
                # cover configs 2, 3, 4, 6, 7, 12 and 13.
                if self.launch_fused_used or (self.fused_ffn_used and self._nomask):
                    shapes = shapes[:2]
                seen: dict = {}
                for site, k, n, gelu in shapes:
                    key = (m, k, n, gelu)
                    if key not in seen:
                        seen[key] = plan(m, k, n, gelu, x.device)
                    tile, why, stats = seen[key]
                    if tile is not None:
                        setattr(self, f"_tile_{site}", tile)
                        self.gemm_stats[site] = stats
                        taken.append(site)
                    reasons.append(f"{site}: {why}")
            except Exception as exc:                  # never fail closed on a tuner
                for site in SITES:
                    setattr(self, f"_tile_{site}", None)
                self.gemm_used = False
                self.gemm_engaged = False
                self.gemm_reason = (
                    f"declined: planning failed ({type(exc).__name__}: {exc})")
                return

            self.gemm_sites = tuple(taken)
            self.gemm_used = bool(taken)
            if self.gemm_used:
                # Pre-transpose ONLY what was taken. nn.Linear stores [out, in]; the
                # kernel contracts over the LEADING axis, so it wants [in, out], and
                # transposing per call would hand back more than the kernel wins.
                # `_ffn_t` (v17) already holds the two FFN matrices that way; these are
                # the two attention ones, and at d_model 1024 they are 8 MB a layer --
                # which is why they are not built when the vendor won.
                self._proj_t = [(c[0].t().contiguous() if self._tile_qkv else None,
                                 c[2].t().contiguous() if self._tile_out else None)
                                for c in self._cache]
            # Will `_core` actually REACH the substituted call sites? A plan that is
            # never executed is the failure mode this candidate already had once (see
            # `_core`), so it is reported rather than assumed.
            self.gemm_engaged = bool(
                taken and (self.launch_fused_used or getattr(self, "_fastpath", False)))
            head = ("triton on " + "+".join(taken) if taken
                    else "declined: the vendor won every site")
            if taken and not self.gemm_engaged:
                head += " BUT NOT ENGAGED (no fast path)"
            self.gemm_reason = head + " | " + " | ".join(reasons)

        # ---------------------------------------------------------- the one seam
        @staticmethod
        def _lin(tile, gelu, a, w, wt, bias, out_dtype=None):
            """`F.linear(a, w, bias)`, or the swept Triton kernel when `tile` is not None.

            `a` is [B, S, K] or [M, K]; the kernel is 2-D, so a 3-D input is flattened and
            viewed back -- both metadata-only on a contiguous tensor.
            """
            if tile is None:
                y = F.linear(a, w, bias)
                return F.gelu(y, approximate="none") if gelu else y
            if a.dim() == 3:
                b, s, k = a.shape
                out = proj_gemm(a.reshape(-1, k), wt, bias, tile, gelu, out_dtype)
                return out.view(b, s, out.shape[-1])
            return proj_gemm(a, wt, bias, tile, gelu, out_dtype)

        # ------------------------------------------------------------------ core
        def _core(self, x, mask):
            if not self.gemm_used:
                return super()._core(x, mask)

            lp = torch.float16
            b, s, d = x.shape

            # ------------------------------------------------ v34's fused branch
            if self.launch_fused_used:
                xn = self.layers[0].norm1(x).to(lp)
                for layer, cached, ffn_t, nrm, pt in zip(
                        self.layers, self._cache, self._ffn_t, self._n_fused,
                        self._proj_t):
                    a = layer.attention
                    qkv_w, qkv_b, out_w, out_b = cached[0], cached[1], cached[2], cached[3]
                    qkv_t, out_t = pt

                    qkv = self._lin(self._tile_qkv, False, xn, qkv_w, qkv_t, qkv_b)
                    if self.attn_used:
                        abm, awarps, astages = self.attn_tile
                        ctx = single_tile_attention(qkv, a.num_heads, a.head_dim, a.scale,
                                                    abm, awarps, astages)
                    else:
                        q, k, v = qkv.split(a.d_model, dim=-1)
                        q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                        k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                        v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                        ctx = F.scaled_dot_product_attention(
                            q, k, v, is_causal=True).transpose(1, 2).reshape(
                                b, s, a.d_model)
                    # Deliberately NOT `.float()`: the megakernel widens it, as in v34.
                    attn = self._lin(self._tile_out, False, ctx, out_w, out_t, out_b)

                    w1t, b1, w2t, b2 = ffn_t
                    n2w, n2b, nnw, nnb, eps, is_last = nrm
                    y, yn = fused_ffn_normed(
                        x.view(-1, d), attn.view(-1, d), n2w, n2b, w1t, b1, w2t, b2,
                        nnw, nnb, eps, self.launch_bm, self.launch_warps,
                        store_next=not is_last)
                    x = y.view(b, s, d)
                    if not is_last:
                        xn = yn.view(b, s, d)
                return self.final_norm(x)

            # -------------------------------------------------- v23's plain branch
            # Reached whenever v8's padding proof holds -- WITH OR WITHOUT v23's attention
            # kernel. That second case is not hypothetical and it is where this candidate
            # nearly shipped doing nothing: config 9 is `heads=1, head_dim=128`, v23
            # declines head_dim 128 (finding 31), and an earlier draft of this branch
            # bailed to the parent whenever `attn_used` was False -- so the #1 headroom
            # row ran the parent's four cuBLAS calls while the plan said "triton on
            # out+ffn_in+ffn_out". It passed every correctness check while doing so. L36:
            # assert the mechanism ENGAGED, not just that the answer was right.
            if not self._fastpath:
                return super()._core(x, mask)

            zero = self._needs_zeroing
            use_ffn = self.fused_ffn_used and self._nomask       # v17's own condition

            for layer, cached, ffn_t, pt in zip(self.layers, self._cache, self._ffn_t,
                                                self._proj_t):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                qkv_t, out_t = pt
                b, s, d = x.shape

                qkv = self._lin(self._tile_qkv, False, layer.norm1(x).to(lp),
                                qkv_w, qkv_t, qkv_b)
                if self.attn_used:
                    bm, warps, stages = self.attn_tile
                    ctx = single_tile_attention(qkv, a.num_heads, a.head_dim, a.scale,
                                                bm, warps, stages)
                else:
                    # v8's path verbatim: fp16, no attn_mask, so flash qualifies and the
                    # key mask is provably redundant under causality (finding 11).
                    q, k, v = qkv.split(a.d_model, dim=-1)
                    q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                    k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                    v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                    ctx = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True).transpose(1, 2).reshape(b, s, a.d_model)
                o = self._lin(self._tile_out, False, ctx, out_w, out_t, out_b).float()
                if zero:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o

                if use_ffn:
                    w1t, b1, w2t, b2 = ffn_t
                    xn = layer.norm2(x).to(lp).view(-1, d)
                    x = fused_ffn(xn, x.view(-1, d), w1t, b1, w2t, b2,
                                  self.BLOCK_M, self.NUM_WARPS).view(b, s, d)
                else:
                    w1t, _b1, w2t, _b2 = ffn_t
                    xn = layer.norm2(x).to(lp)
                    # The GELU rides in the epilogue where the plan took `ffn_in`. Where
                    # it did not, `_lin` applies `F.gelu` itself, so this line is the
                    # parent's arithmetic exactly.
                    h = self._lin(self._tile_ffn_in, True, xn, in_w, w1t, in_b)
                    x = x + self._lin(self._tile_ffn_out, False, h,
                                      ffn_w, w2t, ffn_b).float()
                    if zero:
                        x = x.masked_fill(~mask[..., None], 0)

            x = self.final_norm(x)
            return x.masked_fill(~mask[..., None], 0) if zero else x

        # --------------------------------------------------------------- forward
        def forward(self, x, valid_token_mask=None):
            if not getattr(self.config, "causal", True):
                return super().forward(x, valid_token_mask)   # v26 -> unmodified baseline
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            # THE TWO FUSION DECISIONS FIRST, in the order the parents make them: which
            # branch of `_core` runs is what determines which sites are candidates at
            # all. v17's `_decide_ffn` normally runs several `forward`s deep, well after
            # this point; both are guarded on their own "undecided" sentinel, so pulling
            # them forward is idempotent and the parents' own calls become no-ops.
            if self.fused_ffn_reason == "undecided":
                self._decide_ffn(x)
            if self.launch_reason == "undecided":
                self._decide_launch(x)
            if self.gemm_reason == "undecided":
                self._decide_gemm(x)
            return super().forward(x, valid_token_mask)

    return CandidateV36
