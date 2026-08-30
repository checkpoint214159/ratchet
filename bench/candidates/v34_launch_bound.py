"""Candidate v34 -- stop paying for launches the shape is too small to hide.

Generation 34. Parent: v26_causal_correct (the frontier). Branch: cand/g34/launch-bound.

THE CENSUS THAT MOTIVATES IT
----------------------------
Profiled the real frontier -- built the way `run_matrix` builds it, warmed the way the
harness warms it, so this is the candidate and not a decomposition (L41). Kernels per
forward, counted from device events inside the replayed graph:

    config  2 (B=1,     0.061 ms)   36 kernels     53.2 us device
    config 12 (B=64,    0.103 ms)   36 kernels     95.2 us device
    config  8 (B=64,    6.549 ms)   36 kernels   6340.5 us device

**Thirty-six on every one of them, with an identical decomposition** (L=4 layers):

    16   GEMM            qkv, out_proj, ffn_in, ffn_out          4 per layer
     9   LayerNorm       norm1 x4, norm2 x4, final_norm          each with the residual
                                                                 add and the fp16 downcast
                                                                 already fused in by Inductor
     4   attention       _attn_single_tile (v23) or flash
     4   GELU            NOT fused into the ffn_in epilogue -- Inductor's 68-SM veto
                         (finding 22) disables Triton GEMM templates on this 66-SM card,
                         so every GEMM here is cuBLAS/CUTLASS and takes no epilogue
     3   Memcpy DtoD     _static_x.copy_(x), _static_m.copy_(mask), _static_y.clone()

AND WHAT A KERNEL COSTS WHEN IT COMPUTES ALMOST NOTHING
-------------------------------------------------------
Calibrated on this card by capturing a graph of N identical trivial kernels and fitting
replay time against N over N = 1 .. 256:

    replay(N) = 1.886 + 0.7984 * N  us          one trivial node's device duration: 775 ns

So 36 nodes is a **28.7 us floor on every config** -- 47% of config 2's entire 0.061 ms
wall, 28% of config 12's, and 0.45% of config 8's. The lever on the launch-bound rows is
not a faster kernel. It is fewer kernels.

WHY THE EXISTING FUSION PREDICATE DECLINES EXACTLY HERE
--------------------------------------------------------
`kernels.ffn_fused.amortizes` asks a BANDWIDTH question -- do enough tokens stream past
the hoisted weights to pay for hoisting them? At d_model = ffn_dim = 128 it wants ~32000
tokens, and it declines every launch-bound row. On its own terms it is right, and
finding 29 confirmed it from the other side: v19 folded the norms into the megakernel and
measured **FLAT on config 6**, because config 6 is at 97% of the HBM roofline and fusion
there only moves bytes that were already moving at the achievable rate.

**Config 2 has almost no traffic and pays in launches instead.** The same fusion is worth
having there for the opposite reason. So this candidate adds a SECOND, disjoint predicate
(`kernels.ffn_fused.one_wave`) and reuses v19's already-measured kernel underneath it.

THE PREDICATE IS OCCUPANCY, AND IT IS NOT FITTED TO THIS MATRIX
---------------------------------------------------------------
When every thread block of the fused segment is resident on the device at once, the
segment runs in ONE WAVE: nothing in it is throughput-limited, and its cost is one launch
latency plus one block's serial chain. Splitting that same work over five launches
multiplies the launch latency by five and buys nothing back, because there was never a
second wave for the later launches to overlap with. Above one wave the reverse is true --
later waves hide launch latency behind earlier ones -- and `amortizes` governs instead.

Both inputs (`multi_processor_count`, `shared_memory_per_multiprocessor`) are read off
`torch.cuda.get_device_properties` at run time. No config id, no announced shape, no
crossover constant fitted to these fourteen rows (CLAUDE.md rule 2). Evaluated on the
announced matrix it selects configs 2, 3, 4 and 12, and `amortizes` keeps 6, 7 and 13 --
disjoint sets, which the tests assert rather than assume.

WHAT IS ACTUALLY DELETED
------------------------
Per layer the fused kernel absorbs the attention residual add, norm2, ffn_in, GELU,
ffn_out, and the NEXT layer's norm1: **five kernels become one**. Plus the dead mask copy
(below). Counted:

    36  ->  20 kernels per forward       16 nodes x 0.798 us  =  12.8 us removed

STATED BEFORE MEASURING, AS THE DILUTED FIGURE (L33)
-----------------------------------------------------
12.8 us against each config's own wall, assuming the fused kernel is no slower than the
five it replaces -- which is a ceiling, not a prediction:

    config  2   12.8 us / 0.061 ms   =  21.0%      1.92x -> at most 2.43x
    config 12   12.8 us / 0.103 ms   =  12.4%      1.99x -> at most 2.27x
    config  4   12.8 us / 0.111 ms   =  11.5%      1.98x -> at most 2.24x
    config  3   past the 3.0 score cap; must not regress, cannot help

On the capped weighted score that is at most **+0.065 of 3.000**, and configs 1, 8, 9, 10
get exactly nothing because the predicate declines them. Anything above that should be
disbelieved before it is celebrated. There is upside beyond it -- the five replaced
kernels are 42.5 us of config 12's 95.2 us of device time, and the fused one need not
cost that -- but the launch floor is the part the mechanism guarantees.

THE THIRD MEMCPY WAS DEAD WORK
------------------------------
`_static_m.copy_(mask)` runs on every call of every config. On the fast path it copies a
tensor nothing reads: `_nomask` is True exactly when the mask is all-True, and then
`_needs_zeroing` is False, so `zero` is a Python False baked into the traced graph and
`mask` is not dereferenced on any reachable path through `_core`. Captured with `None`
instead, the copy and its node disappear. Provable rather than hopeful, and reported via
`mask_capture` so the change is observable (L36/finding 18: a silent degradation is what
hides for six generations).

PRECISION
---------
Strictly better than the path it replaces, as finding 29 measured on config 6 (max_abs
1.87e-3 -> 1.56e-3). The residual stays fp32 from the attention add, through norm2, to
the store, never round-tripping HBM, where the Inductor kernel materializes it in fp32 and
its normalized copy in fp16. GELU is the exact erf form (`approximate="none"`). This
candidate passes the LayerNorm weights in **fp32** rather than v19's fp16 -- the kernel
upcasts them to fp32 before use either way, so this removes a rounding step for 512 bytes
of extra traffic per layer. Finding 08's fp32 residual is preserved verbatim.

THE TILE, AND THE LAUNCH WRAPPER
--------------------------------
`launch_tile` narrows the tile in this regime instead of widening it: there are not enough
rows to amortize a weight load, so the right move is to put a block on as many SMs as
possible and let the 48 MB L2 serve the identical weight reads. num_warps is then
confirmed by a two-point sweep at PRIME time, under v23's `DECISIVE` discipline -- the
derived value holds unless something beats it by more than the noise floor.

Every one of those choices is resolved to a Python int in `_decide_launch`, which runs in
`forward` BEFORE `torch.compile` and before graph capture. Nothing about the plan is
resolved inside the traced region; a sibling candidate that got that wrong dropped the
frame to eager and screened at -18.9%.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v26_causal_correct import build as build_v26
from ..kernels.ffn_fused import (amortizes, fits, fused_ffn_normed, launch_tile,
                                 one_wave)

# Warps to try for the fused block, and the derived default. A mechanism argument cannot
# pick this (v23's docstring), so it is confirmed by timing at prime time -- but only
# overridden by more than the margin below, because these kernels run in a few
# microseconds and a candidate whose tile varies run to run adds that variance to every
# measurement taken of it (L29).
WARP_CANDIDATES: tuple[int, ...] = (4, 8)
DERIVED_WARPS = 8
DECISIVE = 0.10


def build(baseline_cls):
    v26_cls = build_v26(baseline_cls)

    class CandidateV34(v26_cls):
        launch_fused_used: bool = False
        launch_reason: str = "undecided"
        launch_bm: int = 0
        launch_warps: int = DERIVED_WARPS
        mask_capture: str = "undecided"

        # ---------------------------------------------------------------- prime
        def _prime(self, mask):
            super()._prime(mask)
            # norm2 and the NEXT consumer's norm, per layer. fp32: the kernel upcasts
            # them anyway, so keeping them fp32 removes a rounding step for free.
            self._n_fused = []
            n = len(self.layers)
            for i, layer in enumerate(self.layers):
                nxt = self.layers[i + 1].norm1 if i + 1 < n else self.final_norm
                self._n_fused.append((
                    layer.norm2.weight.float(), layer.norm2.bias.float(),
                    nxt.weight.float(), nxt.bias.float(),
                    float(layer.norm2.eps), i == n - 1,
                ))

        # ------------------------------------------------------------- decision
        def _decide_launch(self, x):
            """Resolved ONCE, before compilation and graph capture, so every tile
            parameter is a Python constant by the time anything traces `_core`."""
            b, s, d = x.shape
            f = self.layers[0].ffn_in.weight.shape[0]
            tokens = b * s
            props = torch.cuda.get_device_properties(x.device)
            bm = launch_tile(tokens, props.multi_processor_count)

            if not self._nomask:
                self.launch_fused_used = False
                self.launch_reason = "declined: masked input, the kernel does no masking"
                return
            if not fits(d, f, 2, bm, props.shared_memory_per_block_optin):
                self.launch_fused_used = False
                self.launch_reason = (
                    f"declined: d_model={d} ffn_dim={f} exceeds "
                    f"{props.shared_memory_per_block_optin} B opt-in smem")
                return
            if amortizes(tokens, d, f, 2):
                # The throughput regime. Finding 25's crossover already governs here and
                # v26's own path is what was measured; do not disturb it.
                self.launch_fused_used = False
                self.launch_reason = (
                    "declined: above the amortization crossover, so the segment is "
                    "throughput-bound and later waves already hide the launches")
                return
            if not one_wave(tokens, d, f, 2, bm, props.multi_processor_count,
                            props.shared_memory_per_multiprocessor):
                self.launch_fused_used = False
                self.launch_reason = (
                    f"declined: {-(-tokens // bm)} blocks at {bm} rows is more than one "
                    f"wave on {props.multi_processor_count} SMs, so launch latency is "
                    f"already hidden")
                return

            self.launch_bm = bm
            self.launch_warps, how = self._pick_warps(tokens, d, f, bm, x.device)
            self.launch_fused_used = True
            self.launch_reason = (
                f"fused: {-(-tokens // bm)} blocks at {bm} rows fits one wave on "
                f"{props.multi_processor_count} SMs; {how}")

        def _pick_warps(self, tokens, d, f, bm, device):
            """Time the two candidate warp counts once, here, outside every traced region.

            The derived value holds unless something beats it DECISIVELY -- inside that
            margin the ranking is noise and letting it vary would inject that noise into
            every measurement taken of this candidate (L29).
            """
            try:
                import triton.testing as tt
                lp = torch.float16
                xr = torch.randn(tokens, d, device=device, dtype=torch.float32)
                ar = torch.randn(tokens, d, device=device, dtype=torch.float32)
                w1 = torch.randn(d, f, device=device, dtype=lp)
                w2 = torch.randn(f, d, device=device, dtype=lp)
                b1 = torch.randn(f, device=device, dtype=lp)
                b2 = torch.randn(d, device=device, dtype=lp)
                g = torch.randn(d, device=device, dtype=torch.float32)
                timed = {}
                for w in WARP_CANDIDATES:
                    try:
                        fn = (lambda w=w: fused_ffn_normed(
                            xr, ar, g, g, w1, b1, w2, b2, g, g, 1e-5, bm, w, True))
                        fn()
                        timed[w] = min(tt.do_bench(fn, warmup=10, rep=25,
                                                   return_mode="min") for _ in range(2))
                    except Exception:
                        continue
                del xr, ar, w1, w2, b1, b2, g
                if timed:
                    best = min(timed, key=timed.get)
                    base = timed.get(DERIVED_WARPS)
                    if base is None or timed[best] < base * (1.0 - DECISIVE):
                        return best, (f"{best} warps beat the derived "
                                      f"{DERIVED_WARPS} decisively")
                    return DERIVED_WARPS, f"derived {DERIVED_WARPS} warps, confirmed"
            except Exception:
                pass
            return DERIVED_WARPS, "derived (timing unavailable)"

        # ------------------------------------------------------- capture, minus
        # ------------------------------------------------------- one dead copy
        def _try_capture(self, x, mask):
            """Capture with no mask when the mask provably cannot be read.

            `_nomask` is True exactly when every entry is valid, and then
            `_needs_zeroing` is False, so `zero` is a Python constant False inside the
            traced graph and `mask` is not dereferenced on any reachable path through
            `_core`. Capturing with None removes `_static_m` entirely, and with it one
            DtoD memcpy node and its `cudaMemcpyAsync` on every call of every config.
            """
            if getattr(self, "_nomask", False):
                self.mask_capture = "elided (no mask is read on the fast path)"
                mask = None
            else:
                self.mask_capture = "copied (masked input)"
            return super()._try_capture(x, mask)

        # ------------------------------------------------------------------ core
        def _core(self, x, mask):
            if not self.launch_fused_used:
                return super()._core(x, mask)          # v26/v23's path, untouched

            lp = torch.float16
            bm, warps = self.launch_bm, self.launch_warps
            b, s, d = x.shape
            # Only the FIRST norm1 is its own kernel; every later one is emitted by the
            # previous layer's fused block.
            xn = self.layers[0].norm1(x).to(lp)

            for layer, cached, ffn_t, nrm in zip(self.layers, self._cache,
                                                 self._ffn_t, self._n_fused):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b = cached[0], cached[1], cached[2], cached[3]

                qkv = F.linear(xn, qkv_w, qkv_b)
                ctx = self._attention(qkv, a, b, s)
                # NOT `.float()`. The out-projection is an fp16 GEMM over fp16 operands,
                # so its result is already fp16 and the upcast is a lossless widening --
                # but with the residual add now living inside the megakernel there is no
                # LayerNorm epilogue left for Inductor to fuse that cast into, and it
                # becomes its own kernel node per layer. The megakernel widens it instead,
                # bit-identically, and reads half as many bytes doing it.
                attn = F.linear(ctx, out_w, out_b)

                w1t, b1, w2t, b2 = ffn_t
                n2w, n2b, nnw, nnb, eps, is_last = nrm
                y, yn = fused_ffn_normed(
                    x.view(-1, d), attn.view(-1, d), n2w, n2b, w1t, b1, w2t, b2,
                    nnw, nnb, eps, bm, warps, store_next=not is_last)
                x = y.view(b, s, d)
                if not is_last:
                    xn = yn.view(b, s, d)

            # The last layer deliberately did not emit its normalized output: that one is
            # the model's fp32 answer and must not be rounded on the way out.
            return self.final_norm(x)

        # --------------------------------------------------------------- forward
        def forward(self, x, valid_token_mask=None):
            if not getattr(self.config, "causal", True):
                return super().forward(x, valid_token_mask)   # v26 -> unmodified baseline
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if self.launch_reason == "undecided":
                self._decide_launch(x)
            return super().forward(x, valid_token_mask)

    return CandidateV34
