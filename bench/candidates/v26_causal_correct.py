"""Candidate v26 -- honour config.causal instead of assuming it.

Generation 26. Parent: v23_single_tile_attn (the frontier). Branch: cand/g26/causal-correct.
Bug found by the g22 executor while building something else.

THE DEFECT, WHICH IS IN EVERY CANDIDATE FROM v5 TO v23
-------------------------------------------------------
`F.scaled_dot_product_attention(q, k, v, is_causal=True)` -- hardcoded. Only `v1_fused_graph`
ever read `self.config.causal`. Measured on a non-causal config:

    v1_fused_graph          max_abs 7.05e-04   failed     0 / 32768   OK
    v8_padfast              max_abs 1.43e+00   failed 24846 / 32768   WRONG
    v13_safe_capture        max_abs 1.33e+00   failed 25249 / 32768   WRONG
    v18_capture_insurance   max_abs 1.58e+00   failed 24942 / 32768   WRONG
    v23_single_tile_attn    max_abs 1.67e+00   failed 25064 / 32768   WRONG

Three quarters of the output wrong, by three orders of magnitude past the tolerance.

WHY IT WAS NEVER CAUGHT
-----------------------
`bench/matrix.py` says every one of the 14 announced configs is causal, and it is right --
so no ledger row is affected and no test ever exercised the other branch. But **the
reference benchmark's own default is `causal: bool = False`** (line 89), with `--causal` as
an opt-in flag. Everything we have measured used a setting the harness does not default to.

This is L24's shape at its most literal: correct because of how the harness was invoked.
The whole test suite, 177 tests, was green throughout.

Note also that v8's `_fastpath` DOES consult `self.config.causal` -- the redundant-key-mask
proof (finding 11) explicitly depends on causality. So the information was present in the
same function, one line above the call that ignored it.

THE FIX, AND WHY IT IS THE CONSERVATIVE ONE
-------------------------------------------
When `config.causal` is False, delegate to the UNMODIFIED baseline forward. Not a
faster non-causal path -- the reference implementation itself.

That is deliberate. Every optimization in this lineage was designed and measured under
causality, and several depend on it for CORRECTNESS rather than speed:

  * v8's proof that a right-padded key mask is redundant is derived from causal masking
    (finding 11). Without causality a valid query attends to padding, and the mask is
    load-bearing.
  * v23's single-tile kernel skips the causal triangle structurally.

Writing a fast non-causal path would mean re-deriving all of that against a case the
announced matrix never exercises. Correctness first (CLAUDE.md rule 3): be exactly right
on a shape we do not expect, and fast on the fourteen we do.
"""

from __future__ import annotations

from .v23_single_tile_attn import build as build_v23


def build(baseline_cls):
    v23_cls = build_v23(baseline_cls)

    class CandidateV26(v23_cls):
        causal_path: str = "undecided"

        def forward(self, x, valid_token_mask=None):
            if not getattr(self.config, "causal", True):
                # The reference implementation, unmodified. Slower and exactly right.
                self.causal_path = "baseline (non-causal input)"
                return baseline_cls.forward(self, x, valid_token_mask)
            self.causal_path = "optimized (causal input)"
            return super().forward(x, valid_token_mask)

    return CandidateV26
