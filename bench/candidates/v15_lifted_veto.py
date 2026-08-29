"""Candidate v15 -- lift Inductor's 68-SM veto so GEMM epilogue fusion becomes reachable.

Generation 15. Parent: v9b_reduce_overhead. Branch: cand/g15/lifted-veto.
Node chosen by CMP + Thompson sampling over clade metaproductivity (seed 29 -> 2e855f81).

THE FINDING THIS IS BUILT ON
----------------------------
`torch/_inductor/utils.py:1433` reads:

    min_sms = 16 if device.type == "xpu" else 68  # 3080

This card has **66 SMs**. So `is_big_gpu()` returns False, and its single call site
(`_use_template_for_gpu`, line 1488) gates `use_triton_template`, which gates whether
Inductor may emit a Triton GEMM TEMPLATE at all. No template means no epilogue fusion
into a GEMM -- bias, GELU, residual add and casts each stay a separate pass over the
activation tensor.

We logged the `Not enough SMs to use max_autotune_gemm mode` warning back at L15 and drew
the right local conclusion (max-autotune and reduce-overhead measure identically, so use
the cheap one) while missing the consequence entirely. **L15 was right that the two modes
were equivalent, and wrong about why it mattered: they were equivalent because the more
powerful one had been silently disabled.** That is the likeliest single explanation for
the generation 11-14 plateau.

MEASURED, at config 6's shape (D=128, 1.28M tokens), fp16, fresh Inductor cache:

    stock   (is_big_gpu False)   5.576 ms   CUTLASS GEMMs
                                            + triton_poi_fused_addmm_gelu   1.006 ms
                                            + memcpy128                     1.061 ms
    patched (is_big_gpu True)    3.537 ms   triton_tem_fused_addmm_gelu x2
                                            (epilogue fused INTO the GEMM)

1.58x on the FFN pattern, far outside the +/-7% noise floor. Both the standalone
pointwise kernel and the memcpy disappear: they became the GEMM's epilogue.

WHY THIS IS NOT CHEATING, WHICH IS THE PART THAT MATTERS
--------------------------------------------------------
Rule 2 forbids benchmark special-casing, and "monkeypatch a library constant" deserves
suspicion. The defence is that we are not FORCING a kernel choice, we are removing a gate
that prevents one from being CONSIDERED:

  * `max-autotune` already benchmarks every candidate implementation and keeps the
    fastest. The veto does not express "templates are slower here" -- it prevents the
    autotuner from ever timing them.
  * Verified: at D=1024 with the veto lifted, the autotuner still chose CUTLASS. The
    patch widens the search space; it does not dictate the outcome.
  * The threshold is a hardcoded heuristic with `# 3080` written beside it, not a
    correctness condition. 66 vs 68 is a 3% difference in SM count.
  * It is a statement about the DEVICE (this card's SM count vs a constant tuned for a
    different card), not about the benchmark's shapes. It generalizes to every sm_89
    consumer part, all of which sit under 68 SMs.

The lift is applied only when the veto is actually firing on this device, so on a card
with >= 68 SMs this candidate is byte-for-byte v9b with max-autotune.

WHAT THIS TESTS ON ITS LINEAGE
------------------------------
v9b exists to answer "is max-autotune worth its compile cost?" and answered no. That
answer was conditioned on a disabled autotuner. This re-asks the same question with the
autotuner actually enabled -- the sharpest possible test of its parent's conclusion.
"""

from __future__ import annotations

import torch

from .v8_padfast import build as build_v8


def lift_sm_veto() -> bool:
    """Let Inductor CONSIDER Triton GEMM templates on a card just under its threshold.

    Returns True if the veto was firing and has been lifted, False if this device was
    never vetoed (in which case nothing is patched). Idempotent.

    `is_big_gpu` is `functools.cache`d, so we replace the function object rather than
    trying to invalidate its cache.
    """
    import torch._inductor.utils as iu

    if getattr(iu, "_ratchet_veto_lifted", False):
        return True
    try:
        vetoed = not iu.is_big_gpu(0)
    except Exception:
        return False
    if not vetoed:
        return False           # >= 68 SMs: nothing to lift, leave torch alone
    iu.is_big_gpu = lambda *a, **k: True
    iu._ratchet_veto_lifted = True
    return True


def build(baseline_cls):
    v8_cls = build_v8(baseline_cls)

    class CandidateV15(v8_cls):
        use_graph = False          # Inductor owns graph capture, as in v9a/v9b
        veto_lifted: bool = False

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if not hasattr(self, "_compiled"):
                # Lift BEFORE compiling: the gate is read during lowering, so patching
                # afterwards would have no effect on an already-compiled graph.
                self.veto_lifted = lift_sm_veto()
                self._compiled = torch.compile(
                    self._core, mode="max-autotune", dynamic=False)
            return self._compiled(x, valid_token_mask)

    return CandidateV15
