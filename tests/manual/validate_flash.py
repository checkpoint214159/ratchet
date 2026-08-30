"""Manual E2 correctness driver: flash_attention vs the oracle. Not a pytest test yet."""
import functools
import sys

from ratchet.kernels.flash_attention import flash_attention
from ratchet.oracle import (
    CORRECTNESS_SHAPES,
    DeterministicContext,
    check_all,
    correctness_suite,
    reference_fp64,
)

adversarial = "--adv" in sys.argv


def _ref_in_dtype(shape):
    # The oracle's floor lens: fp64 reference viewed in the competition dtype, exactly the
    # `.to(q.dtype)` reference_fp32 applies to its own output (see test_reference_floor).
    def ref(q, k, v):
        return reference_fp64(q, k, v, causal=shape.causal).to(q.dtype)
    return ref


npass = 0
for i, shape in enumerate(CORRECTNESS_SHAPES):
    cand = functools.partial(flash_attention, causal=shape.causal)
    ref = _ref_in_dtype(shape)
    with DeterministicContext():
        suite = correctness_suite(shape, seed=4242 + i * 101, include_adversarial=adversarial)
        res = check_all(cand, ref, suite)
    tag = "PASS" if res.passed else f"FAIL[{res.gate_failed}]"
    npass += res.passed
    print(f"{tag:14s} {shape.key()}  max_abs={res.max_abs_err:.2e} max_rel={res.max_rel_err:.2e}")
    if not res.passed:
        print(f"    per_dist={res.per_distribution}")
        print(f"    {res.diagnostic[:300]}")
print(f"\n{npass}/{len(CORRECTNESS_SHAPES)} shapes passed (adversarial={adversarial})")
