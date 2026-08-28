"""The M2 acceptance gate, encoded: every known-bad kernel is REJECTED, by the gate
its defect was seeded to trip, with a diagnostic that names why.

The known_bad registry is the correctness gate's own test suite. Each fixture carries
exactly one defect from the classes machine-generated kernels actually exhibit; if any
of them ever passes here, the gate is broken and nothing downstream is trustworthy --
including every ledger row written since the regression. These tests therefore pin
three things per (fixture, shape) cell, not one:

  1. passed is False              -- the kernel is rejected at all;
  2. gate_failed == expected_gate -- rejected by the RIGHT gate, so a regression is
                                     detectable by name (a nan_swallower suddenly
                                     caught by 'tolerance' means check_nonfinite
                                     stopped seeing what it used to see);
  3. the diagnostic names the tripping distribution and carries indices -- a bare
     boolean would starve the proposer's feedback loop (see _format_mismatches).

One cell is mathematically exempt and pinned as its own tripwire instead:
wrong_scale at N=1 decode. Softmax over a single score is 1.0 under any monotone
rescaling, so the wrong temperature is invisible there -- the kernel is exactly
correct at that shape. That pass is load-bearing evidence, not a hole: it proves the
fixture carries exactly the one seeded defect (a second, accidental bug would fail at
N=1 too), and it is why N=1 alone could never anchor a correctness suite.
"""

import functools

import pytest
import torch

from ratchet.oracle import (CORRECTNESS_SHAPES, DeterministicContext, check_all,
                            correctness_suite, reference_fp32)
from ratchet.oracle.known_bad import KNOWN_BAD

# The four regimes where kernels actually break, one shape each. The fixtures were
# verified against all nine correctness shapes at seeding time; the golden matrix keeps
# the fast tripwire on the regimes the M2 gate names.
_SHAPE_IDXS = (
    0,   # B2 N127 H4 D64        off-by-one below a power of two (masking/tail bugs)
    3,   # B1 N255 H8 D64        causal
    7,   # B1 N513 H8 D128 Hkv2  GQA, the occupancy-collapse regime
    8,   # B1 N1 H8 D128 Hkv2    single-token decode
)
_SHAPES = tuple(CORRECTNESS_SHAPES[i] for i in _SHAPE_IDXS)

# The diagnostic's leading "[distribution]" tag must name a real member of the suite;
# the roster is pinned here in gate order, same as test_inputs_golden pins it.
_DISTRIBUTION_ROSTER = ("standard", "scaled_up", "scaled_down", "negated", "denormal",
                        "near_overflow", "noncontiguous", "with_nan", "with_inf")

# Every cell of the fixture x shape matrix, minus the one documented mathematical
# exemption (wrong_scale at N=1), which gets its own pinned test below.
_CASES = [
    pytest.param(kb, shape, id=f"{kb.name}-{shape.key()}")
    for kb in KNOWN_BAD
    for shape in _SHAPES
    if not (kb.name == "wrong_scale" and shape.N == 1)
]


def _run_gate(kb, shape):
    """One fixture through the full locked gate: the complete adversarial suite,
    judged against causal-wrapped reference_fp32, exactly as a real candidate is."""
    cand = functools.partial(kb.kernel, causal=shape.causal)
    ref = functools.partial(reference_fp32, causal=shape.causal)
    with DeterministicContext():
        return check_all(cand, ref, correctness_suite(shape))


def test_registry_is_the_seeded_set():
    """Pin the registry by name and expected gate. A fixture silently dropped, renamed,
    or re-pointed at a different gate is a gate regression even if every remaining
    fixture still fails -- the whole point of expected_gate is detection BY NAME."""
    assert {kb.name: kb.expected_gate for kb in KNOWN_BAD} == {
        "identity": "tolerance",
        "wrong_scale": "tolerance",
        "shape_special": "tolerance",
        "nan_swallower": "nonfinite",
    }
    for kb in KNOWN_BAD:
        assert callable(kb.kernel), f"{kb.name}: kernel is not callable"
        assert kb.description, f"{kb.name}: a fixture without a description is unauditable"


def test_matrix_covers_the_contract():
    """The golden shapes must keep exercising the regimes the M2 gate names."""
    assert any(s.N in (127, 128, 129) for s in _SHAPES), "off-by-one shapes must be exercised"
    assert any(s.causal for s in _SHAPES), "causal attention must be exercised"
    assert any(s.H_kv != s.H and s.N > 1 for s in _SHAPES), "GQA must be exercised"
    assert any(s.N == 1 for s in _SHAPES), "single-token decode must be exercised"


@pytest.mark.parametrize("kb,shape", _CASES)
def test_known_bad_is_rejected_by_its_expected_gate(kb, shape):
    """M2 acceptance: rejected, by the right gate, with a diagnostic naming why."""
    res = _run_gate(kb, shape)

    assert not res.passed, (
        f"{kb.name} @ {shape.key()}: a known-bad kernel PASSED the gate. "
        f"The gate is broken; distrust every measurement since. ({kb.description})"
    )
    assert res.gate_failed == kb.expected_gate, (
        f"{kb.name} @ {shape.key()}: rejected by {res.gate_failed!r}, expected "
        f"{kb.expected_gate!r} -- the defect is being caught by the wrong gate, "
        f"which means the expected gate no longer sees it. {res.diagnostic}"
    )
    assert res.diagnostic, (
        f"{kb.name} @ {shape.key()}: rejected without a diagnostic -- a bare boolean "
        f"starves the proposer's feedback loop"
    )
    # check_all prefixes every per-distribution failure with "[distribution]"; the
    # M2 gate reads "a diagnostic naming why", so the name itself is pinned.
    tag = res.diagnostic.split("]", 1)[0].lstrip("[")
    assert res.diagnostic.startswith("[") and tag in _DISTRIBUTION_ROSTER, (
        f"{kb.name} @ {shape.key()}: diagnostic does not name the tripping "
        f"distribution: {res.diagnostic[:120]!r}"
    )


def test_wrong_scale_is_invisible_at_single_token_decode():
    """The documented exemption, pinned from both sides.

    At N=1 the wrong softmax temperature cannot be observed (one score, weight 1.0
    regardless), so wrong_scale must PASS there -- and that pass certifies the fixture
    mirrors reference_fp32's algebra exactly except for the one seeded defect. If this
    test ever fails, either the fixture grew a second bug or single-token decode
    stopped being scale-invariant, and the fixture's docstring is now a lie.
    """
    decode = next(s for s in _SHAPES if s.N == 1)
    kb = next(kb for kb in KNOWN_BAD if kb.name == "wrong_scale")

    res = _run_gate(kb, decode)
    assert res.passed and res.gate_failed is None, (
        f"wrong_scale @ {decode.key()}: expected the documented mathematical pass, "
        f"got gate={res.gate_failed}: {res.diagnostic}"
    )

    # The blind spot is one cell, not a hole: the matrix as a whole still rejects it.
    rejecting = next(s for s in _SHAPES if s.N > 1)
    assert not _run_gate(kb, rejecting).passed
