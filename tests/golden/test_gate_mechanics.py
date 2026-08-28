"""Synthetic-tensor pins on the gate machinery itself.

These tests hand the gate tensors constructed to fail exactly one bound at a time.
That distinction is the load-bearing one: the locked gate demands BOTH |err| <= ABS_TOL
and |err| <= REL_TOL * |exp| per element, which is stricter than numpy's
atol + rtol * |exp| form. The profitable tampering is to quietly switch to the weaker
OR/allclose semantics -- under which an element failing only the abs bound passes -- so
each bound gets a test where it is the only thing failing.
"""

import os

import torch

from ratchet.oracle import DeterministicContext, check_nonfinite, check_tolerance


def test_abs_bound_alone_is_caught():
    """|exp| = 1, error 0.005: rel err 0.5% is comfortably inside REL_TOL, abs err is
    2.5x ABS_TOL. Only the abs bound fails. allclose-style semantics would pass this."""
    exp = torch.ones(64, 64, device="cuda")
    got = exp.clone()
    got[3, 7] += 0.005
    res = check_tolerance(got, exp)
    assert not res.passed
    assert res.gate_failed == "tolerance"
    assert res.n_mismatched == 1
    assert "(3, 7)" in res.diagnostic, "diagnostic must name the offending index"


def test_rel_bound_alone_is_caught():
    """|exp| = 0.01, error 0.001: abs err is half of ABS_TOL, rel err is 10%. Only the
    rel bound fails."""
    exp = torch.full((64, 64), 0.01, device="cuda")
    got = exp.clone()
    got[2, 5] = 0.011
    res = check_tolerance(got, exp)
    assert not res.passed
    assert res.gate_failed == "tolerance"
    assert res.n_mismatched == 1
    assert "(2, 5)" in res.diagnostic


def test_passing_pair_reports_its_error_floor():
    """A pair inside both bounds passes, and the result still carries the max errors --
    the floor is data for the ledger even when nothing failed."""
    exp = torch.ones(32, 32, device="cuda")
    got = exp + 0.001          # abs err 0.001 <= 0.002, rel err 0.1% <= 2%
    res = check_tolerance(got, exp)
    assert res.passed
    assert res.gate_failed is None
    assert res.n_mismatched == 0
    assert res.n_total == 32 * 32
    assert 0.0009 < res.max_abs_err < 0.0011, "max_abs_err must report the real floor"
    assert 0.0009 < res.max_rel_err < 0.0011, "max_rel_err must report the real floor"


def test_shape_mismatch_is_its_own_gate():
    res = check_tolerance(torch.zeros(4, 4, device="cuda"),
                          torch.zeros(4, 5, device="cuda"))
    assert not res.passed
    assert res.gate_failed == "shape"


def test_swallowed_nan_is_caught():
    """Replacing a NaN the reference produced with an ordinary number is the single
    most common defect in machine-generated kernels, and allclose cannot see it."""
    exp = torch.randn(8, 8, device="cuda")
    exp[1, 2] = float("nan")
    got = exp.clone()
    got[1, 2] = 0.0
    ok, msg = check_nonfinite(got, exp)
    assert not ok
    assert "NaN" in msg and "swallowed" in msg, msg


def test_invented_inf_is_caught():
    """The symmetric defect: a non-finite the reference never produced."""
    exp = torch.randn(8, 8, device="cuda")
    got = exp.clone()
    got[4, 4] = float("inf")
    ok, msg = check_nonfinite(got, exp)
    assert not ok
    assert "Inf" in msg and "invented" in msg, msg


def test_matching_nonfinites_pass():
    """Propagation means the same non-finites in the same places is CORRECT -- the gate
    must accept it, or every kernel that propagates honestly gets punished for it."""
    exp = torch.randn(8, 8, device="cuda")
    exp[0, 0] = float("nan")
    exp[7, 7] = float("inf")
    ok, msg = check_nonfinite(exp.clone(), exp)
    assert ok and msg == ""


def test_deterministic_context_restores_state_on_exit():
    """DeterministicContext must pin the flags inside and put back EXACTLY what it
    found outside. A context that leaks its settings changes the arithmetic of
    everything that runs after it -- including the timing baselines."""
    saved = (torch.backends.cudnn.allow_tf32,
             torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.deterministic,
             torch.backends.cudnn.benchmark,
             os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    try:
        # A deliberately non-default starting state, so "restored" is distinguishable
        # from "left at the pinned values".
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

        with DeterministicContext():
            assert not torch.backends.cudnn.allow_tf32
            assert not torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.deterministic
            assert not torch.backends.cudnn.benchmark
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"

        assert torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    finally:
        (torch.backends.cudnn.allow_tf32,
         torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.deterministic,
         torch.backends.cudnn.benchmark) = saved[:4]
        if saved[4] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved[4]
