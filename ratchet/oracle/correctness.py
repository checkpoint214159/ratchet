"""The correctness gate.

ZONE A -- IMMUTABLE. Do not edit as part of an optimization step.

The tolerances below are LOCKED CONSTANTS from the competition specification. They are
not parameters, not configurable, and not negotiable. If a candidate fails, the candidate
is wrong.

This is the single most important rule in the project. A loop whose objective rewards
passing, and which can also move the bar for passing, will move the bar. That is not a
hypothetical: one audit of 2,638 machine-generated kernels already accepted by their own
system found a majority carried at least one contract violation under adversarial gates,
the most common being silently replacing NaN/Inf with ordinary numbers.

The gate checks five things, not one. `torch.allclose` alone certified 9 of 9
deliberately seeded buggy kernels as correct in a published study.
"""

from __future__ import annotations

import os
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Optional

import torch

# ======================================================================================
# LOCKED. From the TechJam problem statement. Do not widen. Do not parameterize.
#
# Note which one binds: on order-1 outputs (attention output, normalized activations,
# post-softmax probabilities) ABS_TOL = 0.002 is effectively a 0.2% relative requirement,
# ten times tighter than REL_TOL. Check what your reference actually produces before
# assuming REL_TOL is the constraint.
# ======================================================================================
REL_TOL = 0.02
ABS_TOL = 0.002

_MAX_REPORTED_MISMATCHES = 8


@dataclass
class CorrectnessResult:
    passed: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    n_mismatched: int = 0
    n_total: int = 0
    nonfinite_ok: bool = True
    deterministic: bool = True
    gate_failed: Optional[str] = None      # which gate, for the ledger
    diagnostic: str = ""                   # human/agent readable, with indices
    per_distribution: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


class DeterministicContext(ContextDecorator):
    """Pin every source of run-to-run nondeterminism, and restore on exit.

    Without this, the determinism gate below produces false failures and the whole suite
    becomes flaky in a way that trains the loop to distrust its own tests.
    """

    def __enter__(self):
        self._tf32_cudnn = torch.backends.cudnn.allow_tf32
        self._tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        self._deterministic = torch.backends.cudnn.deterministic
        self._benchmark = torch.backends.cudnn.benchmark
        self._cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        torch.backends.cudnn.allow_tf32 = self._tf32_cudnn
        torch.backends.cuda.matmul.allow_tf32 = self._tf32_matmul
        torch.backends.cudnn.deterministic = self._deterministic
        torch.backends.cudnn.benchmark = self._benchmark
        if self._cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self._cublas
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        return False


def _format_mismatches(got: torch.Tensor, exp: torch.Tensor,
                       bad: torch.Tensor, limit: int) -> str:
    """Report the first N mismatches WITH INDICES.

    A bare boolean makes the agent feedback loop far weaker: the proposer cannot tell a
    tail-masking bug from a scale-factor bug from an accumulation-order bug. Indices and
    values usually make the class of error obvious at a glance.
    """
    idxs = torch.nonzero(bad, as_tuple=False)
    lines = []
    for i in range(min(limit, idxs.shape[0])):
        idx = tuple(int(v) for v in idxs[i])
        g = float(got[idx])
        e = float(exp[idx])
        lines.append(
            f"  {idx}: got {g:+.6e}  expected {e:+.6e}  "
            f"abs {abs(g - e):.3e}  rel {abs(g - e) / (abs(e) + 1e-30):.3e}"
        )
    if idxs.shape[0] > limit:
        lines.append(f"  ... and {idxs.shape[0] - limit} more")
    return "\n".join(lines)


def check_tolerance(got: torch.Tensor, exp: torch.Tensor) -> CorrectnessResult:
    """Gate 1: elementwise tolerance, both bounds, on the finite entries."""
    if got.shape != exp.shape:
        return CorrectnessResult(
            passed=False, gate_failed="shape",
            diagnostic=f"shape mismatch: got {tuple(got.shape)}, expected {tuple(exp.shape)}",
        )

    g = got.detach().to(torch.float64)
    e = exp.detach().to(torch.float64)
    finite = torch.isfinite(e) & torch.isfinite(g)

    abs_err = (g - e).abs()
    rel_err = abs_err / (e.abs() + 1e-30)

    # BOTH bounds must hold, which is stricter than numpy's atol + rtol*|exp| form.
    # That is deliberate: the competition states them as two separate requirements.
    bad = finite & ((abs_err > ABS_TOL) | (rel_err > REL_TOL))
    n_bad = int(bad.sum())

    res = CorrectnessResult(
        passed=n_bad == 0,
        max_abs_err=float(abs_err[finite].max()) if finite.any() else 0.0,
        max_rel_err=float(rel_err[finite].max()) if finite.any() else 0.0,
        n_mismatched=n_bad,
        n_total=int(finite.sum()),
    )
    if n_bad:
        res.gate_failed = "tolerance"
        res.diagnostic = (
            f"{n_bad}/{res.n_total} elements outside "
            f"(abs<={ABS_TOL}, rel<={REL_TOL}):\n"
            + _format_mismatches(g, e, bad, _MAX_REPORTED_MISMATCHES)
        )
    return res


def check_nonfinite(got: torch.Tensor, exp: torch.Tensor) -> tuple[bool, str]:
    """Gate 2: non-finite propagation.

    Where the reference is NaN or Inf, the candidate must be too. Silently replacing a
    non-finite with an ordinary number is the most common defect in machine-generated
    kernels and it is invisible to allclose, which skips non-finite entries.
    """
    exp_nan, got_nan = torch.isnan(exp), torch.isnan(got)
    exp_inf, got_inf = torch.isinf(exp), torch.isinf(got)

    if not torch.equal(exp_nan, got_nan):
        missing = int((exp_nan & ~got_nan).sum())
        spurious = int((~exp_nan & got_nan).sum())
        return False, (
            f"NaN propagation broken: {missing} NaNs swallowed, {spurious} invented"
        )
    if not torch.equal(exp_inf, got_inf):
        missing = int((exp_inf & ~got_inf).sum())
        spurious = int((~exp_inf & got_inf).sum())
        return False, (
            f"Inf propagation broken: {missing} Infs swallowed, {spurious} invented"
        )
    return True, ""


def check_determinism(fn, inputs, repeats: int = 3) -> tuple[bool, str]:
    """Gate 3: same input, same output.

    A kernel with a race condition frequently passes at low occupancy and fails under
    load. Running the same input several times is a cheap partial detector. If the kernel
    legitimately uses atomics, record the observed spread rather than asserting bitwise
    equality -- but record it, do not wave it away.
    """
    with torch.no_grad():
        first = fn(*inputs)
        first = first.clone() if isinstance(first, torch.Tensor) else first
        for i in range(repeats - 1):
            again = fn(*inputs)
            if not torch.equal(first, again):
                spread = float((first.to(torch.float64) - again.to(torch.float64)).abs().max())
                return False, (
                    f"nondeterministic across repeats (run 0 vs run {i + 1}); "
                    f"max divergence {spread:.3e}. If this kernel uses atomics, say so "
                    f"explicitly and record the bound; otherwise suspect a race."
                )
    return True, ""


def check_all(candidate_fn, reference_fn, input_sets: dict,
              *, determinism_repeats: int = 3) -> CorrectnessResult:
    """Run every gate over every input distribution. Fail closed.

    `input_sets` maps a distribution name to a tuple of input tensors. All four standard
    distributions must be present -- a candidate that passes only on N(0,1) inputs is a
    candidate that has learned the test, not the operation. See docs/04-failure-modes.md.
    """
    overall = CorrectnessResult(passed=True)

    for dist_name, inputs in input_sets.items():
        with torch.no_grad():
            expected = reference_fn(*inputs)
            try:
                got = candidate_fn(*inputs)
            except Exception as exc:
                overall.passed = False
                overall.gate_failed = "exception"
                overall.per_distribution[dist_name] = False
                overall.diagnostic = f"[{dist_name}] raised {type(exc).__name__}: {exc}"
                return overall

        res = check_tolerance(got, expected)
        overall.max_abs_err = max(overall.max_abs_err, res.max_abs_err)
        overall.max_rel_err = max(overall.max_rel_err, res.max_rel_err)
        overall.per_distribution[dist_name] = res.passed

        if not res.passed:
            overall.passed = False
            overall.gate_failed = res.gate_failed
            overall.diagnostic = f"[{dist_name}] {res.diagnostic}"
            return overall

        ok, msg = check_nonfinite(got, expected)
        if not ok:
            overall.passed = False
            overall.nonfinite_ok = False
            overall.gate_failed = "nonfinite"
            overall.diagnostic = f"[{dist_name}] {msg}"
            return overall

    # Determinism once, on the standard distribution -- it is the expensive gate.
    if "standard" in input_sets:
        ok, msg = check_determinism(candidate_fn, input_sets["standard"], determinism_repeats)
        if not ok:
            overall.passed = False
            overall.deterministic = False
            overall.gate_failed = "determinism"
            overall.diagnostic = msg

    return overall
