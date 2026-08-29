"""Test-only scalar near misses for the authoritative tolerance boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass

ATOL = 0.002
RTOL = 0.02


@dataclass(frozen=True, slots=True)
class NumericalNearMiss:
    case_id: str
    boundary: str
    reference: float
    observed: float
    expected_pass: bool
    rationale: str
    scope: str = "test_only"

    def __post_init__(self) -> None:
        if (
            not self.case_id
            or not self.boundary
            or not self.rationale
            or self.scope != "test_only"
            or not math.isfinite(self.reference)
            or not math.isfinite(self.observed)
        ):
            raise ValueError("numerical near miss must be finite and test-only")


NUMERICAL_NEAR_MISSES: tuple[NumericalNearMiss, ...] = (
    NumericalNearMiss(
        "SYN-NUM-001",
        "absolute_threshold_exactly",
        0.0,
        0.002,
        True,
        "absolute error equals atol at a zero reference",
    ),
    NumericalNearMiss(
        "SYN-NUM-002",
        "above_absolute_zero_relative",
        0.0,
        0.0020001,
        False,
        "absolute error exceeds atol and zero reference has no relative allowance",
    ),
    NumericalNearMiss(
        "SYN-NUM-003",
        "positive_relative_exactly",
        100.0,
        102.0,
        True,
        "relative error equals rtol times a positive reference",
    ),
    NumericalNearMiss(
        "SYN-NUM-004",
        "above_positive_relative",
        100.0,
        102.0001,
        False,
        "relative error exceeds rtol times a positive reference",
    ),
    NumericalNearMiss(
        "SYN-NUM-005",
        "negative_reference_abs_relative",
        -100.0,
        -102.0,
        True,
        "relative error equals rtol times a negative reference magnitude",
    ),
    NumericalNearMiss(
        "SYN-NUM-006",
        "additive_tolerance_trap",
        0.1,
        0.103,
        False,
        "the additive tolerance rule would pass but the required OR rule fails",
    ),
)


def passes_authoritative_or(reference: float, observed: float) -> bool:
    """Reproduce only the scalar finite absolute-OR-relative acceptance predicate."""

    if not math.isfinite(reference) or not math.isfinite(observed):
        return False
    absolute_error = abs(observed - reference)
    return absolute_error <= ATOL or absolute_error <= RTOL * abs(reference)
