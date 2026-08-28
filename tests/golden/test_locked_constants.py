"""Tripwire on the locked gate constants and the shape-set contract.

This file is the reason the checksum manifest exists. The search loop's objective
rewards passing the gate, and a loop that can also move the bar for passing will move
the bar -- so the bar is pinned twice: once by the manifest over ratchet/oracle/, and
once here, where an agent that loosens a constant must also break a manifest-protected
golden test to hide it. Two loud failures instead of one silent one.

The comparisons are exact equality on purpose. There is no tolerance on a tolerance.
"""

from ratchet.oracle import ABS_TOL, BENCHMARK_SHAPES, CORRECTNESS_SHAPES, REL_TOL


def test_rel_tol_is_exactly_locked():
    # `type(...) is float` blocks the cute attack of substituting an object whose
    # __eq__ lies; the equality is bit-exact, not approximate.
    assert type(REL_TOL) is float
    assert REL_TOL == 0.02, (
        f"REL_TOL has been tampered with: {REL_TOL!r} != 0.02. This constant comes "
        f"from the competition statement and is not negotiable."
    )


def test_abs_tol_is_exactly_locked():
    assert type(ABS_TOL) is float
    assert ABS_TOL == 0.002, (
        f"ABS_TOL has been tampered with: {ABS_TOL!r} != 0.002. This constant comes "
        f"from the competition statement and is not negotiable."
    )


def test_correctness_and_benchmark_shapes_are_disjoint():
    """Tuning against the shapes you validate on is measuring your own tail.

    The disjointness assertion in inputs.py runs at import time; this one survives an
    edit that deletes that assertion, because this file is manifest-protected.
    """
    correctness = {s.key() for s in CORRECTNESS_SHAPES}
    benchmark = {s.key() for s in BENCHMARK_SHAPES}
    assert correctness, "CORRECTNESS_SHAPES is empty -- the gate checks nothing"
    assert benchmark, "BENCHMARK_SHAPES is empty -- there is nothing to measure"
    overlap = correctness & benchmark
    assert not overlap, (
        f"correctness and benchmark shape sets overlap: {sorted(overlap)}. "
        f"A shape may be validated on or timed on, never both."
    )
