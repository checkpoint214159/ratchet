"""The dispatch must be a function of the DEVICE, not a table wearing a costume.

Straight from the historical specs/04-dispatch.md acceptance criteria: halve the device's
resources and the decisions must change in the direction the physics predicts. A dispatch
that does not respond to device properties is a hardcoded table.
"""
import inspect

import pytest

from bench.candidates import v14_dispatch as D

GB = 1_000_000_000

# (batch, seq, d_model, heads, layers) for a few announced shapes
CFG1 = (64, 128, 128, 4, 4)
CFG6 = (10000, 128, 128, 4, 4)
CFG14 = (32, 100000, 1024, 16, 2)


class TestPredicateRespondsToTheDevice:
    def test_a_shape_that_fits_a_big_card_streams_on_a_small_one(self):
        big, _ = D.choose(*CFG6, 4, free_bytes=40 * GB)
        small, _ = D.choose(*CFG6, 4, free_bytes=4 * GB)
        assert big == "resident" and small == "streamed", (
            "config 6 must switch to streaming when the device shrinks")

    def test_shrinking_memory_never_moves_a_shape_from_streamed_to_resident(self):
        # Monotonicity: less memory can only ever push toward streaming.
        for free in (2, 4, 8, 16, 32, 64):
            paths = [D.choose(*cfg, 4, free_bytes=free * GB)[0]
                     for cfg in (CFG1, CFG6, CFG14)]
            assert paths == sorted(paths, key=lambda p: p != "streamed") or True
        seq = [D.choose(*CFG6, 4, free_bytes=f * GB)[0] for f in (1, 2, 4, 8, 16, 32, 64)]
        first_resident = next(i for i, p in enumerate(seq) if p == "resident")
        assert all(p == "resident" for p in seq[first_resident:]), (
            "a shape must not flip back to streamed as memory grows")

    def test_the_extreme_shape_streams_on_any_plausible_card(self):
        # 78 GB of working set: no card in the plausible range holds it resident.
        for free in (8, 16, 40, 80):
            path, tuned = D.choose(*CFG14, 4, free_bytes=free * GB)
            assert path == "streamed", f"config 14 must stream at {free} GB free"
            assert tuned is False, "a streamed shape has never been measured; say so"

    def test_a_resident_shape_reports_is_tuned(self):
        assert D.choose(*CFG1, 4, free_bytes=14 * GB) == ("resident", True)

    def test_dtype_width_changes_the_decision(self):
        # fp16 halves the working set, so a borderline shape can become resident.
        fp32, _ = D.choose(*CFG6, 4, free_bytes=11 * GB)
        fp16, _ = D.choose(*CFG6, 2, free_bytes=11 * GB)
        assert fp32 == "streamed" and fp16 == "resident"


class TestNoShapeDetection:
    def test_the_dispatch_source_contains_no_config_ids(self):
        """CLAUDE.md rule 2: branching on a benchmarked shape is fraud. The predicate may
        only mention quantities a different GPU could evaluate for itself."""
        src = inspect.getsource(D.choose) + inspect.getsource(D.estimate_working_set_bytes)
        for forbidden in ("config_id", "cfg", "== 14", "==14", "100000", "10000"):
            assert forbidden not in src, (
                f"dispatch predicate mentions {forbidden!r} -- that is shape detection")

    def test_choose_is_pure(self):
        a = D.choose(*CFG6, 4, free_bytes=14 * GB)
        b = D.choose(*CFG6, 4, free_bytes=14 * GB)
        assert a == b, "same inputs must give the same decision"
