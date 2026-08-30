"""v33 must actually stream, and streaming must be exact.

[L36] is the shape of the trap here: a streaming test is worthless if the candidate
quietly took the resident path, because then it asserts that v26 equals v26. Every
equivalence assertion below is preceded by an assertion that the mechanism engaged, and
is paired with a case where it must NOT engage.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from bench.candidates import REGISTRY, v14_dispatch as D

REPO = Path(__file__).resolve().parents[2]
GB = 1_000_000_000


def _reference():
    spec = importlib.util.spec_from_file_location(
        "ref_bench_v33", REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench_v33"] = m
    spec.loader.exec_module(m)
    return m


class TestLineage:
    def test_parent_is_the_frontier(self):
        assert REGISTRY["v33_streamed_long"].parent == "v26_causal_correct"

    def test_it_reuses_v14s_predicate_rather_than_restating_it(self):
        # Two copies of a threshold drift apart; one copy cannot.
        from bench.candidates import v33_streamed_long as V
        assert V.choose is D.choose
        assert V.estimate_working_set_bytes is D.estimate_working_set_bytes
        assert V.RESIDENT_BUDGET == D.RESIDENT_BUDGET


def _build(ref, B, S, d, heads=4, layers=2, causal=True, name="v33_streamed_long"):
    import torch
    cfg = ref.TransformerConfig(batch_size=B, seq_len=S, d_model=d, num_heads=heads,
                                ffn_dim=d, num_layers=layers, causal=causal)
    torch.manual_seed(1234)
    base = ref.BaselineTransformer(cfg)
    cand = REGISTRY[name].build(ref.BaselineTransformer)(cfg)
    ref.copy_model_weights(base, cand)
    return cfg, base.to("cuda", torch.float32).eval(), cand.to("cuda", torch.float32).eval()


@pytest.mark.gpu
class TestTheMechanismEngagesAndOnlyWhenItShould:
    def test_a_shape_that_fits_takes_the_resident_path(self):
        """NEGATIVE CONTROL for every test below it. If v33 streamed unconditionally it
        would pass the equivalence tests and silently cost the other 13 configs their
        CUDA graph."""
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        _cfg, _b, cand = _build(ref, 8, 128, 128)
        x = torch.randn(8, 128, 128, device="cuda")
        m = torch.ones(8, 128, dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            cand(x, m)
        assert cand.stream_path == "resident", cand.stream_reason

    def test_a_shape_that_does_not_fit_the_reported_free_memory_streams(self):
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        _cfg, _b, cand = _build(ref, 8, 128, 128)
        x = torch.randn(8, 128, 128, device="cuda")
        m = torch.ones(8, 128, dtype=torch.bool, device="cuda")
        real = torch.cuda.mem_get_info
        # The predicate reads the device. Shrink what the device reports and the decision
        # must flip -- that is the difference between a predicate and a lookup table.
        torch.cuda.mem_get_info = lambda *a, **k: (1_000_000, real()[1])
        try:
            with torch.inference_mode():
                cand(x, m)
        finally:
            torch.cuda.mem_get_info = real
        assert cand.stream_path == "streamed", cand.stream_reason
        assert cand.stream_slice < 8, "streaming with a full-batch slice is not streaming"

    def test_slicing_the_batch_does_not_move_us_away_from_the_reference(self):
        """Slicing the batch is exact IN EXACT ARITHMETIC -- attention is within-sequence,
        the token mask is per-sequence, everything else is position-wise. It is not
        bitwise exact in floating point, and the reason is not ours: the batch axis is a
        GEMM's M dimension, so cuBLAS picks a different tiling and reduces in a different
        order. Measured, the REFERENCE itself moves 3.46e-4 when its own batch is sliced
        the same way.

        So the assertion that means something is not a threshold on the slicing gap. It
        is that streaming does not move the candidate FURTHER FROM THE REFERENCE, which
        is the only distance the grader ever measures.
        """
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        _cfg, base, cand = _build(ref, 8, 256, 128)
        x = torch.randn(8, 256, 128, device="cuda")
        m = torch.ones(8, 256, dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            expected = base(x, m)
            base_sliced = torch.cat([base(x[i:i + 1], m[i:i + 1]) for i in range(8)])
            cand._prime(m)
            cand._decide_attn(x[:1])
            cand._decide_ffn(x[:1])
            whole = cand._core(x, m)
            sliced = torch.cat([cand._core(x[i:i + 1], m[i:i + 1]) for i in range(8)])
        d_whole = (whole - expected).abs().max().item()
        d_sliced = (sliced - expected).abs().max().item()
        d_ref = (base_sliced - expected).abs().max().item()
        assert d_sliced < 2e-3, f"sliced candidate is {d_sliced:.3e} from the reference"
        assert d_sliced <= d_whole + d_ref, (
            f"streaming moved us away from the reference: {d_whole:.3e} -> {d_sliced:.3e} "
            f"(the reference's own slicing noise is {d_ref:.3e})")
        # NEGATIVE CONTROL: the comparison must be capable of seeing a real change.
        assert (whole - torch.roll(sliced, 1, dims=0)).abs().max().item() > 1e-1

    def test_streamed_matches_resident_inside_the_locked_tolerance(self):
        """End to end. The residual here is NOT slicing -- the test above pins that at
        fp32 noise. It is Inductor's fused kernels against the same ops run eagerly, on a
        candidate whose intermediates are fp16, and it is the same order of magnitude as
        this lineage's error against the reference on every other config."""
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        _cfg, _b, resident = _build(ref, 8, 256, 128)
        _cfg2, _b2, streamed = _build(ref, 8, 256, 128)
        x = torch.randn(8, 256, 128, device="cuda")
        m = torch.ones(8, 256, dtype=torch.bool, device="cuda")
        real = torch.cuda.mem_get_info
        torch.cuda.mem_get_info = lambda *a, **k: (1_000_000, real()[1])
        try:
            with torch.inference_mode():
                ys = streamed(x, m)
        finally:
            torch.cuda.mem_get_info = real
        with torch.inference_mode():
            yr = resident(x, m)
        assert streamed.stream_path == "streamed" and resident.stream_path == "resident"
        gap = (ys - yr).abs().max().item()
        assert gap < 2e-3, f"streaming moved the answer by {gap:.3e}, past the locked atol"

    def test_streaming_still_delegates_a_non_causal_input_to_the_baseline(self):
        # v26's guard (finding 32 / [L42]) must survive the new dispatch. The harness's
        # OWN default is causal=False, so this is the path an unflagged grader run takes.
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        _cfg, base, cand = _build(ref, 8, 128, 128, causal=False)
        x = torch.randn(8, 128, 128, device="cuda")
        m = torch.ones(8, 128, dtype=torch.bool, device="cuda")
        real = torch.cuda.mem_get_info
        torch.cuda.mem_get_info = lambda *a, **k: (1_000_000, real()[1])
        try:
            with torch.inference_mode():
                y = cand(x, m)
                e = base(x, m)
        finally:
            torch.cuda.mem_get_info = real
        assert (y - e).abs().max().item() < 2e-3
        assert cand.causal_path.startswith("baseline")

    def test_a_second_call_at_a_different_batch_size_is_re_decided(self):
        """REGRESSION. v14 latched the dispatch on the first forward and v13's graph
        capture latches to the first shape, so a model warmed at batch 1 and then called
        with the whole batch raised

            output with shape [1, S, D] doesn't match the broadcast shape [B, S, D]

        from inside `_static_x.copy_(x)`. Found by the config-14 capability path, not by
        any test -- because every sweep builds a fresh model per config, so nothing had
        ever called one model at two shapes. [L24] exactly: correct only under the call
        pattern we happen to use.
        """
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        _cfg, base, cand = _build(ref, 8, 128, 128)
        m1 = torch.ones(1, 128, dtype=torch.bool, device="cuda")
        m8 = torch.ones(8, 128, dtype=torch.bool, device="cuda")
        x1 = torch.randn(1, 128, 128, device="cuda")
        x8 = torch.randn(8, 128, 128, device="cuda")
        with torch.inference_mode():
            cand(x1, m1)                       # warm and latch at batch 1
            assert cand.stream_path == "resident"
            y8 = cand(x8, m8)                  # ... then the shape a grader calls
            e8 = base(x8, m8)
        assert y8.shape == x8.shape
        assert (y8 - e8).abs().max().item() < 2e-3, "re-decided path gave a wrong answer"

    def test_a_smaller_second_batch_is_not_silently_broadcast(self):
        """The defect in its dangerous direction, and the proof the fix is load-bearing.

        Warm at batch 8, then call with batch 1. v13's `_static_x.copy_(x)` BROADCASTS
        the smaller input across the static buffer, replays the graph, and returns eight
        rows. Measured on the parent: v26 hands back shape (8, 128, 128) for a
        (1, 128, 128) input. v33 returns (1, 128, 128).

        The parent is asserted here on purpose. Without it this file would only show that
        v33 works, never that there was anything to fix -- and an assurance nobody
        arranged to be capable of failing is [L40]'s recurring complaint.
        """
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        shapes = {}
        for name in ("v26_causal_correct", "v33_streamed_long"):
            torch._dynamo.reset()                       # [L36]: 8 compiles per process
            _cfg, base, cand = _build(ref, 8, 128, 128, name=name)
            x8 = torch.randn(8, 128, 128, device="cuda")
            m8 = torch.ones(8, 128, dtype=torch.bool, device="cuda")
            x1, m1 = x8[3:4].contiguous(), m8[3:4].contiguous()
            with torch.inference_mode():
                cand(x8, m8)                            # warm and latch at batch 8
                try:
                    y = cand(x1, m1)
                    shapes[name] = tuple(y.shape)
                    if name == "v33_streamed_long":
                        e = base(x1, m1)
                        assert (y - e).abs().max().item() < 2e-3
                except RuntimeError:
                    shapes[name] = "raised"
        assert shapes["v33_streamed_long"] == (1, 128, 128), shapes
        assert shapes["v26_causal_correct"] != (1, 128, 128), (
            "the parent no longer exhibits the shape latch this candidate fixes; if that "
            "is a real change, this test and v33's docstring both need rewriting")

    def test_the_slice_shrinks_as_reported_free_memory_shrinks(self):
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        slices = []
        for free in (30_000_000, 6_000_000):
            _cfg, _b, cand = _build(ref, 16, 256, 128)
            x = torch.randn(16, 256, 128, device="cuda")
            m = torch.ones(16, 256, dtype=torch.bool, device="cuda")
            real = torch.cuda.mem_get_info
            torch.cuda.mem_get_info = lambda *a, **k: (free, real()[1])
            try:
                with torch.inference_mode():
                    cand(x, m)
            finally:
                torch.cuda.mem_get_info = real
            assert cand.stream_path == "streamed", cand.stream_reason
            slices.append(cand.stream_slice)
        assert slices[0] > slices[1], f"slice did not respond to free memory: {slices}"
