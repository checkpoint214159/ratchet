"""v38: residency must be ATTEMPTED, and the attempt must be what decides.

The defect this candidate closes is one a passing accuracy suite never saw: v33's
dispatch reads `mem_get_info` at the first forward, which under `bench/run_matrix.py` is
the correctness check with the baseline model still resident, and latches "streamed" into
a timed phase where memory is plentiful. Config 6 -- 83% of the matrix's wall -- ran
1.6x slow on every candidate in the streaming lineage while every correctness test
passed.

So [L36] governs this whole file: **every assertion about an answer is preceded by an
assertion about which path produced it**, and each of the three routes into
`stream_path` (the signature-floor pre-check, the attempt, the OOM fallback) is asserted
by `stream_basis`, which `stream_path` alone cannot distinguish.

[L40]: the tests that matter here are the ones capable of failing. `test_v37_still_has_
the_defect_on_a_config_6_shape` asserts the PARENT's behaviour on purpose -- without it
this file would only show that v38 works, never that there was anything to fix.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.candidates import REGISTRY                                     # noqa: E402
from bench.candidates import v14_dispatch as D                            # noqa: E402
from bench.candidates import v38_stream_fallback as V                     # noqa: E402
from bench.matrix import MATRIX, BY_ID                                    # noqa: E402

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

RTOL, ATOL = 0.02, 0.002          # the harness's own defaults. Never widened.
CANDIDATE = "v38_stream_fallback"


def _reference(tag="ref_bench_t38"):
    path = REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def _build(ref, B, S, d, heads=4, layers=2, causal=True, name=CANDIDATE):
    cfg = ref.TransformerConfig(batch_size=B, seq_len=S, d_model=d, num_heads=heads,
                                ffn_dim=d, num_layers=layers, causal=causal)
    torch.manual_seed(1234)
    base = ref.BaselineTransformer(cfg)
    cand = REGISTRY[name].build(ref.BaselineTransformer)(cfg)
    ref.copy_model_weights(base, cand)
    return (cfg,
            base.to("cuda", torch.float32).eval(),
            cand.to("cuda", torch.float32).eval())


def _build_from_config(ref, cfg, name=CANDIDATE):
    return _build(ref, cfg.batch_size, cfg.seq_len, cfg.d_model, heads=cfg.heads,
                  layers=cfg.layers, causal=cfg.causal, name=name)


class _FakeMemInfo:
    """Report a chosen (free, total) from `torch.cuda.mem_get_info`, leaving the real
    allocator alone. Lets a test starve the PREDICATE without starving the DEVICE."""

    def __init__(self, free=None, total=None):
        self.free, self.total = free, total
        self._real = torch.cuda.mem_get_info

    def __enter__(self):
        real = self._real

        def fake(*a, **k):
            f, t = real(*a, **k)
            return (f if self.free is None else self.free,
                    t if self.total is None else self.total)

        torch.cuda.mem_get_info = fake
        return self

    def __exit__(self, *exc):
        torch.cuda.mem_get_info = self._real
        return False


# ======================================================================================
# Lineage and the predicate. No GPU.
# ======================================================================================

class TestLineage:
    def test_parent_is_v37(self):
        spec = REGISTRY[CANDIDATE]
        assert spec.parent == "v37_recombined2"
        assert spec.generation == 38

    def test_it_imports_the_predicates_rather_than_restating_them(self):
        # Two copies of a threshold drift apart; one copy cannot ([L14]).
        assert V.signature_floor_bytes is D.signature_floor_bytes
        assert V.estimate_working_set_bytes is D.estimate_working_set_bytes
        assert V.RESIDENT_BUDGET is D.RESIDENT_BUDGET

    def test_it_does_not_even_import_the_free_memory_predicate(self):
        """The whole point. `choose()` is what asked the wrong question at the wrong
        moment; if v38 still reached for it the fix would be cosmetic. The binding is
        checked here and the CALL is checked on the GPU, by making `choose` raise."""
        assert not hasattr(V, "choose")


class TestTheSignatureFloorIsExactAndNotAnEstimate:
    def test_it_is_the_two_tensors_the_signature_forces_and_nothing_else(self):
        assert D.signature_floor_bytes(32, 100000, 1024, 4) == 2 * 32 * 100000 * 1024 * 4
        # No coefficient: doubling any axis doubles it, exactly.
        assert (D.signature_floor_bytes(64, 128, 128, 4)
                == 2 * D.signature_floor_bytes(32, 128, 128, 4))

    def test_it_agrees_with_the_figure_the_report_quotes(self):
        """`bench/feasibility.py` states the same floor as impossibility 2. Two modules
        stating one number is exactly the drift [L14] warns about, so they are pinned
        equal here."""
        feas = pytest.importorskip("bench.feasibility",
                                   reason="feasibility.py is not on this branch")
        for c in MATRIX:
            assert (D.signature_floor_bytes(c.batch_size, c.seq_len, c.d_model, 4)
                    == feas.signature_floor_bytes(c.batch_size, c.seq_len, c.d_model, 4))

    def test_it_is_a_function_of_the_card_and_flips_with_it(self):
        """A predicate, not a lookup table: config 14 streams on a 16 GiB card and is
        attempted resident on an 80 GiB one, with no code change and no config id."""
        c14, c6 = BY_ID[14], BY_ID[6]
        f14 = D.signature_floor_bytes(c14.batch_size, c14.seq_len, c14.d_model, 4)
        f6 = D.signature_floor_bytes(c6.batch_size, c6.seq_len, c6.d_model, 4)
        assert f14 > 16 * 2**30 and f14 < 80 * 2**30
        assert f6 < 16 * 2**30


@cuda
class TestThePreCheckAgainstTHISCardsMeasuredMemory:
    def test_only_config_14_is_refused_a_resident_attempt(self):
        """The mechanism claim, evaluated against the memory the device REPORTS -- no
        mock, no constant, no config id in the predicate.

        Config 6 is the row that regressed; config 14 is the row that must keep
        streaming. Every other announced row must also be attempted, or the fix has
        bought config 6 at some other row's expense.
        """
        _free, total = torch.cuda.mem_get_info()
        refused = [c.id for c in MATRIX
                   if D.signature_floor_bytes(c.batch_size, c.seq_len, c.d_model, 4)
                   > total]
        assert refused == [14], (
            f"on a {total / 2**30:.2f} GiB card the floor refuses {refused}")

    def test_the_config_14_capability_path_is_still_attempted_resident(self):
        """`run_matrix`'s capability path calls the model ONE SEQUENCE AT A TIME, and
        every config-14 ledger row records `stream_path: resident, slice=1` there. That
        is where the causal-prefix oracle and the blocked fp64 certificate are produced,
        so the floor must not refuse it."""
        c = BY_ID[14]
        _free, total = torch.cuda.mem_get_info()
        assert D.signature_floor_bytes(1, c.seq_len, c.d_model, 4) < total
        assert D.signature_floor_bytes(c.batch_size, c.seq_len, c.d_model, 4) > total


# ======================================================================================
# The three routes into `stream_path`, each asserted by the mechanism that produced it.
# ======================================================================================

@cuda
class TestRouteOneTheFloorRefusesWithoutAttempting:
    def test_a_shape_past_the_floor_streams_and_never_touches_the_resident_path(self):
        ref = _reference()
        _cfg, base, cand = _build(ref, 64, 128, 128)
        x = torch.randn(64, 128, 128, device="cuda")
        m = torch.ones(64, 128, dtype=torch.bool, device="cuda")
        floor = D.signature_floor_bytes(64, 128, 128, 4)
        # Shrink the reported TOTAL below this shape's floor. The device is untouched --
        # only the property the predicate reads moves, which is the difference between a
        # predicate and a hardcoded table.
        with _FakeMemInfo(total=floor // 2):
            with torch.inference_mode():
                y = cand(x, m)
        assert cand.stream_path == "streamed", cand.stream_reason
        assert cand.stream_basis == "signature_floor", cand.stream_reason
        assert cand.stream_attempted_resident is False, (
            "the floor exists to avoid attempting a shape that cannot fit; it attempted")
        assert cand.stream_fallbacks == 0
        with torch.inference_mode():
            e = base(x, m)
        assert (y - e).abs().max().item() < ATOL
        assert y.shape == x.shape

    def test_no_path_through_a_forward_consults_the_free_memory_predicate(self):
        """The binding test above says v38 does not import `choose`; this says nothing
        it inherits calls it either. v33's layer holds a live reference to the same
        function, so poisoning only v38's namespace would prove nothing."""
        ref = _reference("ref_bench_t38_nochoose")
        from bench.candidates import v33_streamed_long as S

        def poisoned(*a, **k):
            raise AssertionError("the free-memory predicate was consulted")

        saved_d, saved_s = D.choose, S.choose
        D.choose = S.choose = poisoned
        try:
            _cfg, base, cand = _build(ref, 8, 128, 128)
            x = torch.randn(8, 128, 128, device="cuda")
            m = torch.ones(8, 128, dtype=torch.bool, device="cuda")
            with torch.inference_mode():
                y, e = cand(x, m), base(x, m)
        finally:
            D.choose, S.choose = saved_d, saved_s
        assert cand.stream_path == "resident"
        assert (y - e).abs().max().item() < ATOL

    def test_the_same_shape_under_the_real_card_is_attempted(self):
        """NEGATIVE CONTROL for the test above: without the shrunk total, this shape
        takes the resident path. A floor test that refused everything would pass the
        assertions above and mean nothing."""
        ref = _reference()
        _cfg, _base, cand = _build(ref, 64, 128, 128)
        x = torch.randn(64, 128, 128, device="cuda")
        m = torch.ones(64, 128, dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            cand(x, m)
        assert cand.stream_path == "resident", cand.stream_reason
        assert cand.stream_basis == "attempt"
        assert cand.stream_attempted_resident is True
        assert cand.stream_fallbacks == 0


@cuda
class TestRouteTwoTheAttemptSucceedsUnderMEMORYPRESSURE:
    """The defect itself, reproduced and closed.

    `_FakeMemInfo(free=...)` reproduces exactly what `run_matrix` does to the candidate:
    the device reports almost nothing free at the moment of the first forward, because
    the baseline arm has just been timed and the reference output is still live.
    """

    STARVED = 64 * 2**20         # 64 MiB reported free -- v33's predicate refuses at this

    def test_v38_runs_resident_when_the_device_reports_almost_nothing_free(self):
        ref = _reference()
        _cfg, base, cand = _build(ref, 64, 128, 128)
        x = torch.randn(64, 128, 128, device="cuda")
        m = torch.ones(64, 128, dtype=torch.bool, device="cuda")
        with _FakeMemInfo(free=self.STARVED):
            with torch.inference_mode():
                y = cand(x, m)
        assert cand.stream_path == "resident", cand.stream_reason
        assert cand.stream_basis == "attempt"
        assert cand.stream_fallbacks == 0, (
            "the shape fits; falling back means the attempt was not the thing deciding")
        with torch.inference_mode():
            e = base(x, m)
        assert (y - e).abs().max().item() < ATOL

    def test_v37_still_has_the_defect_at_the_same_reported_free_memory(self):
        """[L40]. The parent is asserted here on purpose: it is the proof that the fix is
        load-bearing rather than a restatement of behaviour that already existed."""
        ref = _reference()
        _cfg, _base, parent = _build(ref, 64, 128, 128, name="v37_recombined2")
        x = torch.randn(64, 128, 128, device="cuda")
        m = torch.ones(64, 128, dtype=torch.bool, device="cuda")
        with _FakeMemInfo(free=self.STARVED):
            with torch.inference_mode():
                parent(x, m)
        assert parent.stream_path == "streamed", (
            "the parent no longer streams under memory pressure; if that is a real "
            "change, this file and v38's docstring both need rewriting")

    def test_the_real_config_6_shape_takes_the_resident_path(self):
        """The row the whole candidate is about, at its announced shape, with the
        reported free memory starved the way the harness starves it.

        This asserts the PATH, not a time. Ranking is `bench/abba.py`'s job and a unit
        test may not conclude about speed ([L41], finding 45)."""
        c = BY_ID[6]
        ref = _reference("ref_bench_t38_c6")
        _cfg, _base, cand = _build_from_config(ref, c)
        x, m = ref.generate_random_case(
            ref.TransformerConfig(batch_size=c.batch_size, seq_len=c.seq_len,
                                  d_model=c.d_model, num_heads=c.heads,
                                  ffn_dim=c.ffn_dim, num_layers=c.layers,
                                  causal=c.causal),
            torch.device("cuda"), torch.float32, seed=1234,
            padding_ratio=0.0, input_scale=1.0)
        with _FakeMemInfo(free=self.STARVED):
            with torch.inference_mode():
                y = cand(x, m)
        assert cand.stream_path == "resident", cand.stream_reason
        assert cand.stream_fallbacks == 0, cand.stream_reason
        assert y.shape == x.shape


@cuda
class TestRouteThreeAnActualOOMFallsBack:
    """An injected `OutOfMemoryError`, because the real one is not reproducible on demand
    and a fallback nobody has watched fire is not a fallback ([L38])."""

    @staticmethod
    def _poison(cand, exc):
        """Make the layer BELOW v38's `_resident_forward` raise, once.

        Patched on the class v33's layer defines it on, so what is exercised is v38's own
        try/except and not a mock of it.
        """
        below = next(k for k in type(cand).__mro__[1:]
                     if "_resident_forward" in vars(k))
        real = below._resident_forward
        state = {"fired": False}

        def poisoned(self, x, mask):
            if not state["fired"]:
                state["fired"] = True
                raise exc
            return real(self, x, mask)

        below._resident_forward = poisoned
        return below, real, state

    def test_an_oom_is_caught_streamed_and_the_answer_survives_it(self):
        ref = _reference()
        _cfg, base, cand = _build(ref, 8, 256, 128)
        x = torch.randn(8, 256, 128, device="cuda")
        m = torch.ones(8, 256, dtype=torch.bool, device="cuda")
        below, real, state = self._poison(
            cand, torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate X"))
        try:
            with torch.inference_mode():
                y = cand(x, m)
                e = base(x, m)
        finally:
            below._resident_forward = real
        assert state["fired"], "the injection never ran; the test proves nothing"
        assert cand.stream_fallbacks == 1
        assert cand.stream_path == "streamed"
        assert cand.stream_basis == "oom_fallback", cand.stream_reason
        assert "OutOfMemoryError" in cand.stream_reason
        assert 1 <= cand.stream_slice < 8, (
            "a slice equal to the batch is the computation that just failed, not a "
            "smaller one")
        assert y.shape == x.shape
        assert (y - e).abs().max().item() < ATOL, "the fallback answered wrongly"

    def test_the_fallback_settles_every_shape_latched_decision_on_the_slice(self):
        """The streamed path never reaches v36's `forward`, so `_settle_slice_decisions`
        is the only place the attention tile, the FFN gate, the launch decision and the
        GEMM plan get made. If the fallback skipped it they would still read
        "undecided" and the slices would run on silent defaults."""
        ref = _reference()
        _cfg, _base, cand = _build(ref, 8, 256, 128)
        x = torch.randn(8, 256, 128, device="cuda")
        m = torch.ones(8, 256, dtype=torch.bool, device="cuda")
        below, real, _s = self._poison(cand, torch.cuda.OutOfMemoryError("oom"))
        try:
            with torch.inference_mode():
                cand(x, m)
        finally:
            below._resident_forward = real
        for attr in ("attn_reason", "fused_ffn_reason", "launch_reason", "gemm_reason"):
            assert getattr(cand, attr) != "undecided", f"{attr} was never settled"

    def test_a_non_oom_error_is_NOT_converted_into_a_slower_path(self):
        """The catch is narrow on purpose. A bare `except Exception` would turn every
        bug below this layer into a silent switch to a path that still returns an
        answer -- the silent-wrong-answer shape [L23] and [L25] catalogue."""
        ref = _reference()
        _cfg, _base, cand = _build(ref, 8, 256, 128)
        x = torch.randn(8, 256, 128, device="cuda")
        m = torch.ones(8, 256, dtype=torch.bool, device="cuda")
        below, real, _s = self._poison(cand, RuntimeError("a bug, not an OOM"))
        try:
            with pytest.raises(RuntimeError, match="a bug, not an OOM"):
                with torch.inference_mode():
                    cand(x, m)
        finally:
            below._resident_forward = real
        assert cand.stream_fallbacks == 0
        assert cand.stream_path == "resident"


@cuda
class TestARealAllocatorRefusal:
    """The injected `OutOfMemoryError` above proves the try/except. It does NOT prove the
    allocator can be recovered from, because an injected exception leaves the allocator
    tidy and a real refusal does not.

    `set_per_process_memory_fraction` makes the caching allocator refuse through its own
    code path at a budget chosen BETWEEN what a resident forward peaked at and what it
    was holding at rest. This is the test that caught the first draft of the fix being
    inert: it fell back, computed a slice equal to the batch, and OOMed again.
    """

    B, S, DM = 256, 512, 128

    def test_it_narrows_the_slice_until_the_shape_fits_and_answers_correctly(self):
        ref = _reference("ref_bench_t38_realoom")
        _cfg, base, warm = _build(ref, self.B, self.S, self.DM)
        x = torch.randn(self.B, self.S, self.DM, device="cuda")
        m = torch.ones(self.B, self.S, dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            expected = base(x, m)
        del base
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        with torch.inference_mode():
            warm(x, m)
            torch.cuda.synchronize()
        # What a RESIDENT forward needs ON TOP of what was already live.
        headroom = torch.cuda.max_memory_allocated() - before
        assert warm.stream_path == "resident", warm.stream_reason
        del warm
        torch.cuda.empty_cache()

        _c2, base2, cand = _build(ref, self.B, self.S, self.DM)
        del base2
        torch.cuda.empty_cache()
        # The budget is measured HERE, not earlier: `set_per_process_memory_fraction`
        # caps the process, and a pytest process carries whatever earlier tests left
        # allocated. A cap computed against a stale resting figure lands BELOW what is
        # already live and refuses every allocation, including the output tensor that no
        # slice size can shrink -- which is how this test first failed.
        live = torch.cuda.memory_allocated()
        _free, total = torch.cuda.mem_get_info()
        # The window is real but not wide. Below it, nothing fits -- the OUTPUT tensor is
        # `empty_like(x)` and no slice size shrinks it, so a budget that cannot hold the
        # output refuses the streamed path too and the retry loop correctly gives up.
        # Above it, residency succeeds and the test proves nothing. Both ends are
        # asserted below rather than assumed.
        cap = live + headroom * 0.65
        assert cap < live + headroom, "the cap must not admit a full resident forward"
        torch.cuda.set_per_process_memory_fraction(cap / total, 0)
        try:
            with torch.inference_mode():
                y = cand(x, m)
                torch.cuda.synchronize()
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, 0)

        assert cand.stream_fallbacks == 1, (
            "the allocator never refused; the cap did not bite and this test proves "
            "nothing")
        assert cand.stream_path == "streamed"
        assert cand.stream_basis == "oom_fallback"
        assert cand.stream_slice < self.B, cand.stream_reason
        assert y.shape == x.shape
        assert (y - expected).abs().max().item() < ATOL, (
            "the narrowed streamed path answered outside the locked tolerance")


# ======================================================================================
# What must NOT have changed.
# ======================================================================================

@cuda
class TestTheInheritedGuaranteesSurvive:
    def test_the_derived_shape_latch_reset_is_still_derived_and_still_covers_v35(self):
        """v37's own content, inherited. `SHAPE_LATCHED` is derived between v36 and v26,
        so it covers what v34 and v36 latch and deliberately NOT what the streaming
        layers declare -- `stream_slice` in particular must survive the reset, because
        the fallback and the narrowing both set it and then reset the state around it."""
        ref = _reference("ref_bench_t38_latch")
        cls = REGISTRY[CANDIDATE].build(ref.BaselineTransformer)
        latched = cls.SHAPE_LATCHED
        v35 = REGISTRY["v35_recombined"].build(ref.BaselineTransformer)
        assert set(latched) >= set(v35.SHAPE_LATCHED_BY_V34)
        for name in ("_tile_qkv", "_tile_out", "gemm_reason", "launch_reason"):
            assert name in latched, f"{name} dropped out of the derived reset set"
        for name in ("stream_slice", "stream_path", "stream_fallbacks"):
            assert name not in latched, (
                f"{name} entered the reset set; the fallback sets it and the reset would "
                f"then put it back to the class default")

    def test_a_smaller_second_batch_is_not_silently_broadcast(self):
        """v33's shape-latch fix, re-asserted at generation 38 against the same v26
        control. Warm at batch 8, then call at batch 1: v26 replays a graph latched to
        the first shape and hands back eight rows for a one-row input."""
        ref = _reference("ref_bench_t38_shape")
        shapes = {}
        for name in ("v26_causal_correct", CANDIDATE):
            torch._dynamo.reset()
            _cfg, base, cand = _build(ref, 8, 128, 128, name=name)
            x8 = torch.randn(8, 128, 128, device="cuda")
            m8 = torch.ones(8, 128, dtype=torch.bool, device="cuda")
            x1, m1 = x8[3:4].contiguous(), m8[3:4].contiguous()
            with torch.inference_mode():
                cand(x8, m8)
                try:
                    y = cand(x1, m1)
                    shapes[name] = tuple(y.shape)
                    if name == CANDIDATE:
                        assert (y - base(x1, m1)).abs().max().item() < ATOL
                except RuntimeError:
                    shapes[name] = "raised"
        assert shapes[CANDIDATE] == (1, 128, 128), shapes
        assert shapes["v26_causal_correct"] != (1, 128, 128), (
            "the control no longer exhibits the latch this lineage fixes")

    def test_a_non_causal_input_still_delegates_to_the_unmodified_baseline(self):
        """v26's guard (finding 32 / [L42]). The harness's OWN default is causal=False,
        so this is the path an unflagged grader run takes, and the new dispatch runs in
        front of it."""
        ref = _reference("ref_bench_t38_full")
        _cfg, base, cand = _build(ref, 8, 128, 128, causal=False)
        x = torch.randn(8, 128, 128, device="cuda")
        m = torch.ones(8, 128, dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            y, e = cand(x, m), base(x, m)
        assert (y - e).abs().max().item() < ATOL
        assert cand.causal_path.startswith("baseline")

    def test_it_answers_the_same_as_its_parent_where_both_run_resident(self):
        """No new kernel, so on a shape neither streams v38 must be v37's answer. If
        this ever moves, something changed that the docstring does not claim."""
        ref = _reference("ref_bench_t38_null")
        _c1, base, child = _build(ref, 16, 256, 128, name=CANDIDATE)
        _c2, _b2, parent = _build(ref, 16, 256, 128, name="v37_recombined2")
        x = torch.randn(16, 256, 128, device="cuda")
        m = torch.ones(16, 256, dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            yc, yp, e = child(x, m), parent(x, m), base(x, m)
        assert child.stream_path == "resident" and parent.stream_path == "resident"
        assert (yc - yp).abs().max().item() == 0.0, "same code, different answer"
        assert (yc - e).abs().max().item() < ATOL
