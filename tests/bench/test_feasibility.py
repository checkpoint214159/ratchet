"""The config-14 protocol: an infeasibility claim and an oracle, both made falsifiable.

Two of this repo's recurring failures are in scope here and the tests are shaped around
them. [L38]: a check nobody has watched fail is not a check, so every oracle assertion is
paired with a negative control that must trip it. [L24]: "correct because of how the
harness calls it" is not correct, so the oracle is validated against the reference at
sequence lengths where the reference genuinely runs, not against our own candidates.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from bench import feasibility as FZ
from bench.matrix import BY_ID

REPO = Path(__file__).resolve().parents[2]
GIB = 2**30
TIB = 2**40


def _reference():
    spec = importlib.util.spec_from_file_location(
        "ref_bench_feas", REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench_feas"] = m
    spec.loader.exec_module(m)
    return m


# ======================================================================================
# Impossibility 1 -- the reference's algorithm
# ======================================================================================

class TestReferenceRequirementIsDerivedNotGuessed:
    def test_config_14_score_tensor_is_18_63_TiB(self):
        c = BY_ID[14]
        req = FZ.reference_peak_bytes(c.batch_size, c.seq_len, c.d_model, c.heads, 4)
        assert req.scores_bytes == c.dense_scores_bytes(4), (
            "the requirement must agree with matrix.py, which is the single source of "
            "truth for the shapes")
        assert 18.6 < req.scores_bytes / TIB < 18.7

    def test_even_one_head_of_one_sequence_does_not_fit_a_16_GiB_card(self):
        # The claim that carries the report: this is not a batch-size problem.
        req = FZ.reference_peak_bytes(1, 100000, 1024, 1, 4)
        assert req.scores_bytes / GIB > 37.0
        ok, _ = FZ.reference_feasible(1, 100000, 1024, 1, 4, device_bytes=16 * GIB)
        assert ok is False

    def test_the_thirteen_runnable_configs_are_reported_feasible(self):
        # NEGATIVE CONTROL for the predicate: if it called everything infeasible it would
        # be trivially "right" about config 14 and useless.
        for c in BY_ID.values():
            ok, why = FZ.reference_feasible(c.batch_size, c.seq_len, c.d_model, c.heads,
                                            4, device_bytes=16 * GIB)
            assert ok is (c.id != 14), f"config {c.id}: {why}"

    def test_the_predicate_responds_to_the_device_not_to_a_config_id(self):
        c = BY_ID[13]                       # seq 1024: fits a real card, not a tiny one
        big, _ = FZ.reference_feasible(c.batch_size, c.seq_len, c.d_model, c.heads, 4,
                                       device_bytes=80 * GIB)
        small, _ = FZ.reference_feasible(c.batch_size, c.seq_len, c.d_model, c.heads, 4,
                                         device_bytes=GIB // 2)
        assert big is True and small is False, (
            "config 13 fits an 80 GiB card and not a 0.5 GiB one; a predicate that "
            "answers the same either way is a lookup table")

    def test_the_predicate_carries_no_config_id_and_no_id_parameter(self):
        """Structural, not prose: prose says the rule, this makes it capable of failing.

        Comments and docstrings may discuss config 14; the executable code may not
        branch on it. Checked against the compiled constants and the signature, which
        is what actually runs.
        """
        import inspect
        for fn in (FZ.reference_feasible, FZ.reference_peak_bytes,
                   FZ.signature_floor_bytes):
            params = set(inspect.signature(fn).parameters)
            assert not (params & {"config_id", "config", "cfg", "id"}), (
                f"{fn.__name__} takes a config, not shapes")
            consts = [c for c in fn.__code__.co_consts if isinstance(c, int)]
            assert 14 not in consts and 100000 not in consts, (
                f"{fn.__name__} carries a config-14 literal (CLAUDE.md rule 2)")


# ======================================================================================
# Impossibility 2 -- the signature's own floor
# ======================================================================================

class TestSignatureFloor:
    def test_config_14_needs_24_42_GiB_of_tensors_before_any_arithmetic(self):
        c = BY_ID[14]
        floor = FZ.signature_floor_bytes(c.batch_size, c.seq_len, c.d_model, 4)
        assert 24.4 < floor / GIB < 24.5
        assert floor > 16 * GIB, "this is the claim: the floor alone exceeds this card"

    def test_the_floor_is_exactly_two_activations(self):
        c = BY_ID[6]
        assert FZ.signature_floor_bytes(c.batch_size, c.seq_len, c.d_model, 4) == \
            2 * c.activation_bytes(4)


# ======================================================================================
# Oracle A -- the causal-prefix theorem
# ======================================================================================

class TestCausalPrefixAvailability:
    def test_requires_causality(self):
        import torch
        m = torch.ones(2, 8, dtype=torch.bool)
        assert FZ.causal_prefix_holds(True, m) is True
        assert FZ.causal_prefix_holds(False, m) is False

    def test_requires_every_token_valid(self):
        import torch
        m = torch.ones(2, 8, dtype=torch.bool)
        m[0, -1] = False
        assert FZ.causal_prefix_holds(True, m) is False
        assert FZ.causal_prefix_holds(True, None) is True


@pytest.mark.gpu
class TestCausalPrefixTheoremOnTheReference:
    """The theorem is the load-bearing part of the protocol: it makes the UNMODIFIED
    reference an oracle at the real sequence length for the first P rows."""

    def test_prefix_matches_and_suffix_does_not(self):
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        ref = _reference()
        S, P = 2048, 256
        cfg = ref.TransformerConfig(batch_size=1, seq_len=S, d_model=256, num_heads=4,
                                    ffn_dim=256, num_layers=2, causal=True)
        torch.manual_seed(1234)
        m = ref.BaselineTransformer(cfg).to("cuda", torch.float32).eval()
        g = torch.Generator(device="cuda")
        g.manual_seed(7)
        x = torch.randn(1, S, 256, generator=g, device="cuda")
        msk = torch.ones(1, S, device="cuda", dtype=torch.bool)
        with torch.inference_mode():
            full = m(x, msk)
            m.config = ref.TransformerConfig(batch_size=1, seq_len=P, d_model=256,
                                             num_heads=4, ffn_dim=256, num_layers=2,
                                             causal=True)
            pref = m(x[:, :P].contiguous(), msk[:, :P].contiguous())
            suff = m(x[:, -P:].contiguous(), msk[:, :P].contiguous())
        assert (full[:, :P] - pref).abs().max().item() < 2e-3, (
            "the prefix theorem must hold inside the locked tolerance")
        # NEGATIVE CONTROL. A suffix is NOT a closed computation under causality, so if
        # this also matched, the first assertion would be measuring nothing.
        assert (full[:, -P:] - suff).abs().max().item() > 1e-1


# ======================================================================================
# Oracle B -- the blocked fp64 evaluation of the reference's own arithmetic
# ======================================================================================

@pytest.mark.gpu
class TestBlockedFp64Oracle:
    @staticmethod
    def _setup(S=512, d=256, heads=4, causal=True):
        import torch
        ref = _reference()
        cfg = ref.TransformerConfig(batch_size=1, seq_len=S, d_model=d, num_heads=heads,
                                    ffn_dim=d, num_layers=2, causal=causal)
        torch.manual_seed(1234)
        m = ref.BaselineTransformer(cfg).to("cuda", torch.float32).eval()
        g = torch.Generator(device="cuda")
        g.manual_seed(5)
        x = torch.randn(1, S, d, generator=g, device="cuda")
        msk = torch.ones(1, S, device="cuda", dtype=torch.bool)
        return ref, m, x, msk

    def test_agrees_with_the_reference_in_strict_fp32(self):
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            _ref, m, x, msk = self._setup()
            with torch.inference_mode():
                y = m(x, msk)
                o = FZ.blocked_reference_forward(m, x, msk, causal=True, q_block=128)
            gap = (y.double() - o).abs().max().item()
            # The oracle is a stand-in for the reference only if this is negligible
            # against the locked 2e-3 -- 0.5% of budget, not 40%.
            assert gap < 1e-5, f"oracle disagrees with the reference by {gap:.3e}"
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
            torch.set_float32_matmul_precision("high")

    def test_query_blocking_changes_nothing(self):
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        _ref, m, x, msk = self._setup()
        with torch.inference_mode():
            a = FZ.blocked_reference_forward(m, x, msk, causal=True, q_block=64)
            b = FZ.blocked_reference_forward(m, x, msk, causal=True, q_block=512)
        assert (a - b).abs().max().item() < 1e-12, (
            "blocking the query axis must be exact; softmax reduces over keys")

    def test_a_perturbed_weight_is_detected(self):
        # NEGATIVE CONTROL. Without this the agreement test could be passing because the
        # oracle secretly reproduces whatever it is handed.
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        _ref, m, x, msk = self._setup()
        with torch.inference_mode():
            good = FZ.blocked_reference_forward(m, x, msk, causal=True, q_block=128)
            m.layers[0].ffn_in.bias[0] += 0.05
            bad = FZ.blocked_reference_forward(m, x, msk, causal=True, q_block=128)
        assert (good - bad).abs().max().item() > 2e-3

    def test_the_causal_flag_is_honoured(self):
        # [L42]: the harness's own default is causal=False, so the oracle must not
        # quietly assume the announced matrix's setting.
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        _ref, m, x, msk = self._setup()
        with torch.inference_mode():
            c = FZ.blocked_reference_forward(m, x, msk, causal=True, q_block=128)
            n = FZ.blocked_reference_forward(m, x, msk, causal=False, q_block=128)
        assert (c - n).abs().max().item() > 1e-1

    def test_peak_memory_is_linear_in_sequence_not_quadratic(self):
        # The whole reason the oracle exists: it must not build an [S, S] tensor.
        import torch
        torch.cuda.is_available() or pytest.skip("no CUDA")
        peaks = []
        for S in (1024, 4096):
            _ref, m, x, msk = self._setup(S=S)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                FZ.blocked_reference_forward(m, x, msk, causal=True, q_block=256)
            peaks.append(torch.cuda.max_memory_allocated())
            del m, x, msk
            torch.cuda.empty_cache()
        # 4x the sequence must cost far less than the 16x a materialised score matrix
        # would; allow generous slack for the fixed weight footprint.
        assert peaks[1] < 8 * peaks[0], f"peaks {peaks} look quadratic in S"
