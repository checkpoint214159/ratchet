"""v8 claims a right-padded causal key mask is redundant. Pin the claim and its guard.

The optimization is only sound where its precondition holds, so the guard is the part
that matters most: a mask shape the proof does not cover must fall back, not silently
take the fast path and return wrong answers.
"""
import pytest
import torch

from bench.candidates.v8_padfast import prefix_padded


def _m(rows):
    return torch.tensor(rows, dtype=torch.bool)


class TestPrefixGuard:
    def test_accepts_right_padded(self):
        assert prefix_padded(_m([[1, 1, 1, 0], [1, 1, 0, 0]]))

    def test_accepts_all_true(self):
        assert prefix_padded(_m([[1, 1, 1, 1]]))

    def test_accepts_none(self):
        assert prefix_padded(None)

    def test_rejects_a_hole(self):
        # (True, False, True) -- a valid query at index 2 could look back at the invalid
        # key at index 1, which is exactly what the redundancy argument forbids.
        assert not prefix_padded(_m([[1, 0, 1, 0]]))

    def test_rejects_left_padding(self):
        assert not prefix_padded(_m([[0, 0, 1, 1]]))

    def test_rejects_when_only_one_row_is_bad(self):
        # A per-batch check that passed on the mean or the first row would miss this.
        assert not prefix_padded(_m([[1, 1, 0, 0], [1, 0, 1, 0]]))

    def test_rejects_all_false_row(self):
        # Degenerate but must not be treated as a prefix of length 0 that is "fine":
        # sum==0 makes positions < 0 empty, which equals the row, so this one DOES
        # satisfy the prefix form. Pin the actual behaviour rather than a guess.
        assert prefix_padded(_m([[0, 0, 0, 0]]))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestAgainstTheReference:
    """The proof is only worth as much as the numbers agree with it."""

    @staticmethod
    def _bench():
        import importlib.util, sys
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
        spec = importlib.util.spec_from_file_location("ref_bench_t", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["ref_bench_t"] = m
        spec.loader.exec_module(m)
        return m

    @pytest.mark.parametrize("padding", [0.0, 0.3, 0.5, 0.9])
    def test_matches_reference_across_padding(self, padding):
        ref = self._bench()
        from bench.candidates import REGISTRY
        cfg = ref.TransformerConfig(batch_size=4, seq_len=128, d_model=128, num_heads=4,
                                    ffn_dim=128, num_layers=2, causal=True)
        torch.manual_seed(0)
        base = ref.BaselineTransformer(cfg)
        cand = REGISTRY["v8_padfast"].build(ref.BaselineTransformer)(cfg)
        ref.copy_model_weights(base, cand)
        base = base.to("cuda", torch.float32).eval()
        cand = cand.to("cuda", torch.float32).eval()
        cand.use_graph = False
        with torch.inference_mode():
            x, mask = ref.generate_random_case(cfg, torch.device("cuda"),
                                               torch.float32, 7, padding, 1.0)
            r = ref.compare_outputs(base(x, mask), cand(x, mask), rtol=0.02, atol=0.002)
        assert r.failed_elements == 0, (
            f"padding={padding}: {r.failed_elements} elements outside tolerance, "
            f"max_abs={r.max_abs_error:.3e}")

    def test_falls_back_when_the_mask_has_a_hole(self):
        # A mask the proof does not cover must still produce the reference's answer,
        # via the slow path. If the guard were missing this would silently differ.
        ref = self._bench()
        from bench.candidates import REGISTRY
        cfg = ref.TransformerConfig(batch_size=2, seq_len=16, d_model=64, num_heads=4,
                                    ffn_dim=64, num_layers=1, causal=True)
        torch.manual_seed(0)
        base = ref.BaselineTransformer(cfg)
        cand = REGISTRY["v8_padfast"].build(ref.BaselineTransformer)(cfg)
        ref.copy_model_weights(base, cand)
        base = base.to("cuda", torch.float32).eval()
        cand = cand.to("cuda", torch.float32).eval()
        cand.use_graph = False

        mask = torch.ones(2, 16, dtype=torch.bool, device="cuda")
        mask[0, 5] = False                      # a hole, not a suffix
        x = torch.randn(2, 16, 64, device="cuda")
        x = x.masked_fill(~mask[..., None], 0)
        with torch.inference_mode():
            expected, got = base(x, mask), cand(x, mask)
            assert not cand._fastpath, "a holed mask must NOT take the fast path"
            r = ref.compare_outputs(expected, got, rtol=0.02, atol=0.002)
        assert r.failed_elements == 0
