"""v13 claims its graph is either verified-real or unused. Pin both halves.

The failure this guards against is silence: an empty CUDA graph replays as a no-op and
returns the static output buffer unchanged -- right shape, right dtype, stale values.
"""
import pytest
import torch


def _bench():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v13", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v13"] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestSafeCapture:
    @staticmethod
    def _build(cfg_kw=None):
        ref = _bench()
        from bench.candidates import REGISTRY
        kw = dict(batch_size=4, seq_len=64, d_model=64, num_heads=4,
                  ffn_dim=64, num_layers=2, causal=True)
        kw.update(cfg_kw or {})
        cfg = ref.TransformerConfig(**kw)
        torch.manual_seed(0)
        base = ref.BaselineTransformer(cfg)
        cand = REGISTRY["v13_safe_capture"].build(ref.BaselineTransformer)(cfg)
        ref.copy_model_weights(base, cand)
        return ref, cfg, base.to("cuda", torch.float32).eval(), cand.to("cuda", torch.float32).eval()

    def test_output_tracks_the_input_it_was_given(self):
        """The direct test for a stale buffer: two different inputs must not agree."""
        ref, cfg, base, cand = self._build()
        x1, m = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 1, 0.0, 1.0)
        x2, _ = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 2, 0.0, 1.0)
        with torch.inference_mode():
            y1 = cand(x1, m).clone()
            y2 = cand(x2, m).clone()
            assert not torch.allclose(y1, y2), \
                "identical output for different inputs -- the graph is replaying stale state"
            assert torch.allclose(y1, cand(x1, m), rtol=1e-3, atol=1e-4), \
                "re-running the first input must reproduce the first output"

    def test_matches_the_reference_whether_or_not_capture_succeeded(self):
        ref, cfg, base, cand = self._build()
        with torch.inference_mode():
            for seed in (3, 4, 5):
                x, m = ref.generate_random_case(cfg, torch.device("cuda"),
                                                torch.float32, seed, 0.0, 1.0)
                r = ref.compare_outputs(base(x, m), cand(x, m), rtol=0.02, atol=0.002)
                assert r.failed_elements == 0, f"seed {seed}: {r.failed_elements} bad"

    def test_capture_is_attempted_exactly_once(self):
        """A failed capture must not be retried on every call -- that would cost more
        than the graph ever saved."""
        ref, cfg, base, cand = self._build()
        x, m = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 6, 0.0, 1.0)
        with torch.inference_mode():
            cand(x, m)
            assert cand._capture_attempted is True
            attempts = []
            orig = cand._try_capture
            cand._try_capture = lambda *a, **k: (attempts.append(1), orig(*a, **k))[1]
            for _ in range(5):
                cand(x, m)
            assert attempts == [], "capture retried after the first attempt"

    def test_falls_back_correctly_when_capture_is_forced_to_fail(self):
        """With capture sabotaged, the candidate must still be correct -- just slower."""
        ref, cfg, base, cand = self._build()
        cand._try_capture = lambda *a, **k: False        # simulate a hostile environment
        x, m = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 7, 0.0, 1.0)
        with torch.inference_mode():
            r = ref.compare_outputs(base(x, m), cand(x, m), rtol=0.02, atol=0.002)
        assert cand._graph is None, "no graph should exist after a forced failure"
        assert cand.graph_verified is False
        assert r.failed_elements == 0, "fallback path must still be correct"
