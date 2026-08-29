"""v17: recombination of the g16 kernel into the g13 frontier, gated on amortization."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels.ffn_fused import amortizes, fits
from bench.matrix import MATRIX


def _ref():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v17", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v17"] = m
    spec.loader.exec_module(m)
    return m


def test_predicate_matches_where_g16_actually_won():
    """Pins the predicate against MEASURED per-config outcomes (finding 25), so a future
    edit that widens it has to argue with the data rather than with an intuition."""
    won = {6, 7, 13}          # v16 vs v13: -7.6%, -5.7%, -2.5%
    chosen = {c.id for c in MATRIX if c.id != 14
              and fits(c.d_model, c.ffn_dim, 2, 64, 99 * 1024)
              and amortizes(c.tokens, c.d_model, c.ffn_dim, 2)}
    assert chosen == won, f"predicate selects {chosen}, g16 won on {won}"


def test_predicate_is_monotone_in_tokens():
    """More tokens can only make the hoist MORE worth paying for. A predicate that is
    not monotone here is fitted to configs, not derived from the mechanism."""
    prev = False
    for tokens in (64, 128, 512, 2048, 8192, 65536, 1280000):
        now = amortizes(tokens, 128, 128, 2)
        assert not (prev and not now), "amortization must not switch back off"
        prev = now


def test_predicate_contains_no_benchmark_knowledge():
    """Rule 2. A shape nobody here has seen must get a sensible answer."""
    assert amortizes(10 ** 7, 256, 256, 2), "a huge unseen shape should fuse"
    assert not amortizes(16, 256, 256, 2), "a tiny unseen shape should not"


@pytest.mark.gpu
def test_matches_the_baseline_on_a_shape_that_fuses():
    ref = _ref()
    # Config 7's shape: 8192 tokens at d_model=32 gives 0.5 weight-bytes/token, under
    # the 0.64 crossover for that width. At 4096 tokens it is 1.0 and correctly declines,
    # which is a real edge this test originally landed on by accident.
    cfg = ref.TransformerConfig(batch_size=64, seq_len=128, d_model=32, num_heads=4,
                                ffn_dim=32, num_layers=4, causal=True)
    assert amortizes(64 * 128, 32, 32, 2)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v17_dispatched_megakernel"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    x = torch.randn(64, 128, 32, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.fused_ffn_used is True, cand.fused_ffn_reason
    d = (got.float() - want.float()).abs()
    assert ((d <= 2e-3) | (d <= 2e-2 * want.float().abs())).all(), f"max_abs {d.max():.3e}"


@pytest.mark.gpu
def test_matches_the_baseline_on_a_shape_that_falls_back():
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=1, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v17_dispatched_megakernel"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    x = torch.randn(1, 128, 128, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.fused_ffn_used is False and "crossover" in cand.fused_ffn_reason
    d = (got.float() - want.float()).abs()
    assert ((d <= 2e-3) | (d <= 2e-2 * want.float().abs())).all()


@pytest.mark.gpu
def test_reports_its_decision_truthfully():
    """v14's is_tuned discipline: an untuned fallback is never presented as tuned."""
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=1, seq_len=64, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=1, causal=True)
    cand = REGISTRY["v17_dispatched_megakernel"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    with torch.no_grad():
        cand(torch.randn(1, 64, 128, device="cuda"))
    assert cand.fused_ffn_reason != "undecided"
    assert cand.fused_ffn_used is False
