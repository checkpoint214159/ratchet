"""config.causal must be honoured, not assumed.

The reference benchmark defaults to `causal: bool = False` (line 89) with `--causal` as an
opt-in flag. Every candidate from v5 to v23 hardcoded `is_causal=True` and returned
three-quarters of its output wrong on that default, with all 177 tests green -- because
every announced config is causal and nothing ever exercised the other branch.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY


def _ref():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_causal", p)
    m = importlib.util.module_from_spec(spec); sys.modules["ref_causal"] = m
    spec.loader.exec_module(m)
    return m


def test_the_reference_really_does_default_to_non_causal():
    """Pins the premise. If a future reference flips this default, v26's delegation is
    dead weight and this test says so rather than leaving it unexamined."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "benchmarks/reference/torch_transformer_benchmark.py").read_text()
    assert "causal: bool = False" in src


@pytest.mark.gpu
@pytest.mark.parametrize("causal", [True, False])
def test_v26_is_correct_on_both_settings(causal):
    m = _ref()
    cfg = m.TransformerConfig(batch_size=4, seq_len=64, d_model=128, num_heads=4,
                              ffn_dim=128, num_layers=2, causal=causal)
    torch.manual_seed(0)
    base = m.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v26_causal_correct"].build(m.BaselineTransformer)(cfg).cuda().eval()
    m.copy_model_weights(base, cand)
    with torch.inference_mode():
        x = torch.randn(4, 64, 128, device="cuda")
        want, got = base(x), cand(x)
    d = (got.float() - want.float()).abs()
    ok = (d <= 2e-3) | (d <= 2e-2 * want.float().abs())
    assert ok.all(), f"causal={causal}: {(~ok).sum().item()} failed, max_abs {d.max():.3e}"


@pytest.mark.gpu
def test_v26_reports_which_path_it_took():
    m = _ref()
    for causal, expect in ((True, "optimized"), (False, "baseline")):
        cfg = m.TransformerConfig(batch_size=2, seq_len=32, d_model=128, num_heads=4,
                                  ffn_dim=128, num_layers=1, causal=causal)
        cand = REGISTRY["v26_causal_correct"].build(m.BaselineTransformer)(cfg).cuda().eval()
        with torch.inference_mode():
            cand(torch.randn(2, 32, 128, device="cuda"))
        assert cand.causal_path.startswith(expect), cand.causal_path


@pytest.mark.gpu
def test_the_defect_is_still_present_in_the_parent():
    """Pins the BUG, so v26's delegation cannot rot into dead code unnoticed. If a future
    parent starts honouring config.causal, this fails and v26 can be simplified."""
    m = _ref()
    cfg = m.TransformerConfig(batch_size=4, seq_len=64, d_model=128, num_heads=4,
                              ffn_dim=128, num_layers=2, causal=False)
    torch.manual_seed(0)
    base = m.BaselineTransformer(cfg).cuda().eval()
    parent = REGISTRY["v23_single_tile_attn"].build(m.BaselineTransformer)(cfg).cuda().eval()
    m.copy_model_weights(base, parent)
    with torch.inference_mode():
        x = torch.randn(4, 64, 128, device="cuda")
        want, got = base(x), parent(x)
    d = (got.float() - want.float()).abs()
    ok = (d <= 2e-3) | (d <= 2e-2 * want.float().abs())
    assert not ok.all(), "v23 now honours config.causal; v26's delegation may be obsolete"
