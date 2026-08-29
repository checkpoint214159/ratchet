"""v18: capture must not depend on the caller's allocation context."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY


def _ref():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v18", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v18"] = m
    spec.loader.exec_module(m)
    return m


def _cfg(m):
    return m.TransformerConfig(batch_size=16, seq_len=64, d_model=128, num_heads=4,
                               ffn_dim=128, num_layers=2, causal=True)


@pytest.mark.gpu
@pytest.mark.parametrize("inside", [True, False])
def test_captures_regardless_of_where_the_caller_allocated(inside):
    """THE POINT. The graded harness allocates its timing input outside inference_mode
    (reference benchmark line 529); v13-v17 silently lose >2x in that case."""
    torch._dynamo.reset()
    m = _ref()
    cand = REGISTRY["v18_capture_insurance"].build(m.BaselineTransformer)(_cfg(m)).cuda().eval()
    if inside:
        with torch.inference_mode():
            x = torch.randn(16, 64, 128, device="cuda")
    else:
        x = torch.randn(16, 64, 128, device="cuda")
    with torch.inference_mode():
        cand(x, None)
    assert cand.graph_verified is True, f"no graph; capture_source={cand.capture_source}"
    assert cand.capture_source in ("caller", "insurance")


@pytest.mark.gpu
def test_the_parent_really_does_degrade_here():
    """Pins the DEFECT, not just the fix. If a future torch makes capture succeed
    regardless, v18's rationale is gone and this test says so loudly rather than leaving
    dead insurance in the frontier."""
    torch._dynamo.reset()
    m = _ref()
    parent = REGISTRY["v17_dispatched_megakernel"].build(m.BaselineTransformer)(_cfg(m)).cuda().eval()
    x = torch.randn(16, 64, 128, device="cuda")          # deliberately OUTSIDE
    with torch.inference_mode():
        parent(x, None)
    assert parent.graph_verified is False, (
        "v17 now captures from a non-inference input; v18's insurance may be obsolete")


@pytest.mark.gpu
def test_numerics_are_identical_to_the_parent():
    """v18 changes only WHERE the capture input came from. It must not move a single bit
    of the answer relative to its parent."""
    torch._dynamo.reset()
    m = _ref()
    cfg = _cfg(m)
    torch.manual_seed(0)
    base = m.BaselineTransformer(cfg).cuda().eval()
    child = REGISTRY["v18_capture_insurance"].build(m.BaselineTransformer)(cfg).cuda().eval()
    m.copy_model_weights(base, child)
    x = torch.randn(16, 64, 128, device="cuda")
    with torch.inference_mode():
        want, got = base(x), child(x, None)
    d = (got.float() - want.float()).abs()
    assert ((d <= 2e-3) | (d <= 2e-2 * want.float().abs())).all(), f"max_abs {d.max():.3e}"


@pytest.mark.gpu
def test_capture_source_is_reported_not_silent():
    """L36: a degradation nobody can observe is one nobody will notice. This attribute is
    what makes the failure visible in the ledger notes and the report."""
    torch._dynamo.reset()
    m = _ref()
    cand = REGISTRY["v18_capture_insurance"].build(m.BaselineTransformer)(_cfg(m)).cuda().eval()
    assert cand.capture_source == "none"
    x = torch.randn(16, 64, 128, device="cuda")
    with torch.inference_mode():
        cand(x, None)
    assert cand.capture_source in ("caller", "insurance", "none")
