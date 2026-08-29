"""v15: lifting Inductor's SM veto. Correctness and honesty of the lift itself."""
import pytest
import torch

from bench.candidates import REGISTRY
from bench.candidates.v15_lifted_veto import lift_sm_veto

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_registered_with_the_sampled_parent():
    spec = REGISTRY["v15_lifted_veto"]
    assert spec.parent == "v9b_reduce_overhead" and spec.generation == 15


def test_lift_is_idempotent_and_reports_truthfully():
    """Two calls must not double-patch, and the return value must say what happened."""
    import torch._inductor.utils as iu
    first = lift_sm_veto()
    second = lift_sm_veto()
    assert first == second
    if first:
        assert iu.is_big_gpu(0) is True
        assert getattr(iu, "_ratchet_veto_lifted", False) is True


def test_lift_only_fires_on_a_card_that_was_actually_vetoed():
    """On >= 68 SMs there is nothing to lift and torch must be left alone -- otherwise
    this is a blanket monkeypatch rather than a device-conditioned one."""
    prop = torch.cuda.get_device_properties(0)
    lifted = lift_sm_veto()
    if prop.multi_processor_count >= 68:
        assert lifted is False, "must not patch a card torch never vetoed"
    else:
        assert lifted is True


def test_the_veto_was_really_firing_on_this_card():
    """Pins the premise. If a torch upgrade lowers min_sms, this candidate's entire
    rationale evaporates and we want a red test, not a silent no-op."""
    import importlib, torch._inductor.utils as iu
    src = importlib.import_module("torch._inductor.utils").__file__
    text = open(src).read()
    assert "min_sms" in text, "torch no longer has an SM threshold; re-derive v15"
    prop = torch.cuda.get_device_properties(0)
    assert prop.multi_processor_count == 66, "device changed; re-measure before trusting v15"


def _ref():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v15", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v15"] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.gpu
def test_output_matches_the_reference_within_the_locked_tolerance():
    """The whole point: fusing an epilogue must not change the answer.

    Tolerances are the LOCKED ones (atol 2e-3 / rtol 2e-2) and the criterion is OR, as
    the oracle applies it -- an element passes on either bound.
    """
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=4, seq_len=64, d_model=64, num_heads=4,
                                ffn_dim=64, num_layers=2, causal=True)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v15_lifted_veto"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)

    torch.manual_seed(1)
    x = torch.randn(4, 64, 64, device="cuda")
    with torch.no_grad():
        want = base(x)
        got = cand(x)
    assert got.shape == want.shape and got.dtype == want.dtype

    d = (got.float() - want.float()).abs()
    ok = (d <= 2e-3) | (d <= 2e-2 * want.float().abs())
    assert ok.all(), f"{(~ok).sum().item()} elements outside tolerance, max_abs={d.max():.3e}"


@pytest.mark.gpu
def test_v15_actually_reaches_a_triton_gemm_template():
    """Pins the MECHANISM, not just the outcome.

    If a future torch makes templates unreachable for another reason, v15 would quietly
    degrade to 'v9b with a slower compile mode' and still pass every other test here.
    """
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    torch.manual_seed(0)
    cand = REGISTRY["v15_lifted_veto"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    x = torch.randn(64, 128, 128, device="cuda")
    with torch.no_grad():
        cand(x)
    torch.cuda.synchronize()
    assert cand.veto_lifted is True, "v15 must report that it lifted the veto"

    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        cand(x)
        torch.cuda.synchronize()
    names = " ".join(e.key for e in prof.key_averages())
    assert "triton_tem" in names, f"no Triton GEMM template was generated; got: {names[:400]}"
