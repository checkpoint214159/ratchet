"""v16: the first hand-written kernel. Correctness, dispatch honesty, and the mechanism."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels.ffn_fused import fits, fused_ffn, smem_bytes


def _ref():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v16", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v16"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- the kernel itself

def test_kernel_matches_the_fp32_reference_formulation():
    """Against the fp32 reference the oracle actually compares to -- NOT against the
    fp16 candidate path, which carries its own error."""
    torch.manual_seed(0)
    M, D = 4096, 128
    xn = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.5
    res = torch.randn(M, D, device="cuda", dtype=torch.float32)
    w1 = torch.randn(D, D, device="cuda", dtype=torch.float16) * 0.05
    b1 = torch.randn(D, device="cuda", dtype=torch.float16) * 0.05
    w2 = torch.randn(D, D, device="cuda", dtype=torch.float16) * 0.05
    b2 = torch.randn(D, device="cuda", dtype=torch.float16) * 0.05

    ref = res + (F.gelu(xn.float() @ w1.float() + b1.float(), approximate="none")
                 @ w2.float() + b2.float())
    got = fused_ffn(xn, res, w1, b1, w2, b2)
    d = (got - ref).abs()
    ok = (d <= 2e-3) | (d <= 2e-2 * ref.abs())
    assert ok.all(), f"max_abs {d.max():.3e}"


def test_kernel_uses_erf_gelu_not_the_tanh_approximation():
    """The reference is approximate='none'. tanh differs by ~1e-3, half our entire
    budget spent on an approximation nobody asked for -- and the proposal's own probe
    kernel used tanh, so this is a real trap and not a hypothetical one."""
    torch.manual_seed(1)
    M, D = 512, 128
    xn = torch.randn(M, D, device="cuda", dtype=torch.float16)
    res = torch.zeros(M, D, device="cuda", dtype=torch.float32)
    w1 = torch.eye(D, device="cuda", dtype=torch.float16)
    b1 = torch.zeros(D, device="cuda", dtype=torch.float16)
    w2 = torch.eye(D, device="cuda", dtype=torch.float16)
    b2 = torch.zeros(D, device="cuda", dtype=torch.float16)
    got = fused_ffn(xn, res, w1, b1, w2, b2)

    exact = F.gelu(xn.float(), approximate="none")
    tanh_ = F.gelu(xn.float(), approximate="tanh")
    d_exact = (got - exact).abs().max().item()
    d_tanh = (got - tanh_).abs().max().item()
    assert d_exact < d_tanh, f"kernel is closer to tanh ({d_tanh:.2e}) than erf ({d_exact:.2e})"


def test_kernel_is_more_accurate_than_the_fp16_path_it_replaces():
    """Keeping h in fp32 registers RETURNS tolerance margin rather than spending it."""
    torch.manual_seed(2)
    M, D = 4096, 128
    xn = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.5
    res = torch.randn(M, D, device="cuda", dtype=torch.float32)
    w1 = torch.randn(D, D, device="cuda", dtype=torch.float16) * 0.05
    b1 = torch.randn(D, device="cuda", dtype=torch.float16) * 0.05
    w2 = torch.randn(D, D, device="cuda", dtype=torch.float16) * 0.05
    b2 = torch.randn(D, device="cuda", dtype=torch.float16) * 0.05

    ref = res + (F.gelu(xn.float() @ w1.float() + b1.float(), approximate="none")
                 @ w2.float() + b2.float())
    fused = fused_ffn(xn, res, w1, b1, w2, b2)
    h = F.linear(xn, w1.t().contiguous(), b1)
    current = res + F.linear(F.gelu(h.float(), approximate="none").to(torch.float16),
                             w2.t().contiguous(), b2).float()
    assert (fused - ref).abs().max() < (current - ref).abs().max()


# ---------------------------------------------------------------- the dispatch predicate

def test_predicate_declines_when_the_weights_cannot_fit():
    """d_model=1024 needs 4.25 MB of weights against 99 KB. Declining is correct; a
    kernel that tried would fail to compile at run time on the graded machine."""
    assert not fits(1024, 1024, 2, 64, 99 * 1024)
    assert fits(128, 128, 2, 64, 99 * 1024)


def test_predicate_uses_no_config_ids_or_literals():
    """CLAUDE.md rule 2. The predicate must be a function of shapes and MEASURED device
    properties, so it generalizes to a card we have never seen."""
    small = 48 * 1024
    assert not fits(128, 128, 2, 64, small), "must decline on a 48 KB-smem card"
    assert fits(128, 128, 2, 64, 99 * 1024), "must accept on this one"


def test_predicate_rejects_dimensions_below_the_mma_width():
    """sm_89's MMA is m16n8k16, so tl.dot needs every dimension >= 16."""
    assert not fits(8, 8, 2, 64, 99 * 1024)


def test_predicate_rejects_non_power_of_two(): 
    assert not fits(96, 96, 2, 64, 99 * 1024)


# ---------------------------------------------------------------- end to end

@pytest.mark.gpu
def test_candidate_matches_the_baseline_within_the_locked_tolerance():
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=8, seq_len=64, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v16_ffn_megakernel"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    torch.manual_seed(3)
    x = torch.randn(8, 64, 128, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    d = (got.float() - want.float()).abs()
    ok = (d <= 2e-3) | (d <= 2e-2 * want.float().abs())
    assert ok.all(), f"{(~ok).sum().item()} elements outside tolerance, max_abs={d.max():.3e}"


@pytest.mark.gpu
def test_candidate_reports_truthfully_whether_it_used_the_kernel():
    """An untuned fallback must never be presented as the tuned path (v14's discipline)."""
    ref = _ref()
    torch.manual_seed(0)
    wide = ref.TransformerConfig(batch_size=2, seq_len=32, d_model=1024, num_heads=4,
                                 ffn_dim=1024, num_layers=1, causal=True)
    cand = REGISTRY["v16_ffn_megakernel"].build(ref.BaselineTransformer)(wide).cuda().eval()
    with torch.no_grad():
        cand(torch.randn(2, 32, 1024, device="cuda"))
    assert cand._ffn_decision.used is False
    assert "declined" in cand._ffn_decision.reason


@pytest.mark.gpu
def test_wide_model_falls_back_and_is_still_correct():
    """The declined path must produce the right answer, not merely decline."""
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=2, seq_len=32, d_model=1024, num_heads=4,
                                ffn_dim=1024, num_layers=1, causal=True)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v16_ffn_megakernel"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    x = torch.randn(2, 32, 1024, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    d = (got.float() - want.float()).abs()
    assert ((d <= 2e-3) | (d <= 2e-2 * want.float().abs())).all()
