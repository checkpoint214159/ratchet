"""v20: the QKV GEMM writes head-major so flash reads contiguously."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels.qkv_headmajor import qkv_headmajor, worth_it


def _ref():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v20", p)
    m = importlib.util.module_from_spec(spec); sys.modules["ref_v20"] = m
    spec.loader.exec_module(m)
    return m


def test_kernel_matches_the_split_and_transpose_it_replaces():
    """Exact same values, different layout. This is a LAYOUT change, not a math change."""
    torch.manual_seed(0)
    B, S, D, H = 8, 64, 128, 4
    hd, M = D // H, B * S
    xn = torch.randn(M, D, device="cuda", dtype=torch.float16) * 0.5
    w = torch.randn(3 * D, D, device="cuda", dtype=torch.float16) * 0.05
    bi = torch.randn(3 * D, device="cuda", dtype=torch.float16) * 0.05

    qkv = F.linear(xn, w, bi).view(B, S, 3 * D)
    qr, kr, vr = (t.view(B, S, H, hd).transpose(1, 2) for t in qkv.split(D, dim=-1))
    q, k, v = qkv_headmajor(xn, w.t().contiguous(), bi, B, S, H)
    for got, want, nm in ((q, qr, "q"), (k, kr, "k"), (v, vr, "v")):
        assert torch.equal(got, want.contiguous()), f"{nm} differs"


def test_the_outputs_are_actually_contiguous():
    """The entire point. If they are not contiguous, flash pays the same tax and the
    kernel is pure cost."""
    q, k, v = qkv_headmajor(
        torch.randn(512, 128, device="cuda", dtype=torch.float16),
        torch.randn(128, 384, device="cuda", dtype=torch.float16),
        torch.randn(384, device="cuda", dtype=torch.float16), 4, 128, 4)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()


def test_predicate_declines_below_the_measured_crossover():
    """Measured: 1.78x tax at 1.28M tokens, 1.07x at 65k, none at 8k. Below the
    crossover the hand-written GEMM loses to cuBLAS and must not be used."""
    assert worth_it(10000, 128, 128, 4)          # config 6, 1.28M tokens
    assert not worth_it(64, 1024, 128, 4)        # config 13, 65k
    assert not worth_it(64, 128, 128, 4)         # config 1, 8k


def test_predicate_rejects_head_dim_below_the_mma_width():
    assert not worth_it(10000, 128, 128, 16)     # head_dim 8 < 16


@pytest.mark.gpu
@pytest.mark.parametrize("batch,expect", [(4096, True), (64, False)])
def test_end_to_end_correct_on_both_paths(batch, expect):
    m = _ref()
    cfg = m.TransformerConfig(batch_size=batch, seq_len=128, d_model=128, num_heads=4,
                              ffn_dim=128, num_layers=2, causal=True)
    torch.manual_seed(0)
    base = m.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v20_headmajor_qkv"].build(m.BaselineTransformer)(cfg).cuda().eval()
    m.copy_model_weights(base, cand)
    with torch.inference_mode():
        x = torch.randn(batch, 128, 128, device="cuda")
        want, got = base(x), cand(x)
    assert cand.headmajor_used is expect, cand.headmajor_reason
    d = (got.float() - want.float()).abs()
    assert ((d <= 2e-3) | (d <= 2e-2 * want.float().abs())).all(), f"max_abs {d.max():.3e}"


def test_v19_and_v20_are_siblings_not_a_chain():
    """Finding 28's discipline, made concrete: both fork from v18 and neither descends
    from the other. This is the first real fork in the tree."""
    from bench.candidates import REGISTRY as R
    assert R["v19_norm_fused"].parent == "v18_capture_insurance"
    assert R["v20_headmajor_qkv"].parent == "v18_capture_insurance"
