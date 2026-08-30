"""`CandidateV23._attention` must be exactly the four inline blocks it replaced.

v40 needed one place to override how attention is computed. The `if self.attn_used:
single_tile_attention(...) else: <sdpa + head-major repack>` block existed inline in FOUR
places -- v23's `_core`, v34's `_core`, and both branches of v36's -- so overriding it
meant copying three long `_core` bodies and keeping them in sync forever. That is the
[L14] shape: a mechanical edit across duplicated code that produces something valid and
wrong.

The block was moved into `CandidateV23._attention` unchanged and the four call sites became
`self._attention(qkv, a, b, s)`. **The claim that this is behaviour-preserving is
load-bearing** -- v38 is the shipping candidate and v40's ABBA comparison uses it as the
control arm -- so it is asserted here rather than assumed.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels.attn_single_tile import single_tile_attention

ATOL, RTOL = 2e-3, 2e-2


def _ref_module():
    p = (Path(__file__).resolve().parents[2]
         / "benchmarks/reference/torch_transformer_benchmark.py")
    spec = importlib.util.spec_from_file_location("ref_extpoint", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_extpoint"] = m
    spec.loader.exec_module(m)
    return m


def _inline_single_tile(model, qkv, a):
    """The block v23/v34/v36 used to run when `attn_used` was True, written out."""
    bm, warps, stages = model.attn_tile
    return single_tile_attention(qkv, a.num_heads, a.head_dim, a.scale, bm, warps, stages)


def _inline_sdpa(qkv, a, b, s):
    """The block v23/v34/v36 used to run when `attn_used` was False, written out."""
    q, k, v = qkv.split(a.d_model, dim=-1)
    q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
    k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
    v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
    return F.scaled_dot_product_attention(
        q, k, v, is_causal=True).transpose(1, 2).reshape(b, s, a.d_model)


def _model(name, batch, heads, d_model, seq_len):
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=batch, seq_len=seq_len, d_model=d_model,
                                num_heads=heads, ffn_dim=d_model, num_layers=4,
                                causal=True)
    cfg.validate()
    torch.manual_seed(99)
    base = ref.BaselineTransformer(cfg)
    m = REGISTRY[name].build(ref.BaselineTransformer)(cfg)
    ref.copy_model_weights(base, m)
    return m.to(device="cuda", dtype=torch.float32).eval(), cfg, ref


@pytest.mark.parametrize("name", ["v23_single_tile_attn", "v34_launch_bound",
                                  "v36_gemm_gelu", "v38_stream_fallback"])
def test_attention_method_reproduces_the_single_tile_branch(name):
    m, cfg, ref = _model(name, 8, 2, 128, 128)
    x, mask = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       seed=99, padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        m(x, mask)                                  # settles `_decide_attn`
    if not getattr(m, "attn_used", False):
        pytest.skip(f"{name} declined the single-tile kernel on this shape")

    a = m.layers[0].attention
    b, s = 8, 128
    g = torch.Generator(device="cuda").manual_seed(3)
    qkv = torch.randn(b, s, 3 * a.d_model, device="cuda", dtype=torch.float16,
                      generator=g) * 0.3
    with torch.inference_mode():
        got = m._attention(qkv, a, b, s)
        want = _inline_single_tile(m, qkv, a)
    assert torch.equal(got, want), "the extension point is not the inline block"


@pytest.mark.parametrize("name", ["v34_launch_bound", "v36_gemm_gelu",
                                  "v38_stream_fallback"])
def test_attention_method_reproduces_the_sdpa_branch(name):
    """The `else` branch is reached wherever `attn_single_tile` declines -- head_dim 128
    (config 9) is one such shape. It is the branch that carries the repack, so a
    regression here is a wrong LAYOUT, which the next op consumes silently."""
    m, cfg, ref = _model(name, 8, 1, 128, 128)
    x, mask = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       seed=99, padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        m(x, mask)
    assert not getattr(m, "attn_used", True), (
        "head_dim 128 was expected to decline the single-tile kernel; this test is no "
        "longer exercising the sdpa branch")

    a = m.layers[0].attention
    b, s = 8, 128
    g = torch.Generator(device="cuda").manual_seed(3)
    qkv = torch.randn(b, s, 3 * a.d_model, device="cuda", dtype=torch.float16,
                      generator=g) * 0.3
    with torch.inference_mode():
        got = m._attention(qkv, a, b, s)
        want = _inline_sdpa(qkv, a, b, s)
    assert torch.equal(got, want), "the extension point is not the inline block"


@pytest.mark.parametrize("name", ["v23_single_tile_attn", "v26_causal_correct",
                                  "v34_launch_bound", "v36_gemm_gelu",
                                  "v37_recombined2", "v38_stream_fallback"])
@pytest.mark.parametrize("heads,d_model", [(2, 128), (1, 128), (4, 32)])
def test_every_ancestor_still_matches_the_baseline_end_to_end(name, heads, d_model):
    """The refactor touched three shipped candidates. This is the check that they all
    still produce the reference's answer at the locked tolerance, on a shape that takes
    the single-tile branch (heads=2), one that takes the sdpa branch (heads=1), and one
    with head_dim 32."""
    m, cfg, ref = _model(name, 8, heads, d_model, 128)
    torch.manual_seed(99)
    base = ref.BaselineTransformer(cfg).to(device="cuda", dtype=torch.float32).eval()
    ref.copy_model_weights(base, m)
    x, mask = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       seed=99, padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        want = base(x, mask)
        got = m(x, mask)
    res = ref.compare_outputs(want, got, rtol=RTOL, atol=ATOL)
    assert res.passed, f"{name}: max_abs {res.max_abs_error:.3e}"
