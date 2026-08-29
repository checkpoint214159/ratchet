"""v24: the out-projection / fp32-widen / residual-add fusion.

Correctness at the LOCKED tolerance, the layout claim the whole proposal turned on, the
dispatch predicate's honesty, and the fallback path.
"""
import inspect

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels import outproj_resid as K

ATOL, RTOL = 2e-3, 2e-2          # LOCKED. CLAUDE.md rule 1. Never widened here.


def _ref_module():
    import importlib.util
    import sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v24", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v24"] = m
    spec.loader.exec_module(m)
    return m


def _within_locked_tolerance(got, ref):
    d = (got - ref).abs()
    return ((d <= ATOL) | (d <= RTOL * ref.abs())).all(), float(d.max())


def _ctx(bsz, seq, heads, hd, seed=0, scale=0.2):
    torch.manual_seed(seed)
    d = heads * hd
    qkv = torch.randn(bsz, seq, 3 * d, device="cuda", dtype=torch.float16) * scale
    q, k, v = qkv.split(d, dim=-1)
    q = q.view(bsz, seq, heads, hd).transpose(1, 2)
    k = k.view(bsz, seq, heads, hd).transpose(1, 2)
    v = v.view(bsz, seq, heads, hd).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _weights(d, seed=1):
    torch.manual_seed(seed)
    w = torch.randn(d, d, device="cuda", dtype=torch.float16) * 0.05
    b = torch.randn(d, device="cuda", dtype=torch.float16) * 0.05
    return w, b


# ------------------------------------------------------------------ the kernel itself

@pytest.mark.parametrize("bsz,seq,heads,hd", [
    (1, 128, 4, 32),        # one token block; the small tile
    (64, 32, 4, 32),        # below SM saturation
    (64, 128, 4, 32),       # mainstream
    (64, 128, 16, 8),       # head_dim 8
    (64, 128, 4, 8),        # d_model 32
    (64, 128, 1, 128),      # one head
    (16, 128, 4, 256),      # d_model 1024, head_dim 256
])
def test_kernel_matches_the_fp32_reference_at_the_locked_tolerance(bsz, seq, heads, hd):
    """Against the fp32 formulation the oracle compares to -- NOT against the fp16
    candidate path, which carries its own error."""
    d = heads * hd
    ctx = _ctx(bsz, seq, heads, hd)
    w, b = _weights(d)
    res = torch.randn(bsz * seq, d, device="cuda", dtype=torch.float32)
    props = torch.cuda.get_device_properties("cuda")
    tile = K.tiling_for(bsz * seq, d, props.multi_processor_count)
    assert K.fits(d, heads, 2, tile[0], tile[1], tile[2],
                  props.shared_memory_per_block_optin)

    ref = res + (ctx.transpose(1, 2).reshape(bsz * seq, d).float() @ w.t().float()
                 + b.float())
    got = K.outproj_resid(ctx, res, w.t().contiguous(), b, *tile)
    ok, mx = _within_locked_tolerance(got, ref)
    assert ok, f"max_abs {mx:.3e} at B{bsz} S{seq} H{heads} hd{hd}"


def test_kernel_agrees_with_the_two_kernel_path_it_replaces():
    """Equivalence against the exact expression in v18's `_core`."""
    bsz, seq, heads, hd = 64, 128, 4, 32
    d = heads * hd
    ctx = _ctx(bsz, seq, heads, hd, seed=3)
    w, b = _weights(d, seed=4)
    res = torch.randn(bsz * seq, d, device="cuda", dtype=torch.float32)
    props = torch.cuda.get_device_properties("cuda")
    tile = K.tiling_for(bsz * seq, d, props.multi_processor_count)

    two = res + F.linear(ctx.transpose(1, 2).reshape(bsz * seq, d), w, b).float()
    got = K.outproj_resid(ctx, res, w.t().contiguous(), b, *tile)
    ok, mx = _within_locked_tolerance(got, two)
    assert ok, f"disagrees with the path it replaces by {mx:.3e}"


def test_kernel_is_more_accurate_than_the_path_it_replaces():
    """The fusion DELETES an fp16 rounding step rather than adding one, so it returns
    tolerance margin. L26: margin is a first-class metric, not just pass/fail."""
    bsz, seq, heads, hd = 64, 128, 4, 32
    d = heads * hd
    ctx = _ctx(bsz, seq, heads, hd, seed=5)
    w, b = _weights(d, seed=6)
    res = torch.randn(bsz * seq, d, device="cuda", dtype=torch.float32)
    props = torch.cuda.get_device_properties("cuda")
    tile = K.tiling_for(bsz * seq, d, props.multi_processor_count)

    ref64 = res.double() + (ctx.transpose(1, 2).reshape(bsz * seq, d).double()
                            @ w.t().double() + b.double())
    two = res + F.linear(ctx.transpose(1, 2).reshape(bsz * seq, d), w, b).float()
    fused = K.outproj_resid(ctx, res, w.t().contiguous(), b, *tile)
    e_two = float((two - ref64).abs().max())
    e_fused = float((fused - ref64).abs().max())
    assert e_fused < e_two, f"fused {e_fused:.2e} is not tighter than two-kernel {e_two:.2e}"


def test_the_residual_add_is_fp32():
    """Finding 08: an fp16 residual fails 12 of 14 configs. The kernel must not round the
    residual. Detected by feeding a residual whose value is unrepresentable in fp16
    without loss and a zero projection -- the output must carry it back exactly."""
    bsz, seq, heads, hd = 4, 128, 4, 32
    d = heads * hd
    ctx = _ctx(bsz, seq, heads, hd, seed=7)
    w = torch.zeros(d, d, device="cuda", dtype=torch.float16)
    b = torch.zeros(d, device="cuda", dtype=torch.float16)
    res = torch.full((bsz * seq, d), 1.0 + 2.0 ** -13, device="cuda", dtype=torch.float32)
    props = torch.cuda.get_device_properties("cuda")
    tile = K.tiling_for(bsz * seq, d, props.multi_processor_count)

    got = K.outproj_resid(ctx, res, w.t().contiguous(), b, *tile)
    assert torch.equal(got, res), (
        "the residual was rounded; max deviation "
        f"{float((got - res).abs().max()):.3e} (fp16 ulp at 1.0 is 2**-11)")


def test_the_contiguous_fast_path_and_the_general_gather_agree():
    """The vectorized form is taken only when the token-major view is contiguous. Both
    address forms must produce the same answer, or the fast path is a silent bug on any
    backend that returns a genuinely head-major ctx."""
    bsz, seq, heads, hd = 8, 128, 4, 32
    d = heads * hd
    ctx = _ctx(bsz, seq, heads, hd, seed=8)
    w, b = _weights(d, seed=9)
    res = torch.randn(bsz * seq, d, device="cuda", dtype=torch.float32)
    props = torch.cuda.get_device_properties("cuda")
    tile = K.tiling_for(bsz * seq, d, props.multi_processor_count)

    assert ctx.transpose(1, 2).is_contiguous(), "fixture no longer exercises the fast path"
    fast = K.outproj_resid(ctx, res, w.t().contiguous(), b, *tile)

    # A genuinely head-major ctx: same values, [B, H, S, hd]-contiguous strides.
    head_major = ctx.contiguous()
    assert not head_major.transpose(1, 2).is_contiguous()
    slow = K.outproj_resid(head_major, res, w.t().contiguous(), b, *tile)
    assert torch.equal(fast, slow), f"paths differ by {float((fast-slow).abs().max()):.3e}"


# ------------------------------------------------------------------ dispatch honesty

def _executable_body(fn) -> str:
    """Source with the docstring removed. Prose is allowed to name a config; the code
    that decides is not."""
    src = inspect.getsource(fn)
    doc = inspect.getdoc(fn)
    if doc:
        for line in doc.splitlines():
            src = src.replace(line, "")
    return src


def test_predicates_name_no_config_ids_and_no_announced_constants():
    """CLAUDE.md rule 2: a dispatch predicate is shapes and measured device properties.

    The predicate must respond to the DEVICE and the SHAPE, so nothing in its executable
    body may name a config id or an announced constant from bench/matrix.py."""
    body = "".join(_executable_body(f) for f in (K.fits, K.tiling_for, K.programs))
    banned = ("config", "cfg", "10000", "1280000", "65536", "8192", "matrix")
    for token in banned:
        assert token not in body, f"predicate body mentions {token!r}"

    # And it must actually read a measured device property rather than a literal.
    cand_src = inspect.getsource(
        __import__("bench.candidates.v24_outproj_prologue", fromlist=["build"]))
    assert "get_device_properties" in cand_src
    assert "multi_processor_count" in cand_src
    assert "shared_memory_per_block_optin" in cand_src


def test_tiling_responds_to_the_measured_sm_count():
    """The tile choice must be a function of the DEVICE, not of the shape alone. Pretend
    the card has four times the SMs and the same shape must stop saturating it."""
    d = 128
    tokens = 8192
    assert K.tiling_for(tokens, d, 66) == K._clamp(K.WIDE_TILE, d)
    assert K.tiling_for(tokens, d, 66 * 8) == K._clamp(K.SMALL_TILE, d)
    assert K.tiling_for(512, d, 66) == K._clamp(K.SMALL_TILE, d)


def test_fits_declines_on_a_small_smem_card():
    """The 'another GPU can evaluate it' test v14_dispatch was built to satisfy."""
    assert K.fits(128, 4, 2, 64, 128, 64, 101376)
    assert not K.fits(128, 4, 2, 64, 128, 64, 48 * 1024)


def test_fits_rejects_illegal_dot_shapes():
    assert not K.fits(128, 4, 2, 8, 128, 64, 101376)      # BM < 16
    assert not K.fits(128, 4, 2, 64, 128, 8, 101376)      # BK < 16
    assert not K.fits(96, 4, 2, 64, 64, 64, 101376)       # d_model not a power of two
    assert not K.fits(128, 3, 2, 64, 64, 64, 101376)      # d_model not divisible by heads


# ------------------------------------------------------------------ the candidate

def _pair(ref, bsz, seq, heads, d, layers, seed=11):
    """A reference model and a v24 candidate holding identical weights."""
    cfg = ref.TransformerConfig(batch_size=bsz, seq_len=seq, d_model=d, num_heads=heads,
                                ffn_dim=d, num_layers=layers, causal=True)
    torch.manual_seed(seed)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v24_outproj_prologue"].build(
        ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    return base, cand


def test_registered_with_the_declared_parent():
    spec = REGISTRY["v24_outproj_prologue"]
    assert spec.parent == "v18_capture_insurance"
    assert spec.generation == 24


@pytest.mark.parametrize("bsz,seq,heads,d,layers", [
    (8, 128, 4, 128, 2),
    (4, 128, 16, 128, 2),
    (4, 128, 4, 32, 2),
])
def test_candidate_matches_the_fp32_reference_at_the_locked_tolerance(bsz, seq, heads, d, layers):
    torch._dynamo.reset()
    ref = _ref_module()
    base, cand = _pair(ref, bsz, seq, heads, d, layers)

    torch.manual_seed(21)
    x = torch.randn(bsz, seq, d, device="cuda")
    with torch.inference_mode():
        expected = base(x)
        got = cand(x)
    assert cand.outproj_used, cand.outproj_reason
    ok, mx = _within_locked_tolerance(got, expected)
    assert ok, f"max_abs {mx:.3e}"


def test_candidate_falls_back_and_stays_correct_when_masked():
    """The kernel absorbs the residual add, which is exactly where the padded path applies
    its masked_fill -- so the masked input must go to the parent, and still be right."""
    torch._dynamo.reset()
    ref = _ref_module()
    d, heads, layers, bsz, seq = 128, 4, 2, 4, 128
    base, cand = _pair(ref, bsz, seq, heads, d, layers, seed=12)

    torch.manual_seed(22)
    x = torch.randn(bsz, seq, d, device="cuda")
    lengths = torch.randint(1, seq + 1, (bsz,), device="cuda")
    mask = torch.arange(seq, device="cuda")[None, :] < lengths[:, None]
    with torch.inference_mode():
        expected = base(x, mask)
        got = cand(x, mask)
    assert not cand._nomask
    ok, mx = _within_locked_tolerance(got, expected)
    assert ok, f"masked fallback max_abs {mx:.3e}"


def test_candidate_declines_rather_than_crashes_on_a_shape_the_kernel_refuses():
    """A predicate that cannot decline is a hardcoded table wearing a costume."""
    torch._dynamo.reset()
    ref = _ref_module()
    d, heads, layers = 96, 4, 1        # d_model not a power of two -> fits() says no
    base, cand = _pair(ref, 2, 128, heads, d, layers, seed=13)

    torch.manual_seed(23)
    x = torch.randn(2, 128, d, device="cuda")
    with torch.inference_mode():
        expected = base(x)
        got = cand(x)
    assert not cand.outproj_used
    assert "declined" in cand.outproj_reason
    ok, mx = _within_locked_tolerance(got, expected)
    assert ok, f"declined path max_abs {mx:.3e}"


def test_output_depends_on_the_input():
    """L23/L25: the crudest invariant, and the only one that catches a stale buffer. This
    candidate inherits v13's graph capture, which is where that bug lives."""
    torch._dynamo.reset()
    ref = _ref_module()
    d, heads, layers = 128, 4, 2
    _base, cand = _pair(ref, 4, 128, heads, d, layers, seed=14)

    torch.manual_seed(24)
    a = torch.randn(4, 128, d, device="cuda")
    b = torch.randn(4, 128, d, device="cuda")
    with torch.inference_mode():
        ya = cand(a).clone()
        yb = cand(b).clone()
        ya2 = cand(a).clone()
    assert not torch.equal(ya, yb), "different inputs produced identical output"
    assert torch.allclose(ya, ya2, atol=1e-5), "same input produced different output"
