"""v23: hand-written single-tile attention. Correctness, exactness, and the predicate.

The tolerance here is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR, judged
by failed elements and never by max_abs alone (L4). It is not widened anywhere in this
file. Margin is reported alongside pass/fail because a candidate at 90% of budget and one
at 40% are not equally correct (L26).
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels.attn_single_tile import (applies, autotune_tile, choose_tile, fits,
                                            next_pow2, padded_head_dim, pays,
                                            register_bytes, resident_blocks,
                                            single_tile_attention)

ATOL, RTOL = 2e-3, 2e-2


def _ref_module():
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v23", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v23"] = m
    spec.loader.exec_module(m)
    return m


def _fp32_attention(qkv, heads, head_dim):
    """The reference's own formulation: fp32 scores, fp32 softmax, fp32 PV, then the
    head-major repack. This is what the oracle compares against -- NOT SDPA, which
    carries its own error."""
    b, s, _ = qkv.shape
    dm = heads * head_dim
    q, k, v = qkv.float().split(dm, dim=-1)
    q = q.view(b, s, heads, head_dim).transpose(1, 2)
    k = k.view(b, s, heads, head_dim).transpose(1, 2)
    v = v.view(b, s, heads, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    causal = torch.ones(s, s, device=qkv.device, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).transpose(1, 2).reshape(b, s, dm)


def _within(got, want):
    d = (got.float() - want.float()).abs()
    ok = (d <= ATOL) | (d <= RTOL * want.float().abs())
    return ok, d.max().item()


def _tile(s, hd, heads, b):
    props = torch.cuda.get_device_properties("cuda")
    return choose_tile(s, hd, props.regs_per_multiprocessor,
                       props.max_threads_per_multi_processor, props.warp_size)


# ------------------------------------------------------------------- the kernel

@pytest.mark.parametrize("b,heads,s,hd", [
    (8, 4, 128, 32),      # the mainstream shape
    (4, 4, 128, 8),       # head_dim below the MMA width, padded inside the kernel
    (4, 16, 128, 8),
    (4, 4, 32, 32),       # short sequence
    (2, 2, 128, 64),
    (3, 4, 96, 32),       # NOT a power of two: BN rounds up, the mask must handle it
    (2, 4, 16, 16),       # the smallest legal tile
])
def test_kernel_matches_the_fp32_reference_at_the_locked_tolerance(b, heads, s, hd):
    torch.manual_seed(0)
    qkv = torch.randn(b, s, 3 * heads * hd, device="cuda", dtype=torch.float16)
    bm, w, st = _tile(s, hd, heads, b)
    got = single_tile_attention(qkv, heads, hd, hd ** -0.5, bm, w, st)
    ok, max_abs = _within(got, _fp32_attention(qkv, heads, hd))
    assert ok.all(), (f"{(~ok).sum().item()} elements outside tolerance, "
                      f"max_abs={max_abs:.3e}")
    assert max_abs < ATOL, f"margin: {max_abs:.3e} of {ATOL:.0e} ({100*max_abs/ATOL:.0f}%)"


def test_kernel_produces_the_same_layout_as_transpose_reshape():
    """The kernel writes head-major [B, S, d_model] directly. If that layout is wrong the
    out_proj silently mixes heads, which no tolerance check on attention alone would
    catch -- the values are all still plausible."""
    torch.manual_seed(1)
    b, heads, s, hd = 4, 4, 128, 32
    qkv = torch.randn(b, s, 3 * heads * hd, device="cuda", dtype=torch.float16)
    dm = heads * hd
    q, k, v = qkv.split(dm, dim=-1)
    q = q.view(b, s, heads, hd).transpose(1, 2)
    k = k.view(b, s, heads, hd).transpose(1, 2)
    v = v.view(b, s, heads, hd).transpose(1, 2)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    sdpa = sdpa.transpose(1, 2).reshape(b, s, dm)

    bm, w, st = _tile(s, hd, heads, b)
    got = single_tile_attention(qkv, heads, hd, hd ** -0.5, bm, w, st)
    ok, max_abs = _within(got, sdpa)
    assert ok.all(), f"layout mismatch against SDPA's repack, max_abs={max_abs:.3e}"


def test_in_kernel_head_dim_padding_is_exact():
    """Padding head_dim 8 -> 16 with zeros must contribute EXACTLY zero: a zero row of K
    times anything is zero, and the padded output lanes are never stored. Compared
    against the same kernel run on a head_dim it does not have to pad."""
    torch.manual_seed(2)
    b, heads, s = 4, 4, 64
    qkv8 = torch.randn(b, s, 3 * heads * 8, device="cuda", dtype=torch.float16)
    bm, w, st = _tile(s, 8, heads, b)
    got = single_tile_attention(qkv8, heads, 8, 8 ** -0.5, bm, w, st)
    ok, max_abs = _within(got, _fp32_attention(qkv8, heads, 8))
    assert ok.all()
    assert padded_head_dim(8) == 16 and padded_head_dim(32) == 32


def test_causal_mask_is_exact_not_approximate():
    """Row 0 must attend to key 0 only. If the triangle leaked, row 0 would be an average
    over the whole sequence and would differ by far more than a tolerance."""
    torch.manual_seed(3)
    b, heads, s, hd = 1, 1, 64, 32
    qkv = torch.randn(b, s, 3 * hd, device="cuda", dtype=torch.float16)
    bm, w, st = _tile(s, hd, heads, b)
    got = single_tile_attention(qkv, heads, hd, hd ** -0.5, bm, w, st)
    v0 = qkv[0, 0, 2 * hd:3 * hd].float()
    assert torch.allclose(got[0, 0].float(), v0, atol=1e-3), \
        "query 0 must be exactly value 0 under a causal mask"


# --------------------------------------------------------------- the predicate

def test_predicate_declines_where_the_kernel_was_measured_to_lose():
    """head_dim 128 measured 0.94x and head_dim 256 measured 0.84x at the op. Both are
    legal and correct; declining them is the whole point of having a predicate."""
    props = torch.cuda.get_device_properties("cuda")
    assert applies(128, 32, props)[0]
    assert applies(128, 8, props)[0]
    assert applies(32, 32, props)[0]
    assert not applies(128, 128, props)[0]
    assert not applies(128, 256, props)[0]


def test_predicate_declines_sequences_whose_score_tile_cannot_be_held():
    """S=1024 and S=100000 are what FlashAttention is FOR. A 128x1024 fp32 score tile is
    512 KB against a 256 KB register file."""
    props = torch.cuda.get_device_properties("cuda")
    assert not applies(1024, 32, props)[0]
    assert not applies(100000, 64, props)[0]


def test_predicate_is_a_function_of_measured_device_properties():
    """CLAUDE.md rule 2. Shrink the register file and a shape that was accepted must be
    refused -- that is the 'another GPU could evaluate it' test, and a hardcoded table
    wearing a costume would not respond."""
    big, small = 65536, 16384          # 32-bit registers per SM
    assert choose_tile(128, 32, big, 1536) is not None
    assert choose_tile(128, 32, small, 1536) is None
    # ... and the tile chosen on the small card, where one exists, is narrower.
    assert choose_tile(128, 8, big, 1536)[0] >= choose_tile(128, 8, 32768, 1536)[0]


def test_predicate_source_contains_no_config_ids_or_announced_shapes():
    src = (Path(__file__).resolve().parents[2]
           / "bench/kernels/attn_single_tile.py").read_text()
    body = src.split("# ----------------------------------------------------------- does it PAY?")[1]
    body = body.split("# --------------------------------------------------------------- offline sweep")[0]
    code = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    for forbidden in ("config_id", "cfg", "== 128", "== 10000", "seq_len == "):
        assert forbidden not in code, f"predicate mentions {forbidden!r}"


def test_resident_block_count_is_the_thing_that_flips():
    """The measured crossover: >= 4 resident blocks won, <= 2.3 lost. Pin the ordering so
    a future edit to register_bytes cannot silently invert the predicate."""
    props = torch.cuda.get_device_properties("cuda")
    r = props.regs_per_multiprocessor
    t = props.max_threads_per_multi_processor
    assert resident_blocks(128, 8, 64, 4, r, t) >= 4
    assert resident_blocks(128, 32, 64, 4, r, t) >= 4
    assert resident_blocks(128, 128, 64, 4, r, t) < 4
    assert resident_blocks(128, 256, 64, 8, r, t) < 4
    assert register_bytes(1024, 32, 64) > register_bytes(128, 32, 64)


def test_autotuner_returns_a_tile_that_fits_and_pays():
    props = torch.cuda.get_device_properties("cuda")
    tile, why = autotune_tile(128, 32, 4, 64)
    bm, w, st = tile
    assert fits(128, 32, bm, w, props.regs_per_multiprocessor, props.warp_size)
    assert pays(128, 32, bm, w, props.regs_per_multiprocessor,
                props.max_threads_per_multi_processor, props.warp_size)
    assert "autotuned" in why or "derived" in why


# ------------------------------------------------------------------- end to end

def _pair(cfg, seed=0):
    ref = _ref_module()
    torch.manual_seed(seed)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v23_single_tile_attn"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    return base, cand


@pytest.mark.gpu
def test_candidate_matches_the_baseline_within_the_locked_tolerance():
    torch._dynamo.reset()          # L36: a shared Dynamo cache can silently un-compile
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=8, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(7)
    x = torch.randn(8, 128, 128, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.attn_used, cand.attn_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_candidate_is_correct_at_head_dim_8_where_the_kernel_pads():
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=8, seq_len=128, d_model=32, num_heads=4,
                                ffn_dim=32, num_layers=4, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(8)
    x = torch.randn(8, 128, 32, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.attn_used, cand.attn_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_the_declined_path_reports_honestly_and_is_still_correct():
    """head_dim 256. The kernel must not run, the candidate must SAY it did not run, and
    v18's SDPA path must still produce the right answer. An untuned fallback presented as
    a tuned path is the failure v14 was built to prevent."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=2, seq_len=128, d_model=1024, num_heads=4,
                                ffn_dim=1024, num_layers=1, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(9)
    x = torch.randn(2, 128, 1024, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.attn_used is False
    assert "declined" in cand.attn_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"fallback path wrong: max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_the_declined_path_also_covers_a_long_sequence():
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=2, seq_len=512, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=1, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(10)
    x = torch.randn(2, 512, 128, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.attn_used is False and "declined" in cand.attn_reason
    ok, _ = _within(got, want)
    assert ok.all()


@pytest.mark.gpu
def test_padded_input_still_takes_the_kernel_and_is_correct():
    """v8's proof (a right-padded causal key mask is redundant) is what lets the kernel
    skip the mask entirely. The invalid rows are zeroed afterwards, as the reference
    does. Every measurement before finding 11 was blind to this path (L5)."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=4, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(11)
    x = torch.randn(4, 128, 128, device="cuda")
    lengths = torch.tensor([128, 96, 64, 1], device="cuda")
    mask = torch.arange(128, device="cuda")[None, :] < lengths[:, None]
    with torch.no_grad():
        want, got = base(x, mask), cand(x, mask)
    assert cand.attn_used, cand.attn_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_output_depends_on_the_input():
    """L23/L25: a stateful candidate can return a stale static buffer that is the right
    shape, the right dtype and silently wrong. No tolerances needed -- two different
    inputs must not give identical output."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=4, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(12)
    a = torch.randn(4, 128, 128, device="cuda")
    b = torch.randn(4, 128, 128, device="cuda")
    with torch.no_grad():
        ya, yb = cand(a).clone(), cand(b).clone()
    assert not torch.equal(ya, yb)


@pytest.mark.gpu
def test_the_returned_tensor_survives_the_next_call():
    """L25: reduce-overhead style static buffers are overwritten by the following call.
    Correctness that depends on the caller comparing before calling again is not
    correctness (L24)."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=4, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(13)
    a = torch.randn(4, 128, 128, device="cuda")
    b = torch.randn(4, 128, 128, device="cuda")
    with torch.no_grad():
        ya = cand(a)
        snapshot = ya.clone()
        cand(b)
    assert torch.equal(ya, snapshot), "the previous return value was mutated in place"


@pytest.mark.gpu
def test_the_kernel_actually_runs_inside_the_compiled_captured_path():
    """L36: a test can pass because its subject was never built. Everything above would
    still be green if Dynamo had fallen back to eager and the candidate had quietly used
    SDPA. Observe the mechanism directly: our kernel must appear in the CUDA profile and
    FlashAttention must NOT."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=8, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(14)
    x = torch.randn(8, 128, 128, device="cuda")
    with torch.inference_mode():
        for _ in range(3):
            cand(x)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                cand(x)

    names = {e.key for e in prof.key_averages()}
    assert any("attn_single_tile" in n for n in names), \
        f"the hand-written kernel never launched; kernels seen: {sorted(names)[:20]}"
    assert not any("flash" in n.lower() for n in names), \
        "FlashAttention is still running -- the kernel did not replace it"
    assert cand.graph_verified, "graph capture degraded; this is not the measured path"
