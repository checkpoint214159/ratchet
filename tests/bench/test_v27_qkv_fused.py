"""v27: the Q/K/V projection fused into the attention kernel.

The tolerance here is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR, judged
by failed elements and never by max_abs alone (L4). It is not widened anywhere in this
file. Margin is reported alongside pass/fail because a candidate at 90% of budget and one
at 40% are not equally correct (L26).

The fallback path is tested explicitly: this candidate declines five of the fourteen
announced shapes and a decline that silently produces the wrong thing is worse than no
candidate at all.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels.attn_qkv_fused import (applies, autotune_tile, choose_tile,
                                          fits, fused_qkv_attention, fused_read_bytes,
                                          is_pow2, moves_fewer_bytes, next_pow2,
                                          padded_head_dim, pays, projection_redundancy,
                                          register_bytes, ridge_point,
                                          smem_resident_bytes, split_io_bytes,
                                          viable_tiles)
from bench.matrix import BY_ID

ATOL, RTOL = 2e-3, 2e-2


def _ref_module():
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v27", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v27"] = m
    spec.loader.exec_module(m)
    return m


def _fp32_projected_attention(x, w_t, bias, heads, head_dim):
    """The reference's own formulation in fp32: project, split heads, fp32 scores,
    causal mask via triu(1), fp32 softmax, fp32 PV, head-major repack. NOT SDPA, which
    carries its own error."""
    b, s, _ = x.shape
    dm = heads * head_dim
    qkv = (x.float() @ w_t.float()) + bias.float()
    q, k, v = qkv.split(dm, dim=-1)
    q = q.view(b, s, heads, head_dim).transpose(1, 2)
    k = k.view(b, s, heads, head_dim).transpose(1, 2)
    v = v.view(b, s, heads, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    causal = torch.ones(s, s, device=x.device, dtype=torch.bool).triu(diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v).transpose(1, 2).reshape(b, s, dm)


def _within(got, want):
    d = (got.float() - want.float()).abs()
    ok = (d <= ATOL) | (d <= RTOL * want.float().abs())
    return ok, d.max().item()


def _props():
    return torch.cuda.get_device_properties("cuda")


def _operands(b, heads, s, hd, d, seed=0, scale=0.1):
    torch.manual_seed(seed)
    dm = heads * hd
    x = torch.randn(b, s, d, device="cuda", dtype=torch.float16)
    w_t = torch.randn(d, 3 * dm, device="cuda", dtype=torch.float16) * scale
    bias = torch.randn(3 * dm, device="cuda", dtype=torch.float16) * scale
    return x, w_t, bias


# ------------------------------------------------------------------- the kernel

SHAPES = [
    (8, 4, 128, 32, 128),     # the mainstream shape (configs 1-6)
    (4, 4, 128, 8, 32),       # config 7's width; head_dim below the MMA width
    (4, 4, 32, 32, 128),      # config 12's short sequence
    (2, 2, 128, 64, 128),     # config 10
    (3, 4, 96, 32, 128),      # S NOT a power of two: BN rounds up, the mask must handle it
    (2, 4, 16, 16, 64),       # the smallest legal tile
]


@pytest.mark.parametrize("b,heads,s,hd,d", SHAPES)
def test_kernel_matches_the_fp32_reference_at_the_locked_tolerance(b, heads, s, hd, d):
    x, w_t, bias = _operands(b, heads, s, hd, d)
    tile = choose_tile(s, d, hd, heads, _props())
    assert tile is not None
    got = fused_qkv_attention(x, w_t, bias, heads, hd, hd ** -0.5, *tile)
    ok, max_abs = _within(got, _fp32_projected_attention(x, w_t, bias, heads, hd))
    assert ok.all(), (f"{(~ok).sum().item()} elements outside tolerance, "
                      f"max_abs={max_abs:.3e}")
    # Margin, reported not asserted: the gate is the elementwise OR (L4), and bf16/fp16
    # rounding alone lands near the absolute bound on well-conditioned outputs.
    print(f"\n  B{b} H{heads} S{s} hd{hd} D{d}: margin {max_abs:.3e} "
          f"({100 * max_abs / ATOL:.0f}% of atol)")


@pytest.mark.parametrize("b,heads,s,hd,d", SHAPES)
def test_every_viable_tile_agrees_with_every_other(b, heads, s, hd, d):
    """The tile is autotuned, so ALL of them ship, not just the derived one. A tile that
    is fast and wrong is worse than one that is slow."""
    x, w_t, bias = _operands(b, heads, s, hd, d, seed=3)
    want = _fp32_projected_attention(x, w_t, bias, heads, hd)
    tiles = viable_tiles(s, d, hd, heads, _props())
    assert tiles, "no viable tile for a shape the predicate accepts"
    for tile in tiles:
        got = fused_qkv_attention(x, w_t, bias, heads, hd, hd ** -0.5, *tile)
        ok, max_abs = _within(got, want)
        assert ok.all(), (f"tile {tile}: {(~ok).sum().item()} elements outside "
                          f"tolerance, max_abs={max_abs:.3e}")


def test_kernel_produces_the_same_layout_as_transpose_reshape():
    """The kernel writes head-major [B, S, d_model] directly. If that layout is wrong the
    out_proj silently mixes heads, which no tolerance check on attention alone would
    catch -- the values are all still plausible."""
    b, heads, s, hd, d = 4, 4, 128, 32, 128
    x, w_t, bias = _operands(b, heads, s, hd, d, seed=1)
    dm = heads * hd
    qkv = F.linear(x, w_t.t().contiguous(), bias)
    q, k, v = qkv.split(dm, dim=-1)
    q = q.view(b, s, heads, hd).transpose(1, 2)
    k = k.view(b, s, heads, hd).transpose(1, 2)
    v = v.view(b, s, heads, hd).transpose(1, 2)
    sdpa = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    sdpa = sdpa.transpose(1, 2).reshape(b, s, dm)

    got = fused_qkv_attention(x, w_t, bias, heads, hd, hd ** -0.5,
                              *choose_tile(s, d, hd, heads, _props()))
    ok, max_abs = _within(got, sdpa)
    assert ok.all(), f"layout mismatch against SDPA's repack, max_abs={max_abs:.3e}"


def test_in_kernel_head_dim_padding_is_exact():
    """Padding head_dim 8 -> 16 must contribute EXACTLY zero: the padded weight columns
    and the padded bias lanes both load as zero, so the padded Q/K/V lanes are zero and
    a zero row of K times anything is zero."""
    b, heads, s, hd, d = 4, 4, 64, 8, 32
    x, w_t, bias = _operands(b, heads, s, hd, d, seed=2)
    got = fused_qkv_attention(x, w_t, bias, heads, hd, hd ** -0.5,
                              *choose_tile(s, d, hd, heads, _props()))
    ok, max_abs = _within(got, _fp32_projected_attention(x, w_t, bias, heads, hd))
    assert ok.all(), f"padding is not exact, max_abs={max_abs:.3e}"
    assert padded_head_dim(8) == 16 and padded_head_dim(32) == 32


def test_causal_triangle_is_the_reference_triangle():
    """Row 0 attends to itself alone. If the mask were off by one this is the only test
    that would notice, and it needs no tolerance at all."""
    b, heads, s, hd, d = 2, 4, 32, 32, 128
    x, w_t, bias = _operands(b, heads, s, hd, d, seed=4)
    got = fused_qkv_attention(x, w_t, bias, heads, hd, hd ** -0.5,
                              *choose_tile(s, d, hd, heads, _props()))
    # With one visible key, softmax is 1.0 and the output row IS v[0].
    dm = heads * hd
    v0 = ((x[:, :1, :].float() @ w_t.float()) + bias.float())[:, 0, 2 * dm:]
    ok, max_abs = _within(got[:, 0, :], v0)
    assert ok.all(), f"query 0 does not equal v[0]; the triangle is wrong ({max_abs:.3e})"


@pytest.mark.parametrize("b,heads,s,hd,d", SHAPES)
def test_the_fusion_is_the_only_difference_from_the_parent_kernel(b, heads, s, hd, d):
    """Isolates exactly the half that is new. v23's kernel fed by `F.linear` is the path
    this candidate replaces and is separately tested against the fp32 reference, so
    agreement with it says the in-kernel projection -- the weight transpose, the per-head
    column slice, the fp32 accumulate, the bias, the fp16 round -- is right, without the
    softmax's error standing in the way."""
    from bench.kernels.attn_single_tile import (choose_tile as v23_choose_tile,
                                                single_tile_attention)
    x, w_t, bias = _operands(b, heads, s, hd, d, seed=6)
    p = _props()
    v23_tile = v23_choose_tile(s, hd, p.regs_per_multiprocessor,
                               p.max_threads_per_multi_processor, p.warp_size)
    if v23_tile is None:
        pytest.skip("the parent kernel declines this shape")
    want = single_tile_attention(F.linear(x, w_t.t().contiguous(), bias),
                                 heads, hd, hd ** -0.5, *v23_tile)
    got = fused_qkv_attention(x, w_t, bias, heads, hd, hd ** -0.5,
                              *choose_tile(s, d, hd, heads, p))
    ok, max_abs = _within(got, want)
    assert ok.all(), (f"the fused kernel disagrees with GEMM + parent kernel: "
                      f"{(~ok).sum().item()} elements, max_abs={max_abs:.3e}")


# ------------------------------------------------------------------ the predicate

def test_predicate_source_names_no_config_id_and_no_announced_shape():
    """A dispatch that does not respond to the device is a hardcoded table wearing a
    costume (L28). Rule 2: shapes and MEASURED device properties only."""
    src = (Path(__file__).resolve().parents[2]
           / "bench/kernels/attn_qkv_fused.py").read_text()
    code = "\n".join(line for line in src.splitlines()
                     if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[-1]          # drop the module docstring
    for fn in ("def fits", "def pays", "def moves_fewer_bytes", "def applies"):
        assert fn in body
    for bad in ("config_id", "cfg", "MATRIX", "BY_ID"):
        assert bad not in body, f"predicate source mentions {bad!r}"


@pytest.mark.parametrize("cid,expected", [
    (1, True), (2, True), (3, True), (4, True), (5, True), (6, True), (7, True),
    (8, False),      # d_model 1024: a 128x1024 input tile is 256 KB of operands
    (9, False),      # head_dim 128: a 96 KB weight slice plus the input tile
    (10, True),
    (11, False),     # 16 heads each re-read the input tile: more bytes than it saves
    (12, True),
    (13, False),     # seq_len 1024: a 1024x1024 fp32 score tile is 4 MB
    (14, False),     # seq_len 100000: likewise, by orders of magnitude
])
def test_predicate_decides_each_announced_shape_as_the_mechanism_says(cid, expected):
    c = BY_ID[cid]
    ok, why = applies(c.seq_len, c.d_model, c.head_dim, c.heads, _props())
    assert ok is expected, f"config {cid}: {why}"
    if not ok:
        assert "declined" in why and len(why) > 20, why


def test_config_11_is_declined_on_the_byte_count_not_on_capacity():
    """The reason matters: config 11 FITS comfortably (44 KB of operands against 99 KB of
    smem). It is declined because the fusion would move more bytes than it deletes, which
    is the one decline the capacity argument alone does not predict."""
    c = BY_ID[11]
    bn = next_pow2(c.seq_len)
    assert smem_resident_bytes(c.seq_len, c.d_model, c.head_dim, bn) < \
        _props().shared_memory_per_block_optin
    assert not moves_fewer_bytes(c.seq_len, c.d_model, c.head_dim, c.heads, bn)
    assert fused_read_bytes(c.seq_len, c.d_model, c.heads, bn) > \
        split_io_bytes(c.seq_len, c.d_model, c.head_dim, c.heads, bn)
    ok, why = applies(c.seq_len, c.d_model, c.head_dim, c.heads, _props())
    assert not ok and "MOVE MORE BYTES" in why


def test_a_smaller_smem_card_declines_more_without_being_retuned():
    """The 'another GPU can evaluate it' test. Halving opt-in shared memory must refuse
    head_dim 64 (config 10), which the real card accepts."""
    c = BY_ID[10]
    bn = next_pow2(c.seq_len)
    p = _props()
    big = fits(c.seq_len, c.d_model, c.head_dim, bn, 8, p.regs_per_multiprocessor,
               p.shared_memory_per_block_optin, p.warp_size)
    small = fits(c.seq_len, c.d_model, c.head_dim, bn, 8, p.regs_per_multiprocessor,
                 49152, p.warp_size)
    assert big and not small


def test_the_derived_tile_projects_k_and_v_exactly_once():
    """v23 swept block_m 64 as its best tile. That reverses here, and for a reason that
    is arithmetic rather than a sweep: a query block needs every key row, so K and V are
    re-projected once per query block, and only block_m == next_pow2(S) does it once."""
    for cid in (1, 6, 7, 10, 12):
        c = BY_ID[cid]
        bm, _w, _s = choose_tile(c.seq_len, c.d_model, c.head_dim, c.heads, _props())
        assert bm == next_pow2(c.seq_len), f"config {cid} chose block_m {bm}"
        assert projection_redundancy(c.seq_len, bm) == pytest.approx(1.0)
    assert projection_redundancy(128, 64) == pytest.approx(640 / 384)


def test_the_derived_tile_is_one_the_autotuner_can_time():
    """If the derived tile is not a member of viable_tiles, the autotuner has no incumbent
    to beat and would promote whatever it measured fastest, noise included."""
    for cid in (1, 6, 7, 10, 12):
        c = BY_ID[cid]
        tiles = viable_tiles(c.seq_len, c.d_model, c.head_dim, c.heads, _props())
        assert choose_tile(c.seq_len, c.d_model, c.head_dim, c.heads, _props()) in tiles


def test_ridge_point_is_measured_and_declining_is_the_failure_mode():
    """The arithmetic half of `pays` is priced at the device's own calibrated ridge point.
    If calibration is unavailable the ridge is 0 and every redundant tiling is REFUSED --
    a guess would be worse than not fusing."""
    assert ridge_point(_props()) > 1.0
    assert not pays(128, 128, 32, 4, 64, 0.0)     # ridge 0 -> refuse the trade
    assert pays(128, 128, 32, 4, 128, 0.0)        # redundancy 1.0 -> nothing to price


def test_non_power_of_two_d_model_is_declined_rather_than_miscomputed():
    """`tl.arange` addresses the contraction axis, so a non-power-of-two d_model would be
    silently truncated. It must be refused, not rounded."""
    assert not is_pow2(96) and is_pow2(128)
    ok, why = applies(128, 96, 32, 3, _props())
    assert not ok and "power of two" in why


# ------------------------------------------------------------------ the candidate

def _build(name):
    ref = _ref_module()
    return REGISTRY[name].build(ref.BaselineTransformer), ref


def _run_pair(cfg_kwargs, causal, seed=0):
    """Baseline (fp32, unmodified) and candidate on identical weights and input."""
    ref = _ref_module()
    cfg = ref.TransformerConfig(causal=causal, **cfg_kwargs)
    torch.manual_seed(seed)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v27_qkv_fused_attn"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    cand.load_state_dict(base.state_dict(), strict=True)
    x = torch.randn(cfg_kwargs["batch_size"], cfg_kwargs["seq_len"],
                    cfg_kwargs["d_model"], device="cuda")
    with torch.inference_mode():
        want = base(x)
        got = cand(x)
    return got, want, cand


CFG = dict(batch_size=8, d_model=128, num_heads=4, seq_len=128, num_layers=4,
           ffn_dim=128)


def test_candidate_matches_the_fp32_baseline_at_the_locked_tolerance():
    got, want, cand = _run_pair(CFG, causal=True)
    ok, max_abs = _within(got, want)
    assert cand.qkv_fused_used, cand.qkv_fused_reason
    assert ok.all(), (f"{(~ok).sum().item()} of {ok.numel()} elements outside tolerance, "
                      f"max_abs={max_abs:.3e}; {cand.qkv_fused_reason}")
    print(f"\n  end-to-end margin {max_abs:.3e} ({100 * max_abs / ATOL:.0f}% of atol); "
          f"{cand.qkv_fused_reason}")


def test_candidate_honours_config_causal_like_its_parent():
    """v5 through v23 returned three quarters of their output wrong on the harness's own
    DEFAULT setting (finding 32, L42). Whatever v27 adds must not reintroduce that."""
    got, want, cand = _run_pair(CFG, causal=False)
    ok, max_abs = _within(got, want)
    assert not cand.qkv_fused_used
    assert "non-causal" in cand.qkv_fused_reason
    assert ok.all(), (f"non-causal: {(~ok).sum().item()} elements outside tolerance, "
                      f"max_abs={max_abs:.3e}")


def test_the_declined_shape_falls_back_and_is_still_right():
    """Config 11's shape: the predicate refuses it, so v23's kernel must run instead and
    the answer must be correct. A decline that silently produces garbage is worse than no
    candidate at all."""
    cfg = dict(CFG, num_heads=16)
    got, want, cand = _run_pair(cfg, causal=True, seed=1)
    ok, max_abs = _within(got, want)
    assert not cand.qkv_fused_used
    assert "MOVE MORE BYTES" in cand.qkv_fused_reason, cand.qkv_fused_reason
    assert ok.all(), (f"fallback path: {(~ok).sum().item()} elements outside tolerance, "
                      f"max_abs={max_abs:.3e}")


def test_the_wide_model_falls_back_and_is_still_right():
    """Config 8's width, where the input tile alone is 256 KB of operands."""
    cfg = dict(CFG, batch_size=2, d_model=1024, ffn_dim=1024)
    got, want, cand = _run_pair(cfg, causal=True, seed=2)
    ok, max_abs = _within(got, want)
    assert not cand.qkv_fused_used and "shared memory" in cand.qkv_fused_reason
    assert ok.all(), f"wide-model fallback wrong, max_abs={max_abs:.3e}"


def test_a_function_of_its_input_depends_on_its_input():
    """The crudest invariant there is, and the one that caught v12's empty-graph staleness
    (L23/L25). Two different inputs must not produce identical output, and the returned
    tensor must survive the next call."""
    ref = _ref_module()
    cfg = ref.TransformerConfig(causal=True, **CFG)
    torch.manual_seed(0)
    cand = REGISTRY["v27_qkv_fused_attn"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    x1 = torch.randn(CFG["batch_size"], CFG["seq_len"], CFG["d_model"], device="cuda")
    x2 = torch.randn_like(x1)
    with torch.inference_mode():
        y1 = cand(x1).clone()
        y1_again = cand(x1)
        y2 = cand(x2)
        assert not torch.equal(y1, y2), "different inputs produced identical output"
        assert torch.equal(y1, y1_again.detach()), "same input produced different output"
        assert torch.equal(y1, y1.clone()), "the returned tensor was mutated"


def test_the_registry_declares_the_parent_the_branch_was_cut_from():
    spec = REGISTRY["v27_qkv_fused_attn"]
    assert spec.parent == "v26_causal_correct"
    assert spec.generation == 27
    assert REGISTRY[spec.parent].generation == 26
