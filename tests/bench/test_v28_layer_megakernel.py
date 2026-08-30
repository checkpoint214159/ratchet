"""v28: the whole layer in one Triton launch. Correctness, the predicate, the fallback.

The tolerance here is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR, judged
by failed elements and never by max_abs alone (L4). It is not widened anywhere in this
file. `max_abs` is reported alongside pass/fail because a candidate at 90% of budget and
one at 40% are not equally correct (L26).

Two properties get more attention than usual, both because the parent's history demands
it. The kernel is the first thing in this lineage that computes a whole layer, so an
arithmetic slip anywhere inside it is invisible from outside: hence the per-stage
exactness checks (causality, head_dim padding, the per-head re-association of the output
projection). And the candidate holds decided state across calls, so L23/L25's invariance
checks apply -- a stale buffer of the right shape and dtype is the failure mode this
project's correctness machinery is structurally blind to.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels import layer_fused as lf
from bench.kernels.layer_fused import (applies, fits, fused_layer, pays, programs,
                                       register_bytes, select_tile, sm_utilization,
                                       viable_tiles)

ATOL, RTOL = 2e-3, 2e-2


def _ref_module():
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v28", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v28"] = m
    spec.loader.exec_module(m)
    return m


def _within(got, want):
    d = (got.float() - want.float()).abs()
    ok = (d <= ATOL) | (d <= RTOL * want.float().abs())
    return ok, d.max().item()


def _props():
    return torch.cuda.get_device_properties("cuda")


# ---------------------------------------------------------------- the kernel itself

def _block(d, f, heads, seed=0):
    """One layer, built by the REFERENCE's own module with the reference's own init.

    Transcribing the layer here and inventing a weight distribution for it would test the
    transcription and the distribution as much as the kernel -- and an invented
    distribution that puts more output elements near zero than the real one turns the
    tolerance's relative bound into a seed lottery, which is a documented hazard on
    shapes this size (tests/golden/test_reference_floor.py). So the oracle is
    `BaselineTransformerBlock.forward` run in fp32, on parameters it initialised itself.
    """
    ref = _ref_module()
    torch.manual_seed(seed)
    return ref.BaselineTransformerBlock(d, heads, f).cuda().eval()


def _fp32_layer(x, blk, causal=True):
    with torch.no_grad():
        return blk(x, None, causal)


def _call(x, blk, heads, tile):
    bm, warps = tile
    d = x.shape[-1]
    dh = d // heads
    h = torch.float16
    a = blk.attention
    return fused_layer(
        x, blk.norm1.weight.float(), blk.norm1.bias.float(),
        blk.norm2.weight.float(), blk.norm2.bias.float(),
        torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).t().contiguous().to(h),
        torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).to(h),
        a.out_proj.weight.t().contiguous().to(h), a.out_proj.bias.to(h),
        blk.ffn_in.weight.t().contiguous().to(h), blk.ffn_in.bias.to(h),
        blk.ffn_out.weight.t().contiguous().to(h), blk.ffn_out.bias.to(h),
        heads, dh, dh ** -0.5, float(blk.norm1.eps), bm, warps)


def _tiles(s, d, f, dh, heads, b):
    p = _props()
    return viable_tiles(s, d, f, dh, heads, b, p.regs_per_multiprocessor,
                        p.multi_processor_count, p.warp_size)


@pytest.mark.parametrize("b,s,d,f,heads", [
    (66, 128, 128, 128, 4),     # the mainstream shape
    (66, 128, 32, 32, 4),       # head_dim 8: below the MMA width, padded in-kernel
    (66, 128, 128, 128, 16),    # head_dim 8 again, sixteen heads
    (66, 128, 128, 128, 2),     # head_dim 64
    (66, 32, 128, 128, 4),      # short sequence
    (66, 64, 64, 64, 4),        # a shape the announced matrix does not contain
])
def test_kernel_matches_the_fp32_reference_at_the_locked_tolerance(b, s, d, f, heads):
    torch.manual_seed(3)
    x = torch.randn(b, s, d, device="cuda")
    blk = _block(d, f, heads)
    tiles = _tiles(s, d, f, d // heads, heads, b)
    assert tiles, "no viable tile for a shape this test expects the kernel to handle"
    got = _call(x, blk, heads, tiles[0])
    want = _fp32_layer(x, blk)
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


def test_the_kernel_is_at_least_as_accurate_as_the_path_it_replaces():
    """The frontier rounds the attention output to fp16 before adding it to the fp32
    residual; this kernel does not. Finding 08 made the fp32 residual load-bearing, so a
    fusion that removes a rounding step from it should not be LESS accurate -- and if it
    ever becomes so, that is a defect, not a tolerance question."""
    from bench.kernels.attn_single_tile import autotune_tile, single_tile_attention
    from bench.kernels.ffn_fused import fused_ffn

    b, s, d, f, heads = 66, 128, 128, 128, 4
    dh = d // heads
    torch.manual_seed(4)
    x = torch.randn(b, s, d, device="cuda")
    blk = _block(d, f, heads, seed=4)
    want = _fp32_layer(x, blk)
    ours = _call(x, blk, heads, _tiles(s, d, f, dh, heads, b)[0])

    # v23/v26's own op sequence for one layer -- NOT an eager decomposition (L41).
    h = torch.float16
    a = blk.attention
    n1 = F.layer_norm(x, (d,), blk.norm1.weight, blk.norm1.bias, blk.norm1.eps)
    qkv = F.linear(n1.to(h),
                   torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).to(h),
                   torch.cat([a.q_proj.bias, a.k_proj.bias, a.v_proj.bias]).to(h))
    (bm, wp, st), _how = autotune_tile(s, dh, heads, b, "cuda")
    ctx = single_tile_attention(qkv, heads, dh, dh ** -0.5, bm, wp, st)
    xr = x + F.linear(ctx, a.out_proj.weight.to(h), a.out_proj.bias.to(h)).float()
    n2 = F.layer_norm(xr, (d,), blk.norm2.weight, blk.norm2.bias,
                      blk.norm2.eps).to(h).view(-1, d)
    theirs = fused_ffn(n2, xr.view(-1, d),
                       blk.ffn_in.weight.t().contiguous().to(h), blk.ffn_in.bias.to(h),
                       blk.ffn_out.weight.t().contiguous().to(h),
                       blk.ffn_out.bias.to(h), 64, 8).view(b, s, d)

    ours_ok, ours_max = _within(ours, want)
    theirs_ok, theirs_max = _within(theirs, want)
    assert ours_ok.all(), f"megakernel outside tolerance, max_abs={ours_max:.3e}"
    assert (~ours_ok).sum() <= (~theirs_ok).sum(), (
        f"the fused layer is LESS accurate than the path it replaces: "
        f"{(~ours_ok).sum().item()} vs {(~theirs_ok).sum().item()} failed elements "
        f"(max_abs {ours_max:.3e} vs {theirs_max:.3e})")


def test_causality_is_exact_not_approximate():
    """A masked entry must carry EXACTLY zero weight, which is what the reference's
    `triu(diagonal=1)` computes. Perturbing a key that only future queries can see must
    change nothing at all -- not 'change little'."""
    b, s, d, f, heads = 66, 32, 128, 128, 4
    torch.manual_seed(5)
    x = torch.randn(b, s, d, device="cuda")
    blk = _block(d, f, heads, seed=5)
    tile = _tiles(s, d, f, d // heads, heads, b)[0]
    a = _call(x, blk, heads, tile)
    x2 = x.clone()
    x2[:, -1, :] += 100.0            # the last position: no earlier query may see it
    bpert = _call(x2, blk, heads, tile)
    assert torch.equal(a[:, :-1], bpert[:, :-1]), \
        "a future key changed a past query's output -- causality is not exact"


def test_head_dim_padding_contributes_exactly_zero():
    """head_dim 8 is below sm_89's m16n8k16 MMA width and is padded to 16 in-kernel with
    masked loads that read 0.0. The padded lanes must contribute nothing: a zero row of K
    times anything is zero. Compared against head_dim 8 computed by the fp32 reference,
    which has no padding at all."""
    b, s, d, f, heads = 66, 128, 128, 128, 16       # head_dim 8
    torch.manual_seed(6)
    x = torch.randn(b, s, d, device="cuda")
    blk = _block(d, f, heads, seed=6)
    tiles = _tiles(s, d, f, 8, heads, b)
    ok, max_abs = _within(_call(x, blk, heads, tiles[0]), _fp32_layer(x, blk))
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


def test_the_query_tile_does_not_change_the_answer():
    """BM is a tuning knob. Splitting a sequence across programs recomputes K and V and
    re-associates nothing, so every legal tile must produce the same answer to within the
    tolerance -- if it does not, the tuner is choosing between right and wrong."""
    b, s, d, f, heads = 66, 128, 32, 32, 4
    torch.manual_seed(7)
    x = torch.randn(b, s, d, device="cuda")
    blk = _block(d, f, heads, seed=7)
    want = _fp32_layer(x, blk)
    tiles = _tiles(s, d, f, d // heads, heads, b)
    assert len({t[0] for t in tiles}) > 1, "this shape should admit more than one BM"
    for t in tiles:
        ok, max_abs = _within(_call(x, blk, heads, t), want)
        assert ok.all(), f"tile {t}: {(~ok).sum().item()} outside tolerance, {max_abs:.3e}"


# ------------------------------------------------------------------- the predicate

def test_predicate_declines_shapes_whose_working_set_cannot_be_held():
    p = _props()
    # d_model 1024 (config 8's shape): the key rows' normalized tile alone is 256 KB.
    ok, why = applies(128, 1024, 1024, 256, 4, 64, p)
    assert not ok and "no tile fits" in why, why
    # seq_len 1024 (config 13's shape).
    ok, why = applies(1024, 128, 128, 32, 4, 64, p)
    assert not ok and "no tile fits" in why, why


def test_predicate_declines_shapes_that_cannot_fill_the_measured_machine():
    p = _props()
    ok, why = applies(128, 128, 128, 32, 4, 1, p)       # config 2's batch of 1
    assert not ok and "utilised" in why, why
    ok, why = applies(128, 128, 128, 32, 4, 4, p)       # config 3's batch of 4
    assert not ok and "utilised" in why, why


def test_predicate_accepts_the_shapes_it_was_built_for():
    p = _props()
    for s, d, f, dh, heads, b in [(128, 128, 128, 32, 4, 10000),   # config 6
                                  (128, 32, 32, 8, 4, 64),         # config 7
                                  (128, 128, 128, 64, 2, 64),      # config 10
                                  (32, 128, 128, 32, 4, 64)]:      # config 12
        ok, why = applies(s, d, f, dh, heads, b, p)
        assert ok, f"({s},{d},{f},{dh},{heads},{b}): {why}"


def test_predicate_responds_to_the_device_and_not_to_the_shape_alone():
    """A dispatch that does not change when the device changes is a hardcoded table
    wearing a costume (L28). Both halves of the predicate must move."""
    p = _props()
    # Quartering the register file must make the mainstream shape unholdable.
    assert fits(128, 128, 128, 32, 4, 64, 8, p.regs_per_multiprocessor, p.warp_size)
    assert not fits(128, 128, 128, 32, 4, 64, 8, p.regs_per_multiprocessor // 4,
                    p.warp_size)
    # Doubling the SM count must make a batch that filled this card stop filling it.
    assert pays(64, 128, 128, p.multi_processor_count)
    assert not pays(64, 128, 128, p.multi_processor_count * 4)


def test_utilization_is_computed_over_whole_waves():
    assert sm_utilization(0, 66) == 0.0
    assert sm_utilization(66, 66) == 1.0
    assert sm_utilization(67, 66) == pytest.approx(67 / 132)
    assert programs(64, 128, 32) == 64 * 4


def test_predicate_source_contains_no_config_ids_or_announced_shapes():
    """CLAUDE.md rule 2. The predicate may read shapes and measured device properties and
    nothing else -- no config ids, no announced constants smuggled in as literals."""
    import ast

    tree = ast.parse(Path(lf.__file__).read_text())
    # Strip every docstring and let `unparse` drop the comments: prose may name the
    # configs it declines, EXECUTABLE CODE may not. Checking the raw text would let a
    # comment fail the test and, worse, let a literal hide inside a docstring.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                node.body.pop(0)
    code = ast.unparse(tree)
    for banned in ("config_id", "cfg", "10000", "100000", "seq_len ==", "d_model ==",
                   "batch_size ==", "== 128", "== 1024"):
        assert banned not in code, f"the dispatch code mentions {banned!r}"


def test_the_compiler_is_the_authority_on_its_own_resources():
    """`register_bytes` models the register file; it does not model the shared memory
    Triton stages `tl.dot` operands through. At head_dim 128 the registers fit and the
    smem does not, and only the compiler knows. `select_tile` must survive that."""
    p = _props()
    assert fits(128, 128, 128, 128, 1, 32, 8, p.regs_per_multiprocessor, p.warp_size)
    with pytest.raises(ValueError):
        select_tile(128, 128, 128, 128, 1, 64, "cuda")


def test_tile_selection_returns_something_legal_and_says_how_it_chose():
    tile, how = select_tile(128, 32, 32, 8, 4, 64, "cuda")
    assert tile in viable_tiles(128, 32, 32, 8, 4, 64,
                                _props().regs_per_multiprocessor,
                                _props().multi_processor_count, _props().warp_size)
    assert "spill" in how, how


# ------------------------------------------------------------------- end to end

def _pair(cfg, seed=0):
    ref = _ref_module()
    torch.manual_seed(seed)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY["v28_layer_megakernel"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    return base, cand


@pytest.mark.gpu
def test_candidate_matches_the_baseline_within_the_locked_tolerance():
    torch._dynamo.reset()          # L36: a shared Dynamo cache can silently un-compile
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=66, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(21)
    x = torch.randn(66, 128, 128, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.layer_fused_used, cand.layer_fused_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_candidate_is_correct_at_head_dim_8_where_the_kernel_pads():
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=66, seq_len=128, d_model=32, num_heads=4,
                                ffn_dim=32, num_layers=4, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(22)
    x = torch.randn(66, 128, 32, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.layer_fused_used, cand.layer_fused_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_the_declined_path_reports_honestly_and_is_still_correct():
    """d_model 1024. The kernel must not run, the candidate must SAY it did not run, and
    v26's path must still be right. An untuned fallback presented as a tuned one is the
    failure v14 was built to prevent."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=66, seq_len=128, d_model=1024, num_heads=4,
                                ffn_dim=1024, num_layers=1, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(23)
    x = torch.randn(66, 128, 1024, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.layer_fused_used is False
    assert "declined" in cand.layer_fused_reason, cand.layer_fused_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"fallback path wrong: max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_the_declined_path_covers_a_batch_that_cannot_fill_the_card():
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=1, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(24)
    x = torch.randn(1, 128, 128, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.layer_fused_used is False
    assert "utilised" in cand.layer_fused_reason, cand.layer_fused_reason
    ok, _ = _within(got, want)
    assert ok.all()


@pytest.mark.gpu
def test_a_padded_input_falls_back_and_is_correct():
    """The kernel writes whole rows and does not apply the reference's per-token zeroing,
    so a masked input must take the parent's path -- the same restriction v17's FFN
    megakernel carries. Every measurement before finding 11 was blind to this path (L5)."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=66, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(25)
    x = torch.randn(66, 128, 128, device="cuda")
    lengths = torch.randint(1, 129, (66,), device="cuda")
    mask = torch.arange(128, device="cuda")[None, :] < lengths[:, None]
    with torch.no_grad():
        want, got = base(x, mask), cand(x, mask)
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_a_non_causal_input_is_exactly_the_baseline():
    """Finding 32: the reference benchmark's own DEFAULT is causal=False, and this kernel
    masks the causal triangle unconditionally. Non-causal must delegate, and must say so."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=66, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=False)
    base, cand = _pair(cfg)
    torch.manual_seed(26)
    x = torch.randn(66, 128, 128, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.layer_fused_used is False
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_output_depends_on_the_input():
    """L23/L25: a stateful candidate can return a stale static buffer that is the right
    shape, the right dtype and silently wrong. No tolerances needed -- two different
    inputs must not give identical output."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=66, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(27)
    a = torch.randn(66, 128, 128, device="cuda")
    b = torch.randn(66, 128, 128, device="cuda")
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
    cfg = ref.TransformerConfig(batch_size=66, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(28)
    a = torch.randn(66, 128, 128, device="cuda")
    b = torch.randn(66, 128, 128, device="cuda")
    with torch.no_grad():
        ya = cand(a)
        snapshot = ya.clone()
        cand(b)
    assert torch.equal(ya, snapshot), "the previous return value was mutated in place"


@pytest.mark.gpu
def test_the_megakernel_actually_runs_and_replaces_both_predecessors():
    """L36: a test can pass because its subject was never built. Everything above would
    still be green if the candidate had quietly fallen back. Observe the mechanism: our
    layer kernel must appear in the CUDA profile, and the two kernels it subsumes --
    `_attn_single_tile` and `_ffn_block` -- must NOT."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=66, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(29)
    x = torch.randn(66, 128, 128, device="cuda")
    with torch.inference_mode():
        for _ in range(3):
            cand(x)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                cand(x)

    names = {e.key for e in prof.key_averages()}
    assert any("layer_block" in n for n in names), \
        f"the fused layer kernel never ran; kernels seen: {sorted(names)}"
    assert not any("attn_single_tile" in n for n in names), \
        "the attention kernel is still running -- it was supposed to be absorbed"
    assert not any("ffn_block" in n for n in names), \
        "the FFN kernel is still running -- it was supposed to be absorbed"
