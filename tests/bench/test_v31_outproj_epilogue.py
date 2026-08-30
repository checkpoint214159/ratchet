"""v31: the out-projection, the fp32 widen, the mask and the fp32 residual add absorbed
into the attention kernel's epilogue.

The tolerance here is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR, judged
by failed elements and never by max_abs alone (L4). It is not widened anywhere in this
file. Margin is reported alongside pass/fail (L26).

Three things this file is deliberately built to catch, because each is a failure mode this
project has already been bitten by:

  * L36 -- a test can pass because its subject was never built. Every end-to-end test here
    asserts `outproj_used` (or asserts it is False and says why), and one asserts the
    kernel's name appears in the CUDA profile, so "green because it silently fell back"
    is not available.
  * L42 -- `config.causal` is a separate test case from the value the matrix implies, and
    it is the more dangerous one because it is the reference's DEFAULT.
  * L39 -- the claim the sweep cannot see (accuracy margin) gets a bespoke falsifier
    rather than being asserted in prose.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels.attn_outproj import (applies, attn_outproj, autotune_tile,
                                        choose_tile, fits, pays, programs,
                                        register_bytes, resident_blocks, viable_tiles)
from bench.kernels.attn_single_tile import (choose_tile as v23_choose_tile,
                                            single_tile_attention)

ATOL, RTOL = 2e-3, 2e-2
KERNEL_SRC = Path(__file__).resolve().parents[2] / "bench/kernels/attn_outproj.py"


def _ref_module():
    p = (Path(__file__).resolve().parents[2]
         / "benchmarks/reference/torch_transformer_benchmark.py")
    spec = importlib.util.spec_from_file_location("ref_v31", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v31"] = m
    spec.loader.exec_module(m)
    return m


def _within(got, want):
    d = (got.float() - want.float()).abs()
    ok = (d <= ATOL) | (d <= RTOL * want.float().abs())
    return ok, d.max().item()


def _props():
    return torch.cuda.get_device_properties("cuda")


def _fp32_segment(qkv, res, w_t, bias, heads, hd, mask=None):
    """The reference's own formulation, in fp32: attention, the head-major repack, the
    projection, the mask, and the fp32 residual add. NOT SDPA -- that carries its own
    error and is not what the oracle compares against."""
    b, s, _ = qkv.shape
    dm = heads * hd
    q, k, v = qkv.float().split(dm, dim=-1)
    q = q.view(b, s, heads, hd).transpose(1, 2)
    k = k.view(b, s, heads, hd).transpose(1, 2)
    v = v.view(b, s, heads, hd).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
    causal = torch.ones(s, s, device=qkv.device, dtype=torch.bool).triu(diagonal=1)
    ctx = (torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1) @ v)
    ctx = ctx.transpose(1, 2).reshape(b, s, dm)
    o = ctx @ w_t.float() + bias.float()
    if mask is not None:
        o = o.masked_fill(~mask[..., None], 0)
    return (res.view(b, s, dm) + o).reshape(b * s, dm)


def _any_tile(s, hd, dm, heads, b):
    """A legal tile for a shape the predicate may decline for occupancy reasons. The
    kernel must be CORRECT wherever it is legal; whether it is fast is the predicate's
    business and is tested separately."""
    p = _props()
    t = choose_tile(s, hd, dm, heads, b, p.regs_per_multiprocessor,
                    p.max_threads_per_multi_processor, p.multi_processor_count,
                    p.warp_size)
    if t is not None:
        return t
    bm = 64
    while bm >= 16:
        for w in (2, 4, 8):
            if fits(s, hd, dm, heads, bm, w, p.regs_per_multiprocessor, p.warp_size):
                return (bm, w, 1)
        bm //= 2
    pytest.skip("no legal tile on this device")


# ------------------------------------------------------------------- the kernel

@pytest.mark.parametrize("b,heads,s,hd", [
    (8, 4, 128, 32),      # the mainstream shape
    (4, 4, 128, 8),       # head_dim below the MMA width, padded inside the kernel
    (4, 16, 128, 8),      # sixteen heads: sixteen split-K steps of the projection
    (4, 4, 32, 32),       # short sequence
    (2, 2, 128, 64),
    (3, 4, 96, 32),       # NOT a power of two: BN rounds up, the mask must handle it
    (2, 4, 16, 16),       # the smallest legal tile
])
def test_kernel_matches_the_fp32_reference_at_the_locked_tolerance(b, heads, s, hd):
    torch.manual_seed(0)
    dm = heads * hd
    qkv = torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16)
    w_t = torch.randn(dm, dm, device="cuda", dtype=torch.float16) * dm ** -0.5
    bias = torch.randn(dm, device="cuda", dtype=torch.float16) * 0.1
    res = torch.randn(b * s, dm, device="cuda", dtype=torch.float32)
    bm, w, st = _any_tile(s, hd, dm, heads, b)
    got = attn_outproj(qkv, res, w_t, bias, None, heads, hd, hd ** -0.5, bm, w, st)
    ok, max_abs = _within(got, _fp32_segment(qkv, res, w_t, bias, heads, hd))
    assert ok.all(), (f"{(~ok).sum().item()} elements outside tolerance, "
                      f"max_abs={max_abs:.3e}")
    assert max_abs < ATOL, f"margin: {max_abs:.3e} of {ATOL:.0e} ({100*max_abs/ATOL:.0f}%)"


@pytest.mark.parametrize("b,heads,s,hd", [(8, 4, 128, 32), (4, 16, 128, 8),
                                          (4, 4, 32, 32)])
def test_the_in_kernel_mask_is_exactly_the_masked_fill_it_replaces(b, heads, s, hd):
    """`g24` declined the padded path wholesale. This kernel implements it, so the
    equivalence has to be pinned: the mask must zero the PROJECTION OUTPUT before the
    residual add, which leaves the residual untouched on an invalid token -- not zero the
    result, and not zero the context. Finding 11 and L5 are why the padded path matters."""
    torch.manual_seed(1)
    dm = heads * hd
    qkv = torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16)
    w_t = torch.randn(dm, dm, device="cuda", dtype=torch.float16) * dm ** -0.5
    bias = torch.randn(dm, device="cuda", dtype=torch.float16) * 0.1
    res = torch.randn(b * s, dm, device="cuda", dtype=torch.float32)
    lengths = torch.randint(1, s + 1, (b,), device="cuda")
    mask = torch.arange(s, device="cuda")[None, :] < lengths[:, None]
    bm, w, st = _any_tile(s, hd, dm, heads, b)
    got = attn_outproj(qkv, res, w_t, bias, mask, heads, hd, hd ** -0.5, bm, w, st)
    ok, max_abs = _within(got, _fp32_segment(qkv, res, w_t, bias, heads, hd, mask))
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"
    # ... and an invalid token must carry the residual EXACTLY, bit for bit.
    invalid = ~mask.reshape(-1)
    assert torch.equal(got[invalid], res[invalid]), \
        "an invalid token must be left holding exactly the residual"


def test_in_kernel_head_dim_padding_is_exact_on_both_sides_of_the_projection():
    """Padding head_dim 8 -> 16 zeroes the context lanes AND the weight rows they would
    multiply. Both sides must be zero or the projection silently mixes in garbage from the
    next head's weight rows -- a failure whose output is entirely plausible."""
    torch.manual_seed(2)
    b, heads, s, hd = 4, 4, 64, 8
    dm = heads * hd
    qkv = torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16)
    w_t = torch.randn(dm, dm, device="cuda", dtype=torch.float16) * dm ** -0.5
    bias = torch.zeros(dm, device="cuda", dtype=torch.float16)
    res = torch.zeros(b * s, dm, device="cuda", dtype=torch.float32)
    bm, w, st = _any_tile(s, hd, dm, heads, b)
    got = attn_outproj(qkv, res, w_t, bias, None, heads, hd, hd ** -0.5, bm, w, st)
    ok, max_abs = _within(got, _fp32_segment(qkv, res, w_t, bias, heads, hd))
    assert ok.all(), f"max_abs={max_abs:.3e}"


def test_the_causal_mask_is_exact_not_approximate():
    """Query 0 attends to key 0 only, so its projected output is exactly
    `res_0 + v_0 @ W + bias`. If the triangle leaked, row 0 would be an average over the
    whole sequence and would miss by orders of magnitude rather than by a tolerance."""
    torch.manual_seed(3)
    b, heads, s, hd = 1, 1, 64, 32
    dm = heads * hd
    qkv = torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16)
    w_t = torch.randn(dm, dm, device="cuda", dtype=torch.float16) * dm ** -0.5
    bias = torch.randn(dm, device="cuda", dtype=torch.float16) * 0.1
    res = torch.randn(b * s, dm, device="cuda", dtype=torch.float32)
    bm, w, st = _any_tile(s, hd, dm, heads, b)
    got = attn_outproj(qkv, res, w_t, bias, None, heads, hd, hd ** -0.5, bm, w, st)
    want = res[0] + (qkv[0, 0, 2 * dm:].float() @ w_t.float() + bias.float())
    assert torch.allclose(got[0], want, atol=1e-3), \
        "query 0 must be exactly value 0, projected"


def test_the_residual_add_and_the_output_are_fp32():
    """Finding 08: an fp16 residual failed 12 of 14 configs. The accumulator, the bias,
    the residual load, the sum and the store are all fp32, and the returned dtype is the
    observable end of that chain."""
    torch.manual_seed(4)
    b, heads, s, hd = 4, 4, 64, 32
    dm = heads * hd
    qkv = torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16)
    w_t = torch.randn(dm, dm, device="cuda", dtype=torch.float16) * dm ** -0.5
    bias = torch.zeros(dm, device="cuda", dtype=torch.float16)
    res = torch.randn(b * s, dm, device="cuda", dtype=torch.float32)
    bm, w, st = _any_tile(s, hd, dm, heads, b)
    out = attn_outproj(qkv, res, w_t, bias, None, heads, hd, hd ** -0.5, bm, w, st)
    assert out.dtype is torch.float32
    src = KERNEL_SRC.read_text()
    assert "tl.zeros((BM, DM), dtype=tl.float32)" in src, \
        "the projection accumulator must be fp32"


# ------------------------------------------------------- the accuracy claim (L39)

@pytest.mark.parametrize("b,heads,s,hd", [(64, 4, 128, 32), (64, 16, 128, 8),
                                          (64, 4, 32, 32)])
def test_the_fusion_is_more_accurate_than_the_split_path_it_replaces(b, heads, s, hd):
    """THE FALSIFIER for the claim the sweep cannot see (L39).

    The split path rounds the projection to fp16 before `.float()` widens it again; the
    fused path never materializes it. Both arms round `ctx` to fp16 -- that is a
    tensor-core operand and is unavoidable -- so this measures exactly one deleted
    rounding step, against an fp64 reference computed from `qkv`.

    Note the reference matters: scored against a reference that takes the fp16 `ctx` as
    GIVEN, the same fusion looks 7x-1177x tighter, which is the shape of `g24`'s reported
    ~600x. That number credits the fusion with removing an error term it does not touch.
    The whole-segment number below is the honest one and it is much smaller.
    """
    torch.manual_seed(5)
    dm = heads * hd
    qkv = torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16)
    w_t = torch.randn(dm, dm, device="cuda", dtype=torch.float16) * dm ** -0.5
    bias = torch.randn(dm, device="cuda", dtype=torch.float16) * 0.1
    res = torch.randn(b * s, dm, device="cuda", dtype=torch.float32)

    bm, w, st = _any_tile(s, hd, dm, heads, b)
    fused = attn_outproj(qkv, res, w_t, bias, None, heads, hd, hd ** -0.5, bm, w, st)

    p = _props()
    vbm, vw, vst = v23_choose_tile(s, hd, p.regs_per_multiprocessor,
                                   p.max_threads_per_multi_processor, p.warp_size)
    ctx = single_tile_attention(qkv, heads, hd, hd ** -0.5, vbm, vw, vst)
    split = res + F.linear(ctx.view(b * s, dm), w_t.t().contiguous(), bias).float()

    q, k, v = qkv.double().split(dm, dim=-1)
    q = q.view(b, s, heads, hd).transpose(1, 2)
    k = k.view(b, s, heads, hd).transpose(1, 2)
    v = v.view(b, s, heads, hd).transpose(1, 2)
    sc = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
    causal = torch.ones(s, s, device=qkv.device, dtype=torch.bool).triu(1)
    c64 = (torch.softmax(sc.masked_fill(causal, float("-inf")), -1) @ v)
    want = res.double() + (c64.transpose(1, 2).reshape(b * s, dm) @ w_t.double()
                           + bias.double())

    e_split = (split.double() - want).abs().max().item()
    e_fused = (fused.double() - want).abs().max().item()
    assert e_fused < e_split, (f"the fusion must not lose accuracy: split {e_split:.3e} "
                               f"fused {e_fused:.3e}")


# --------------------------------------------------------------- the predicate

def test_the_predicate_declines_where_the_accumulator_evicts_the_occupancy():
    """The fp32 [BM, d_model] accumulator is the whole cost of the fusion. At head_dim 64
    and above the score tile plus the accumulator leave fewer than v23's MEASURED
    crossover of 4 resident blocks, and the split path -- which is the frontier and is
    already fast -- keeps those shapes."""
    p = _props()
    big_batch = 64
    assert applies(128, 32, 4, big_batch, p)[0]
    assert applies(128, 8, 4, big_batch, p)[0]
    assert applies(128, 8, 16, big_batch, p)[0]
    for hd, heads in ((64, 2), (128, 1), (256, 4)):
        ok, why = applies(128, hd, heads, big_batch, p)
        assert not ok, f"head_dim {hd} should be declined"
        assert "declined" in why


def test_the_predicate_declines_a_grid_that_no_longer_covers_the_machine():
    """Fusing the epilogue costs a factor of `heads` in program count. `g24` measured that
    this exact crossover is SM saturation and not a token threshold, so a batch that fills
    the card under v23's grid may not fill it under this one, and that is a reason to
    decline rather than a reason to shrug."""
    p = _props()
    ok_small, why_small = applies(128, 32, 4, 1, p)
    assert not ok_small and "do not cover" in why_small
    assert applies(128, 32, 4, 1024, p)[0], "a large batch must saturate"
    # ... and the split path really does emit more programs at the same tile.
    assert programs(128, 16, 32) * 4 > programs(128, 16, 32)


def test_the_predicate_is_a_function_of_measured_device_properties():
    """CLAUDE.md rule 2. Both device numbers it reads must actually move the answer, or
    the predicate is a hardcoded table wearing a costume (the v14 lesson)."""
    big, small = 65536, 16384              # 32-bit registers per SM
    # Register file: shrink it and an accepted shape is refused.
    assert choose_tile(128, 32, 128, 4, 4096, big, 1536, 66) is not None
    assert choose_tile(128, 32, 128, 4, 4096, small, 1536, 66) is None
    # SM count: raise it and a grid that covered a small card stops covering a big one.
    assert choose_tile(128, 32, 128, 4, 32, big, 1536, 16) is not None
    assert choose_tile(128, 32, 128, 4, 32, big, 1536, 4096) is None


def test_the_predicate_source_contains_no_config_ids_or_announced_shapes():
    src = KERNEL_SRC.read_text()
    body = src.split("# ----------------------------------------------------------------- the predicate")[1]
    body = body.split("# --------------------------------------------------------------------- launcher")[0]
    code = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    for forbidden in ("config_id", "cfg", "== 128", "== 10000", "seq_len == ", "100000"):
        assert forbidden not in code, f"predicate mentions {forbidden!r}"


def test_the_working_set_accounting_grows_with_every_term_it_claims():
    """`register_bytes` is what the predicate stands on. Pin the ordering so a future edit
    cannot silently invert it: every one of the four terms must move the answer."""
    base = register_bytes(128, 32, 128, 32)
    assert register_bytes(256, 32, 128, 32) > base       # score tile
    assert register_bytes(128, 64, 128, 32) > base       # operands
    assert register_bytes(128, 32, 256, 32) > base       # accumulator + weight tile
    assert register_bytes(128, 32, 128, 64) > base       # block_m
    # ... and it must exceed v23's, because this kernel holds strictly more.
    from bench.kernels.attn_single_tile import register_bytes as v23_bytes
    assert base > v23_bytes(128, 32, 32)


def test_the_autotuner_returns_a_tile_that_fits_and_pays():
    p = _props()
    tile, why = autotune_tile(128, 32, 4, 1024)
    bm, w, st = tile
    assert fits(128, 32, 128, 4, bm, w, p.regs_per_multiprocessor, p.warp_size)
    assert pays(128, 32, 128, 1024, bm, w, p.regs_per_multiprocessor,
                p.max_threads_per_multi_processor, p.multi_processor_count, p.warp_size)
    assert "autotuned" in why or "derived" in why
    assert tile in viable_tiles(128, 32, 128, 4, 1024, p.regs_per_multiprocessor,
                                p.max_threads_per_multi_processor,
                                p.multi_processor_count, p.warp_size)


def test_the_launch_wrapper_holds_no_data_dependent_python():
    """A sibling's first screen read -18.9% because plan resolution ran inside Dynamo's
    traced region and dropped the frame to eager. Everything shape- or device-dependent is
    resolved in `_decide_outproj`; the wrapper must contain nothing that forces a graph
    break or a guard on tensor CONTENTS."""
    import ast
    tree = ast.parse(KERNEL_SRC.read_text())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "attn_outproj")
    body = fn.body[1:] if (isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body
    code = "\n".join(ast.unparse(n) for n in body)   # docstring and comments dropped
    for forbidden in (".item()", "bool(", ".is_contiguous()", "torch.cuda.get_device",
                      "do_bench", ".cpu()", ".tolist()"):
        assert forbidden not in code, f"the launch wrapper does {forbidden!r} per call"


# ------------------------------------------------------------------- end to end

def _pair(cfg, name="v31_outproj_epilogue"):
    ref = _ref_module()
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY[name].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    return base, cand


@pytest.mark.gpu
@pytest.mark.parametrize("b,s,d,h", [(64, 128, 128, 4),     # mainstream
                                     (64, 128, 32, 4),      # head_dim 8
                                     (64, 128, 128, 16),    # sixteen heads
                                     (64, 32, 128, 4)])     # short sequence
def test_candidate_matches_the_baseline_within_the_locked_tolerance(b, s, d, h):
    torch._dynamo.reset()          # L36: a shared Dynamo cache can silently un-compile
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                ffn_dim=d, num_layers=4, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(7)
    x = torch.randn(b, s, d, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.outproj_used, cand.outproj_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
@pytest.mark.parametrize("b,s,d,h,why", [
    (64, 128, 128, 2, "resident"),      # head_dim 64: the accumulator evicts occupancy
    (4, 128, 128, 4, "cover"),          # too few programs to fill the SMs
    (2, 128, 1024, 4, "legal tile"),    # head_dim 256: no legal tile, as for v23
])
def test_the_declined_paths_report_honestly_and_are_still_correct(b, s, d, h, why):
    """An untuned fallback presented as a tuned path is the failure v14 was built to
    prevent. The candidate must SAY it declined, say WHICH condition refused, and still
    be right -- v23's split path is what runs there and it is the frontier."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=b, seq_len=s, d_model=d, num_heads=h,
                                ffn_dim=d, num_layers=2, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(9)
    x = torch.randn(b, s, d, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.outproj_used is False
    assert "declined" in cand.outproj_reason and why in cand.outproj_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"fallback path wrong: max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_a_forced_fallback_is_correct():
    """The gate must be a gate. Turn the fused path off after priming and the candidate
    has to fall through to the parent's split path and stay right -- if it does not, the
    'declining costs nothing' claim above is untested wherever the predicate happens to
    accept."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(15)
    x = torch.randn(64, 128, 128, device="cuda")
    cand.outproj_used = False
    cand.outproj_reason = "declined: forced off by the test"
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.outproj_used is False
    ok, max_abs = _within(got, want)
    assert ok.all(), f"forced fallback wrong: max_abs={max_abs:.3e}"


@pytest.mark.gpu
def test_padded_input_still_takes_the_fused_path_and_is_correct():
    """v8's proof (a right-padded causal key mask is redundant) is what lets attention skip
    the mask; the projection's own `masked_fill` is what this kernel absorbs. `g24`
    declined this path wholesale -- keeping it is why the mask is in the kernel. Every
    measurement before finding 11 was blind to this path (L5)."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    base, cand = _pair(cfg)
    torch.manual_seed(11)
    x = torch.randn(64, 128, 128, device="cuda")
    lengths = torch.tensor([128, 96, 64, 1] * 16, device="cuda")
    mask = torch.arange(128, device="cuda")[None, :] < lengths[:, None]
    with torch.no_grad():
        want, got = base(x, mask), cand(x, mask)
    assert cand.outproj_used, cand.outproj_reason
    ok, max_abs = _within(got, want)
    assert ok.all(), f"{(~ok).sum().item()} outside tolerance, max_abs={max_abs:.3e}"


@pytest.mark.gpu
@pytest.mark.parametrize("causal", [True, False])
def test_config_causal_is_honoured(causal):
    """L42, inherited from v26 and re-asserted here because this kernel hardcodes the
    causal triangle and would be silently wrong without the gate. The reference's own
    DEFAULT is `causal=False`, which is the dangerous case."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=causal)
    base, cand = _pair(cfg)
    torch.manual_seed(16)
    x = torch.randn(64, 128, 128, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    ok, max_abs = _within(got, want)
    assert ok.all(), (f"causal={causal}: {(~ok).sum().item()} failed, "
                      f"max_abs={max_abs:.3e}")
    if causal:
        assert cand.outproj_used
    else:
        # Delegated to the unmodified baseline: the fused path must never have fired, and
        # no tuner probe should have been spent deciding a path that cannot be taken.
        assert cand.outproj_used is False
        assert cand.outproj_reason == "undecided"


@pytest.mark.gpu
def test_output_depends_on_the_input():
    """L23/L25: a stateful candidate can return a stale static buffer that is the right
    shape, the right dtype and silently wrong. No tolerances needed."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(12)
    a = torch.randn(64, 128, 128, device="cuda")
    b = torch.randn(64, 128, 128, device="cuda")
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
    cfg = ref.TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=2, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(13)
    a = torch.randn(64, 128, 128, device="cuda")
    b = torch.randn(64, 128, 128, device="cuda")
    with torch.no_grad():
        ya = cand(a)
        snapshot = ya.clone()
        cand(b)
    assert torch.equal(ya, snapshot), "the previous return value was mutated in place"


@pytest.mark.gpu
def test_the_fused_kernel_actually_runs_and_the_gemm_it_replaces_does_not():
    """L36: a test can pass because its subject was never built. Everything above would
    still be green if Dynamo had fallen back to eager and the candidate had quietly used
    v23's split path. Observe the mechanism: our fused kernel must appear in the CUDA
    profile, v23's attention kernel must NOT (it was replaced), and the graph must have
    captured -- otherwise this is not the path anything would be measured on."""
    torch._dynamo.reset()
    ref = _ref_module()
    cfg = ref.TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    _, cand = _pair(cfg)
    torch.manual_seed(14)
    x = torch.randn(64, 128, 128, device="cuda")
    with torch.inference_mode():
        for _ in range(3):
            cand(x)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                cand(x)

    names = {e.key for e in prof.key_averages()}
    assert cand.outproj_used, cand.outproj_reason
    assert any("attn_outproj" in n for n in names), \
        f"the fused kernel never launched; kernels seen: {sorted(names)[:20]}"
    assert not any("attn_single_tile" in n for n in names), \
        "v23's split attention kernel is still running -- the fusion did not replace it"
    assert not any("flash" in n.lower() for n in names), \
        "FlashAttention is still running"
    assert cand.graph_verified, "graph capture degraded; this is not the measured path"
