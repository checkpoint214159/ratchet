"""v22: a hand-written attention kernel where head_dim is below the tl.dot K floor.

Everything here is checked at the LOCKED tolerance (atol 2e-3, rtol 2e-2, OR criterion),
which is the custody benchmark's own CLI default and is never widened.

Three things this file deliberately does beyond "the numbers match":

  * asserts the KERNEL ACTUALLY RAN (L36 -- a test can pass because its subject was never
    built; `smallhead_attn_used` is a flag, a call counter is evidence),
  * asserts the FALLBACK path is exercised and correct, not just the fast one,
  * tracks tolerance MARGIN against the parent rather than pass/fail alone (L26 -- our
    worst config already uses 94% of the budget, so "passes" is not the whole answer).
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels import attn_smallhead
from bench.kernels.attn_smallhead import (DEFAULT_TILE, TILING, applicable, attend,
                                          clamp_tile, min_dot_k, padded_head_dim,
                                          plan_for, smallhead_attention)
from bench.matrix import MATRIX

ATOL, RTOL = 2e-3, 2e-2
CANDIDATE = "v22_headdim8_attn"
PARENT = "v18_capture_insurance"


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    """L36: Dynamo's cache_size_limit is 8 and shared per process. Once exhausted,
    torch.compile silently falls back to EAGER and a test can go green because its
    subject was never compiled."""
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _ref():
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v22", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v22"] = m
    spec.loader.exec_module(m)
    return m


def _within_locked_tolerance(got, want):
    d = (got.float() - want.float()).abs()
    return bool((((d <= ATOL) | (d <= RTOL * want.float().abs())).all()).item())


def _failed_elements(got, want):
    d = (got.float() - want.float()).abs()
    return int((~((d <= ATOL) | (d <= RTOL * want.float().abs()))).sum().item())


def _build(name, cfg, ref):
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    cand = REGISTRY[name].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    return base, cand


# ======================================================================================
# The dispatch predicate -- rule 2, and it must survive shapes nobody here has seen
# ======================================================================================

def test_registered_with_the_declared_parent():
    spec = REGISTRY[CANDIDATE]
    assert spec.parent == PARENT
    assert spec.generation == 22


def test_the_mma_floor_is_queried_not_asserted():
    """The whole mechanism rests on this number. It comes out of Triton's own NVIDIA
    backend for THIS device, so it tracks the compiler instead of a comment."""
    assert min_dot_k(16) == 16, "sm_89's mma.sync.m16n8k16 gives a K floor of 16 at fp16"
    assert min_dot_k(8) == 32, "the 8-bit path has a different floor; we are querying it"
    assert padded_head_dim(8) == 16 and padded_head_dim(32) == 32


def test_predicate_selects_exactly_the_configs_below_the_floor():
    chosen = {c.id for c in MATRIX
              if applicable(c.head_dim, c.seq_len, c.d_model, c.heads)}
    below = {c.id for c in MATRIX if c.head_dim < min_dot_k(16)}
    assert chosen == below, f"predicate selects {chosen}, below-floor configs are {below}"
    # Which, on this matrix and this device, is the head_dim=8 pair -- derived, not typed.
    assert all(MATRIX[i - 1].head_dim == 8 for i in chosen)


def test_predicate_contains_no_benchmark_knowledge():
    """Rule 2. Shapes nobody in this repo has measured must get sensible answers."""
    assert applicable(4, 4096, 256, 64), "an unseen shape below the floor should fire"
    assert applicable(2, 17, 64, 32), "sequence length is irrelevant to the mechanism"
    for hd in (16, 32, 64, 128, 256):
        assert not applicable(hd, 128, hd * 4, 4), f"head_dim {hd} needs no in-kernel pad"
    assert not applicable(12, 128, 48, 4), "non-power-of-two head_dim is declined"
    assert not applicable(8, 128, 100, 4), "d_model must equal head_dim * heads"


def test_predicate_is_monotone_in_head_dim():
    """More head lanes can only make the pad LESS worth having. A predicate that switches
    back on above the floor is fitted to configs, not derived from the instruction."""
    seen_false = False
    for hd in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        now = applicable(hd, 128, hd * 4, 4)
        assert not (seen_false and now), "applicability must not switch back on"
        seen_false = seen_false or not now


def _executable_source(module) -> str:
    """The module with every docstring and comment removed. Prose may discuss configs by
    number -- it has to, to record what was measured. Executable code may not."""
    src = Path(module.__file__).read_text()
    src = re.sub(r'("""|\'\'\')(?:.|\n)*?\1', "", src)
    return "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))


def test_source_names_no_config_id_and_no_announced_shape():
    """L28's discipline: a dispatch that hardcodes the benchmark is a table in a costume."""
    from bench.candidates import v22_headdim8_attn
    body = _executable_source(attn_smallhead) + _executable_source(v22_headdim8_attn)
    for token in ("config 7", "config 11", "cfg7", "cfg11", "config_id", "config.id"):
        assert token not in body, f"{token!r} appears in executable code"
    # The two announced shapes this candidate targets, as literals. head_dim 8 must never
    # be written down -- the floor is queried and the comparison is strictly-less-than.
    for literal in ("== 8", "head_dim == ", "d_model == 32", "num_heads == 16"):
        assert literal not in body, f"{literal!r} pins an announced shape"


def test_tile_is_clamped_to_short_sequences():
    """A tile wider than the sequence wastes lanes and can fall under the mma M/N floor."""
    assert clamp_tile((128, 128, 8, 2), 32) == (32, 32, 8, 2)
    assert clamp_tile((128, 128, 8, 2), 8)[0] >= 16, "never below the mma minimum"
    assert clamp_tile(DEFAULT_TILE, 1024) == DEFAULT_TILE
    assert DEFAULT_TILE in TILING, "the shipped tile must be one the sweep actually saw"


# ======================================================================================
# The kernel itself, against an fp32 reference
# ======================================================================================

@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(64, 4, 128, 8), (64, 16, 128, 8), (2, 3, 37, 8),
                                   (1, 1, 16, 8), (3, 5, 200, 4)])
def test_kernel_matches_fp32_attention(shape):
    """Strided q/k/v exactly as the lineage produces them: views of one [Z,S,3D] GEMM
    output. Compared against attention computed in fp32 on the same fp16 inputs."""
    z, h, s, d = shape
    dm = h * d
    torch.manual_seed(0)
    qkv = torch.randn(z, s, 3 * dm, device="cuda", dtype=torch.float16)
    a, b, c = qkv.split(dm, dim=-1)
    q = a.view(z, s, h, d).transpose(1, 2)
    k = b.view(z, s, h, d).transpose(1, 2)
    v = c.view(z, s, h, d).transpose(1, 2)
    want = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True).transpose(1, 2).reshape(z, s, dm)
    got = attend(q, k, v)
    assert got.shape == want.shape, "the kernel returns token-major [Z, S, H*hd]"
    assert _within_locked_tolerance(got, want), (
        f"{_failed_elements(got, want)} failed elements")


@pytest.mark.gpu
def test_every_swept_tile_is_correct_not_just_the_shipped_one():
    """The tile is a performance knob. If any of them is wrong, the sweep that chose one
    was choosing between a correct and an incorrect kernel."""
    z, h, s, d = 4, 4, 128, 8
    dm = h * d
    torch.manual_seed(1)
    qkv = torch.randn(z, s, 3 * dm, device="cuda", dtype=torch.float16)
    a, b, c = qkv.split(dm, dim=-1)
    q = a.view(z, s, h, d).transpose(1, 2)
    k = b.view(z, s, h, d).transpose(1, 2)
    v = c.view(z, s, h, d).transpose(1, 2)
    want = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=True).transpose(1, 2).reshape(z, s, dm)
    for tile in TILING:
        got = attend(q, k, v, tile)
        assert _within_locked_tolerance(got, want), f"tile {tile} is wrong"


@pytest.mark.gpu
def test_the_causal_triangle_is_skipped_exactly():
    """Row 0 may see only key 0. If the loop bound were wrong this is where it shows."""
    z, h, s, d = 1, 1, 64, 8
    torch.manual_seed(2)
    q = torch.randn(z, h, s, d, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    got = attend(q, k, v)
    assert torch.allclose(got[0, 0].float(), v[0, 0, 0].float(), atol=2e-3), (
        "the first query attends to exactly one key, so its output IS that value vector")


# ======================================================================================
# The candidate end to end
# ======================================================================================

def _cfg(ref, **kw):
    base = dict(batch_size=8, seq_len=128, d_model=128, num_heads=16, ffn_dim=128,
                num_layers=4, causal=True)
    base.update(kw)
    return ref.TransformerConfig(**base)


def _cuda_kernels(model, x, mask=None) -> set:
    """The names of the CUDA kernels one call actually launches.

    This is the L36 assertion, and it is done with the profiler rather than by wrapping
    the call site ON PURPOSE: a monkeypatched wrapper is a new closure Dynamo has to guard
    on, which forces a recompile every call and blows the cache limit. Instrumenting a
    compiled region changes it. Ask the device what ran instead.
    """
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        for _ in range(3):
            model(x, mask) if mask is not None else model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(x, mask) if mask is not None else model(x)
            torch.cuda.synchronize()
    return {e.key for e in prof.key_averages() if e.device_time_total > 0}


@pytest.mark.gpu
def test_matches_the_baseline_where_the_kernel_fires():
    """Config 11's shape (head_dim 8), at a smaller batch so the test stays cheap."""
    ref = _ref()
    cfg = _cfg(ref, batch_size=8, d_model=128, num_heads=16)
    base, cand = _build(CANDIDATE, cfg, ref)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.smallhead_attn_used is True, cand.smallhead_attn_reason
    assert _within_locked_tolerance(got, want), (
        f"{_failed_elements(got, want)} failed elements, "
        f"max_abs {(got.float()-want.float()).abs().max():.3e}")

    names = _cuda_kernels(cand, x)
    assert any("_attn_fwd_smallhead" in n for n in names), (
        f"the flag says the kernel fired but no such kernel ran: {sorted(names)}")
    assert not any("flash_fwd_kernel" in n for n in names), (
        "the vendor kernel is still running -- SDPA was not actually replaced")


@pytest.mark.gpu
def test_inductor_still_fuses_around_the_hand_written_kernel():
    """THE BUG THAT COST THIS CANDIDATE ITS FIRST SCREEN, pinned so it cannot come back.

    Resolving the launch plan inside `_core` put an import, a try/except and a locally
    defined class in Dynamo's traced region. Dynamo dropped the whole frame to eager, and
    Inductor's fused LayerNorm kernels were replaced by 9 eager
    `vectorized_layer_norm_kernel` calls at 151 us -- a 1.63x win on attention became a
    2.18x LOSS on config 7, with every correctness test still green and
    `graph_verified` still True.

    So: assert the COMPILER still ran, not just that the answer is right. A `triton_*_fused
    _*` LayerNorm kernel is Inductor's signature; ATen's `vectorized_layer_norm_kernel` is
    the signature of the failure.
    """
    ref = _ref()
    cfg = _cfg(ref, batch_size=64, d_model=32, num_heads=4, ffn_dim=32)
    _, cand = _build(CANDIDATE, cfg, ref)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    names = _cuda_kernels(cand, x)
    assert any(n.startswith("triton_") and "layer_norm" in n for n in names), (
        f"Inductor is no longer fusing the LayerNorms -- the frame fell back to eager. "
        f"kernels: {sorted(names)}")
    assert not any("vectorized_layer_norm_kernel" in n for n in names), (
        f"eager ATen LayerNorm is running, i.e. the compiled region was lost. "
        f"kernels: {sorted(names)}")


@pytest.mark.gpu
def test_matches_the_baseline_on_the_narrow_model_shape():
    """Config 7's shape: d_model 32, head_dim 8 -- also the shape where v17's FFN
    megakernel fires, so this exercises both hand-written kernels in one forward."""
    ref = _ref()
    cfg = _cfg(ref, batch_size=64, d_model=32, num_heads=4, ffn_dim=32)
    base, cand = _build(CANDIDATE, cfg, ref)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.smallhead_attn_used is True, cand.smallhead_attn_reason
    assert _within_locked_tolerance(got, want), (
        f"{_failed_elements(got, want)} failed elements")


@pytest.mark.gpu
def test_the_fallback_path_is_taken_and_is_correct():
    """head_dim 32: the vendor kernel's tiles already fit the hardware, so we decline and
    v18's path must run UNTOUCHED. A dispatch is only trustworthy if its off-branch is."""
    ref = _ref()
    cfg = _cfg(ref, batch_size=8, d_model=128, num_heads=4)
    base, cand = _build(CANDIDATE, cfg, ref)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert cand.smallhead_attn_used is False and "declined" in cand.smallhead_attn_reason
    assert cand.attn_plan is None
    assert _within_locked_tolerance(got, want)
    names = _cuda_kernels(cand, x)
    assert not any("_attn_fwd_smallhead" in n for n in names), (
        "the kernel ran on a shape the predicate declined")
    assert any("flash_fwd_kernel" in n for n in names), (
        f"the declined path should be the vendor's: {sorted(names)}")


@pytest.mark.gpu
def test_non_causal_is_declined_and_the_inherited_defect_is_pinned():
    """Two separate assertions, and the second one is a BUG REPORT, not a pass.

    Ours: the kernel skips the causal triangle by construction, so nothing may route a
    non-causal shape into it. That holds -- `smallhead_attn_used` is False.

    Inherited: **v8 through v18 pass `is_causal=True` to SDPA unconditionally**, ignoring
    `config.causal` entirely, so the whole lineage silently returns causal attention for a
    non-causal model. Measured on this shape, identically for v8 / v13 / v18 / v22:
    max_abs 9.87e-01, 39345 failed elements against a 2e-3 budget.

    Every announced row is causal (`matrix.py`), so no ledger number is affected -- but
    the reference benchmark's own default is `causal=False`, which is L24 exactly: correct
    only because of how our harness happens to call it. This test pins the defect to v22's
    parent so that a future fix is visibly a fix and this candidate is visibly not its
    cause. It deliberately does NOT fix it: that is a separate one-variable change.
    """
    ref = _ref()
    cfg = _cfg(ref, batch_size=4, num_layers=1, causal=False)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()

    failed = {}
    for name in (PARENT, CANDIDATE):
        torch._dynamo.reset()
        torch.manual_seed(0)
        cand = REGISTRY[name].build(ref.BaselineTransformer)(cfg).cuda().eval()
        ref.copy_model_weights(base, cand)
        with torch.no_grad():
            want, got = base(x), cand(x)
        failed[name] = _failed_elements(got, want)
        if name == CANDIDATE:
            assert cand.smallhead_attn_used is False, (
                "the kernel must never see a non-causal shape")

    assert failed[PARENT] > 0, (
        "the inherited non-causal defect has been fixed upstream -- delete this pin and "
        "assert correctness instead")
    assert failed[CANDIDATE] == failed[PARENT], (
        f"v22 changed the non-causal path: {failed[CANDIDATE]} failed elements against "
        f"the parent's {failed[PARENT]}")


@pytest.mark.gpu
def test_matches_the_baseline_on_the_right_padded_path():
    """v8's redundant-mask proof plus the zeroing branch, which the unmasked tests never
    reach. finding 11: every measurement before v8 only ever ran padding_ratio 0."""
    ref = _ref()
    cfg = _cfg(ref, batch_size=8, num_layers=2)
    base, cand = _build(CANDIDATE, cfg, ref)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    lengths = torch.randint(1, cfg.seq_len + 1, (cfg.batch_size,), device="cuda")
    mask = torch.arange(cfg.seq_len, device="cuda")[None, :] < lengths[:, None]
    with torch.no_grad():
        want, got = base(x, mask), cand(x, mask)
    assert cand.smallhead_attn_used is True, cand.smallhead_attn_reason
    assert _within_locked_tolerance(got, want), (
        f"{_failed_elements(got, want)} failed elements")


@pytest.mark.gpu
def test_output_depends_on_the_input_and_survives_the_next_call():
    """L23/L25, the two invariants that caught three bugs the whole accuracy suite could
    not see: a stale static buffer, and a returned tensor the next call overwrites."""
    ref = _ref()
    cfg = _cfg(ref, batch_size=4, num_layers=2)
    _, cand = _build(CANDIDATE, cfg, ref)
    x1 = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    x2 = torch.randn_like(x1)
    with torch.no_grad():
        y1 = cand(x1)
        kept = y1.clone()
        y2 = cand(x2)
    assert not torch.equal(y1, y2), "different inputs produced identical output"
    assert torch.equal(y1, kept), "the returned tensor was mutated by the next call"


@pytest.mark.gpu
def test_does_not_spend_more_tolerance_budget_than_its_parent():
    """L26: margin is a first-class metric. A candidate at 94% of budget and one at 30%
    are not equally correct, and only the second survives a distribution shift. Checked at
    a reduced input scale, where finding 19 showed the margin collapses."""
    ref = _ref()
    cfg = _cfg(ref, batch_size=8, num_layers=4)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda") * 0.05
    margins = {}
    for name in (PARENT, CANDIDATE):
        torch.manual_seed(0)
        cand = REGISTRY[name].build(ref.BaselineTransformer)(cfg).cuda().eval()
        ref.copy_model_weights(base, cand)
        with torch.no_grad():
            want, got = base(x), cand(x)
        margins[name] = (got.float() - want.float()).abs().max().item()
        torch._dynamo.reset()
    # It DOES spend a little more: a different online-softmax reordering rounds
    # differently. Measured at the default scale on the two target shapes, v22 uses
    # 76% of the 2e-3 budget where v18 uses 66% (1.52e-3 / 1.23e-3 on config 7's shape,
    # 1.54e-3 / 1.32e-3 on config 11's). That is a real cost and it is recorded rather
    # than hidden. The bound below is a REGRESSION MONITOR, not the correctness gate --
    # the gate is the locked tolerance, asserted in the tests above and never widened.
    assert margins[CANDIDATE] <= margins[PARENT] * 1.5, (
        f"the new kernel spends materially more of the tolerance budget than v18: "
        f"{margins[CANDIDATE]:.3e} vs {margins[PARENT]:.3e}")


@pytest.mark.gpu
def test_the_kernel_does_not_break_graph_capture():
    """L36 one level up: the frontier's speed comes from a CUDA graph over a compiled
    core, and a user-defined Triton kernel inside that region is exactly the kind of thing
    that makes Dynamo re-trace and the capture fail. v13 then degrades to slower-and-
    correct SILENTLY, which would make this candidate look like a loss for a reason that
    has nothing to do with attention. Assert the mechanism still runs."""
    ref = _ref()
    cfg = _cfg(ref, batch_size=8, num_layers=4)
    _, cand = _build(CANDIDATE, cfg, ref)
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    with torch.no_grad():
        for _ in range(3):
            cand(x)
    assert cand.smallhead_attn_used is True
    assert cand.graph_verified is True, (
        f"graph capture failed with the kernel in the region "
        f"(capture_source={cand.capture_source})")
