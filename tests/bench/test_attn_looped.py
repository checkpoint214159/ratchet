"""The looped attention kernel and the symmetric chooser that may select it.

Tolerance is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR, judged by failed
elements and never by max_abs alone (L4). It is not widened anywhere in this file.

The reference is the benchmark's OWN formulation -- fp32 scores, fp32 softmax, fp32 PV,
then the head-major repack -- not SDPA, which carries its own error and would let a
kernel inherit a mistake it shares with SDPA.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.kernels import attn_choice, attn_looped
from bench.kernels.attn_looped import (SWEEP_TILES, applies, fits, grid_ctas,
                                       looped_attention, pays, register_bytes,
                                       smem_bytes, viable_tiles)
from bench.kernels.attn_single_tile import DECISIVE, single_tile_attention

ATOL, RTOL = 2e-3, 2e-2


def _fp32_attention(qkv, heads, head_dim):
    """The reference's formulation, in fp32 throughout."""
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


def _qkv(b, s, heads, hd, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dm = heads * hd
    return torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16,
                       generator=g) * 0.3


# ------------------------------------------------------------------ the kernel

@pytest.mark.parametrize("b,s,heads,hd", [
    (8, 128, 2, 64),        # config 10's shape, the target
    (8, 128, 1, 128),       # config 9: head_dim 128, where single_tile declines
    (8, 128, 4, 32),        # config 1: head_dim 32
    (8, 128, 4, 8),         # configs 7/11: head_dim BELOW the MMA width, padded to 16
    (4, 1024, 4, 32),       # config 13: the long-sequence row
    (4, 32, 4, 32),         # config 12: S below block_m
])
def test_matches_the_fp32_reference(b, s, heads, hd):
    qkv = _qkv(b, s, heads, hd)
    want = _fp32_attention(qkv, heads, hd)
    got = looped_attention(qkv, heads, hd, hd ** -0.5, min(128, s), 16, 4, 3)
    ok, mx = _within(got, want)
    assert ok.all(), (f"{(~ok).sum().item()} of {ok.numel()} elements past the locked "
                      f"tolerance, max_abs {mx:.3e}")


def test_layout_is_byte_identical_to_the_single_tile_form():
    """Both kernels must write head-major `[B, S, heads*head_dim]`, because both are
    drop-ins for the same call site and the out-projection reads that layout directly.
    A layout difference would be a silent wrong answer at the NEXT op, not this one."""
    b, s, heads, hd = 8, 128, 2, 64
    qkv = _qkv(b, s, heads, hd)
    a = single_tile_attention(qkv, heads, hd, hd ** -0.5, 32, 8, 1)
    c = looped_attention(qkv, heads, hd, hd ** -0.5, 128, 16, 8, 4)
    ok, mx = _within(c, a)
    assert ok.all(), f"the two forms disagree: max_abs {mx:.3e}"


def test_non_multiple_sequence_length_masks_rather_than_producing_nan():
    """Rows with `rm >= S` exist whenever S is not a multiple of block_m. They keep every
    key column valid, so no row is entirely -inf -- if that reasoning is wrong the
    symptom is NaN in the discarded lanes, which can still poison a later reduction."""
    qkv = _qkv(4, 96, 2, 64)
    got = looped_attention(qkv, 2, 64, 64 ** -0.5, 64, 16, 4, 3)
    assert torch.isfinite(got.float()).all()
    ok, mx = _within(got, _fp32_attention(qkv, 2, 64))
    assert ok.all(), f"max_abs {mx:.3e}"


def test_the_chosen_tile_spills_nothing():
    """A spilling tile is slower than the vendor and the whole premise is occupancy."""
    qkv = _qkv(8, 128, 2, 64)
    _, h = looped_attention(qkv, 2, 64, 64 ** -0.5, 128, 16, 8, 4, _return_handle=True)
    assert h.n_spills == 0, f"n_spills={h.n_spills}, n_regs={h.n_regs}"


# --------------------------------------------------------------- the predicate

def test_every_swept_tile_gives_the_loop_more_than_one_trip():
    """`block_n < block_m` is the mechanism constraint, and finding 47 is why it is a
    test: F-03 was priced at +0.0138 on arms whose grid-stride loop ran exactly once."""
    for bm, bn, w, st in SWEEP_TILES:
        assert bn < bm, f"tile ({bm},{bn},{w},{st}) runs one trip on the first block"
        assert st >= 2, f"tile ({bm},{bn},{w},{st}) disables the pipeliner"


def test_predicate_declines_a_single_wave():
    """The hypothesis is that pipelining needs more than one wave to hide behind. A
    batch of 1 gives 4 CTAs on 66 SMs and must be declined."""
    props = torch.cuda.get_device_properties("cuda")
    assert not pays(1, 4, 128, 64, props.multi_processor_count)
    ok, why = applies(1, 4, 128, 32, props)
    assert not ok, why


def test_predicate_reads_only_shapes_and_measured_device_properties():
    """CLAUDE.md rule 2. `grid_ctas` and `pays` take shapes and an SM count; a card with
    a different SM count evaluates them differently without being retuned."""
    assert grid_ctas(64, 2, 128, 128) == 128
    assert pays(64, 2, 128, 128, 66) is True
    assert pays(64, 2, 128, 128, 512) is False       # a much wider card declines


def test_smem_ceiling_is_the_measured_optin_limit():
    """A 128-wide K/V tile at head_dim 128 over 4 stages wants 256 KB of shared memory
    against this card's 99 KB opt-in ceiling. The register budget alone does NOT catch
    it (163 KB against a 255 KB budget at 8 warps), so this is the smem check firing."""
    props = torch.cuda.get_device_properties("cuda")
    from bench.kernels.attn_single_tile import register_budget
    assert register_bytes(128, 128, 128) <= register_budget(
        8, props.regs_per_multiprocessor, props.warp_size)
    assert smem_bytes(128, 128, 4) > props.shared_memory_per_block_optin
    assert not fits(128, 128, 128, 128, 8, 4, props.regs_per_multiprocessor,
                    props.shared_memory_per_block_optin, props.warp_size)


# ----------------------------------------------------------------- the chooser

def test_sweep_admits_only_arms_that_match_and_do_not_spill():
    """Correctness before timing, per arm: a fast wrong kernel must not win a sweep."""
    rows = []
    attn_choice.autotune_looped(128, 64, 2, 64, "cuda", reps=1, collect=rows)
    assert rows, "the sweep timed nothing"
    assert all(r[4] == 0 for r in rows), "a spilling arm reached the timing set"
    forms = {r[0] for r in rows}
    assert "sdpa" in forms, "sdpa was not swept as an arm"
    assert "single_tile" in forms and "looped" in forms, (
        f"both Triton forms must be swept symmetrically, got {forms}")


def test_challenger_must_clear_the_decisive_margin():
    """The incumbent holds the ground. Reconstructing the decision from the collected
    rows must agree with what `autotune` returned -- if it does not, the margin is being
    applied somewhere other than where this test can see it."""
    rows = []
    tile, why = attn_choice.autotune_looped(128, 64, 2, 64, "cuda", reps=2, collect=rows)
    best = {}
    for f, t, ms, *_ in rows:
        best[(f, t)] = min(best.get((f, t), float("inf")), ms)
    from bench.kernels.attn_single_tile import choose_tile
    props = torch.cuda.get_device_properties("cuda")
    derived = choose_tile(128, 64, props.regs_per_multiprocessor,
                          props.max_threads_per_multi_processor, props.warp_size)
    # The margin is applied against the INCUMBENT -- what the model runs today -- and
    # sdpa is a separate hard floor rather than a second 10% margin.
    incumbent = best[("single_tile", derived)]
    assert best[("looped", tile)] < incumbent * (1.0 - DECISIVE), (
        f"looped{tile} was selected at {best[('looped',tile)]:.5f} ms against an "
        f"incumbent {incumbent:.5f} ms -- inside the {DECISIVE:.0%} margin. {why}")
    assert best[("looped", tile)] <= best[("sdpa", ())], (
        f"looped{tile} was selected while sdpa is faster. {why}")


def test_the_tuner_times_in_the_call_sites_cache_regime():
    """[L53]. The kernel runs L2-hot inside a replayed graph, so the tuner must not rank
    arms with a timer that flushes L2 and pays a launch. The two regimes disagree about
    config 1 by enough to cross the DECISIVE margin, so this is not cosmetic.

    Asserted by behaviour rather than by reading the source: the hot timer must actually
    work on this device, because the fallback to `do_bench` is silent by design and a
    silent fallback that always fires is [L36]."""
    import triton.testing as tt
    qkv = _qkv(8, 128, 2, 64)
    ms = tt.do_bench_cudagraph(
        lambda: looped_attention(qkv, 2, 64, 64 ** -0.5, 128, 16, 8, 4),
        rep=25, return_mode="min")
    assert ms > 0, "do_bench_cudagraph did not produce a time on this device"
    # And the sweep must still complete afterwards -- i.e. capture left the context sane.
    tile, why = attn_choice.autotune_looped(128, 64, 2, 64, "cuda", reps=1)
    assert len(tile) == 4, why


def test_the_probe_allocation_is_bounded():
    """Config 14's shape would allocate 9.8 GiB to tune a kernel. The sweep must decline
    it rather than OOM the model it is tuning."""
    with pytest.raises(ValueError, match="budget"):
        attn_choice.autotune_looped(100000, 64, 16, 32, "cuda", reps=1)


def test_a_single_wave_shape_keeps_the_incumbent_and_never_the_looped_form():
    """Batch 1 gives 4 CTAs on 66 SMs. The looped form must not even be swept there, and
    the shape must still get the single-tile kernel it had before this file existed --
    adding a second form may not cost a shape the first one."""
    with pytest.raises(ValueError, match="no looped tile|no viable tile"):
        attn_choice.autotune_looped(128, 32, 4, 1, "cuda", reps=1)


def test_the_chooser_raises_rather_than_returning_something_it_did_not_earn():
    """The caller's fallback is SDPA and it is reached through an exception. A chooser
    that returned a tile it had not timed, or that quietly returned `sdpa` as if it were
    a tile, would be the [L36] defect: a mechanism reporting success while doing nothing.

    S=1024 at batch 1 exercises the harder of the two decline paths. The single-tile form
    declines outright (a 128x1024 fp32 score tile is 512 KB), so there is no derived tile
    to fall back to -- and the looped form IS viable here, because `pays` counts CTAs and
    a narrow `block_m` at S=1024 gives 1*4*64 = 256 of them. So the chooser has arms, it
    times them, none clears the sdpa bar, and it must then decline rather than ship the
    fastest thing that happened to compile."""
    with pytest.raises(ValueError, match="did not clear|slower than sdpa"):
        attn_choice.autotune_looped(1024, 32, 4, 1, "cuda", reps=1)


def test_the_other_decline_path_when_no_form_has_a_tile_at_all():
    """head_dim 256 at batch 1: the single-tile score tile overflows the register file
    and the looped form's grid is 1*4*1 = 4 CTAs at any legal block_m, under one wave."""
    with pytest.raises(ValueError, match="no viable tile"):
        attn_choice.autotune_looped(128, 256, 4, 1, "cuda", reps=1)
