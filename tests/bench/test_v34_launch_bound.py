"""v34: the launch-bound fusion predicate, and the kernel count it is supposed to cut.

L36 governs the shape of this file: **assert the mechanism engaged before asserting
anything about speed.** A fused path that silently declined would pass every accuracy
test in this repository, because declining means running the parent's code.

The GPU tests are marked and skipped without CUDA; the predicate tests are pure and run
anywhere, which is the point -- a dispatch predicate that needs a GPU to be checked is a
hardcoded table wearing a costume (L28).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.matrix import MATRIX, BY_ID                                    # noqa: E402
from bench.kernels.ffn_fused import (MAX_TILE_ROWS, MIN_TILE_ROWS,        # noqa: E402
                                     amortizes, blocks_per_sm, fits,
                                     launch_tile, one_wave, smem_bytes)

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# This card, from ledger/device.json. Used only to evaluate the predicate in a pure test;
# the candidate itself reads these off get_device_properties at run time.
SM_COUNT = 66
SMEM_PER_SM = 102400
SMEM_OPTIN = 101376

RTOL, ATOL = 0.02, 0.002          # the harness's own defaults. Never widened.


# ======================================================================================
# The predicate, with no GPU
# ======================================================================================

def _launch_fused(cfg, sm=SM_COUNT, smem_sm=SMEM_PER_SM, smem_opt=SMEM_OPTIN):
    """The candidate's decision, as a pure function, in the candidate's own order."""
    bm = launch_tile(cfg.tokens, sm)
    if not fits(cfg.d_model, cfg.ffn_dim, 2, bm, smem_opt):
        return False
    if amortizes(cfg.tokens, cfg.d_model, cfg.ffn_dim, 2):
        return False              # the throughput regime; finding 25 governs there
    return one_wave(cfg.tokens, cfg.d_model, cfg.ffn_dim, 2, bm, sm, smem_sm)


def test_the_two_predicates_are_disjoint():
    """`amortizes` and `one_wave` must never both claim a shape.

    They are opposite arguments -- one says there is enough work to hide a weight load,
    the other says there is so little work that launches dominate -- and a shape that
    satisfies both would mean one of the two mechanisms is misstated.
    """
    for cfg in MATRIX:
        bm = launch_tile(cfg.tokens, SM_COUNT)
        if not fits(cfg.d_model, cfg.ffn_dim, 2, bm, SMEM_OPTIN):
            continue
        both = (amortizes(cfg.tokens, cfg.d_model, cfg.ffn_dim, 2)
                and one_wave(cfg.tokens, cfg.d_model, cfg.ffn_dim, 2, bm,
                             SM_COUNT, SMEM_PER_SM))
        if both:
            # Config 7's tiny weights satisfy both; the candidate resolves it by giving
            # `amortizes` precedence, so the LAUNCH path must not claim it.
            assert not _launch_fused(cfg), (
                f"config {cfg.id} claimed by both predicates AND by the launch path")


def test_selects_only_the_shapes_that_cannot_fill_the_device():
    selected = {c.id for c in MATRIX if _launch_fused(c)}
    assert selected == {2, 3, 4, 12}, selected
    # And every selected shape really is under one wave, by the numbers.
    for cid in selected:
        cfg = BY_ID[cid]
        bm = launch_tile(cfg.tokens, SM_COUNT)
        grid = -(-cfg.tokens // bm)
        cap = SM_COUNT * blocks_per_sm(cfg.d_model, cfg.ffn_dim, 2, bm, SMEM_PER_SM)
        assert grid <= cap, (cid, grid, cap)


def test_the_big_configs_are_declined():
    """Config 6 is 83% of wall time and at 97% of the HBM roofline (finding 29). The
    launch path must not touch it, nor 8, 13, 14 or the mainstream rows."""
    for cid in (1, 5, 6, 7, 8, 9, 10, 11, 13, 14):
        assert not _launch_fused(BY_ID[cid]), cid


def test_predicate_responds_to_the_device_not_to_the_matrix():
    """Halve the SM count and shapes fall out of one wave; quarter it and more do.

    A predicate that answers the same on every device is a lookup table.
    """
    cfg12 = BY_ID[12]
    assert _launch_fused(cfg12, sm=SM_COUNT)
    assert not _launch_fused(cfg12, sm=8)
    # A card with 48 KB of shared memory cannot hold both weight matrices at all.
    assert not _launch_fused(BY_ID[2], smem_opt=48 * 1024, smem_sm=48 * 1024)


def test_no_config_ids_or_announced_shapes_in_the_predicate_source():
    """CLAUDE.md rule 2, asserted on the source the way test_v14_dispatch does."""
    src = (REPO / "bench" / "kernels" / "ffn_fused.py").read_text()
    body = src[src.index("def one_wave"):src.index("def launch_tile") + 900]
    for forbidden in ("10000", "100000", "config", "cfg"):
        assert forbidden not in body.lower(), forbidden


def test_launch_tile_stays_legal_and_bounded():
    for tokens in (1, 16, 128, 2048, 8192, 1_280_000):
        bm = launch_tile(tokens, SM_COUNT)
        assert MIN_TILE_ROWS <= bm <= MAX_TILE_ROWS
        assert bm & (bm - 1) == 0, "tl.arange needs a power of two"
    # The tile widens with work and never exceeds what shared memory holds.
    assert launch_tile(128, SM_COUNT) == 16
    assert launch_tile(2048, SM_COUNT) == 32
    assert smem_bytes(128, 128, 2, launch_tile(8192, SM_COUNT)) <= SMEM_OPTIN


def test_one_wave_rejects_a_tile_that_does_not_fit():
    """blocks_per_sm == 0 must decline rather than divide by nothing."""
    assert blocks_per_sm(1024, 1024, 2, 64, SMEM_PER_SM) == 0
    assert not one_wave(64, 1024, 1024, 2, 64, SM_COUNT, SMEM_PER_SM)
    assert not one_wave(0, 128, 128, 2, 16, SM_COUNT, SMEM_PER_SM)
    assert not one_wave(128, 128, 128, 2, 8, SM_COUNT, SMEM_PER_SM)   # below MMA width


# ======================================================================================
# On the device: does the mechanism actually engage, and is it still correct?
# ======================================================================================

def _reference():
    path = REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_bench_t34", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench_t34"] = m
    spec.loader.exec_module(m)
    return m


def _build(config_id, candidate="v34_launch_bound", causal=None, padding=0.0):
    import torch
    from bench.candidates import REGISTRY
    torch._dynamo.reset()          # L36: a shared Dynamo cache silently falls back to eager
    ref = _reference()
    cfg = BY_ID[config_id]
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    tcfg = ref.TransformerConfig(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, d_model=cfg.d_model,
        num_heads=cfg.heads, ffn_dim=cfg.ffn_dim, num_layers=cfg.layers,
        causal=cfg.causal if causal is None else causal)
    tcfg.validate()
    torch.manual_seed(1234)
    base = ref.BaselineTransformer(tcfg)
    model = REGISTRY[candidate].build(ref.BaselineTransformer)(tcfg)
    ref.copy_model_weights(base, model)
    base = base.to(device=dev, dtype=torch.float32).eval()
    model = model.to(device=dev, dtype=torch.float32).eval()

    def make(seed):
        return ref.generate_random_case(tcfg, dev, torch.float32, seed=seed,
                                        padding_ratio=padding, input_scale=1.0)
    return ref, base, model, make


def _kernels_per_forward(model, x, m, n=10):
    import torch
    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        for _ in range(10):
            model(x, m)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                model(x, m)
            torch.cuda.synchronize()
    k = sum(1 for ev in prof.events() if str(ev.device_type) == "DeviceType.CUDA")
    return k / n


@cuda
@pytest.mark.parametrize("cid", [2, 12])
def test_mechanism_engages_and_the_kernel_count_actually_falls(cid):
    """L36: assert the mechanism ran. A silent decline would pass every accuracy test.

    The parent launches 36 nodes per forward on every config; this asserts the fused
    path is chosen AND that fewer nodes are actually replayed. Speed is not claimed here
    -- only the count, which is what the launch-floor argument rests on.
    """
    import torch
    _, _, model, make = _build(cid)
    x, m = make(1234)
    with torch.inference_mode():
        model(x, m)
    assert model.launch_fused_used, model.launch_reason
    assert model.launch_bm >= MIN_TILE_ROWS
    assert model.mask_capture.startswith("elided"), model.mask_capture

    _, _, parent, pmake = _build(cid, candidate="v26_causal_correct")
    px, pm = pmake(1234)
    n_parent = _kernels_per_forward(parent, px, pm)
    n_child = _kernels_per_forward(model, x, m)
    # ASSERT THE DROP, NOT THE ABSOLUTE COUNT. Alone the counts are 36.0 / 20.0; inside
    # the full suite both models undercount by a constant ~7 profiler events (35.3 /
    # 19.3), because CUDA module and context state accumulates across the 36 candidates
    # compiled in one process. The DIFFERENCE is 16.0 in both contexts, so that is the
    # invariant the launch-floor argument actually rests on. Pinning the absolute number
    # made this test pass alone and fail in the suite -- a test whose verdict depends on
    # what ran before it is not testing the candidate (L36).
    assert n_parent >= 30.0, f"parent count implausible: {n_parent}"
    assert n_parent - n_child == pytest.approx(16.0, abs=1.5), (n_parent, n_child)


@cuda
@pytest.mark.parametrize("cid", [2, 4, 12])
def test_accuracy_at_the_locked_tolerance(cid):
    import torch
    ref, base, model, make = _build(cid)
    with torch.inference_mode():
        for t in range(5):
            x, m = make(1234 + t)
            res = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
            assert res.passed, (cid, t, float(res.max_abs_error))
    assert model.launch_fused_used, model.launch_reason


@cuda
@pytest.mark.parametrize("cid", [2, 12])
def test_margin_is_no_worse_than_the_parents(cid, capsys):
    """L26: a candidate at 94% of the budget and one at 30% are not equally correct.

    The fused path carries strictly fewer rounding steps than the one it replaces -- the
    residual never round-trips HBM, and `h` stays in fp32 through GELU where the parent
    rounds it to fp16 first -- which is what finding 29 measured on config 6
    (1.87e-3 -> 1.56e-3). It does NOT follow that max_abs falls on every shape, and
    measured here it does not: config 12 improves (1.371e-3 -> 1.322e-3) and config 2
    worsens (9.50e-4 -> 1.053e-3), both with zero failed elements. Different rounding is
    not monotonically better rounding, and a single seed's max_abs is a noisy order
    statistic. So this asserts what L4 says the gate actually is -- **failed_elements** --
    plus a budget ceiling well inside the locked tolerance, and RECORDS the margin so a
    real drift is visible (L26). It deliberately does not assert an ordering that the
    numerics do not guarantee.
    """
    import torch
    ref, base, model, make = _build(cid)
    _, _, parent, pmake = _build(cid, candidate="v26_causal_correct")
    with torch.inference_mode():
        x, m = make(7)
        px, pm = pmake(7)
        ours = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
        theirs = ref.compare_outputs(base(px, pm), parent(px, pm), rtol=RTOL, atol=ATOL)
    with capsys.disabled():
        print(f"\n  cfg{cid} fused max_abs {float(ours.max_abs_error):.3e} "
              f"({100*float(ours.max_abs_error)/ATOL:.0f}% of the atol budget) "
              f"vs parent {float(theirs.max_abs_error):.3e} "
              f"({100*float(theirs.max_abs_error)/ATOL:.0f}%)")
    assert model.launch_fused_used, model.launch_reason
    assert ours.passed and int(ours.failed_elements) == 0, int(ours.failed_elements)
    # A ceiling, not a comparison. 75% leaves room for the seed-to-seed spread in a max
    # over 16k elements while still failing loudly if the fused path ever started eating
    # the budget the way v7 did (L4: seven configs pinned at 96.9% of it).
    assert float(ours.max_abs_error) <= 0.75 * ATOL, float(ours.max_abs_error)


@cuda
@pytest.mark.parametrize("cid", [8, 10])
def test_declines_take_the_parents_plan_and_cost_no_accuracy(cid, capsys):
    """The configs this candidate must not touch must run the parent's code, choose the
    parent's plan, and be no less accurate than the parent against the fp32 reference.

    NOT byte-identity, deliberately. Run in isolation these two are bit-for-bit equal
    (checked, 3/3 trials on both configs), but the equality is not a property the system
    guarantees: `attn_single_tile.autotune_tile` picks its tile by TIMING at prime time,
    and Dynamo's per-process cache limit can drop either model to eager without saying so
    (L36). Asserting equality here produces a test that fails for reasons unrelated to the
    change, and a test that fails for the wrong reason eventually gets weakened. So assert
    the two things that are actually guaranteed -- the plan and the accuracy -- and record
    the deviation so a real drift would still be visible.
    """
    import torch
    ref, base, model, make = _build(cid)
    _, _, parent, pmake = _build(cid, candidate="v26_causal_correct")
    x, m = make(99)
    px, pm = pmake(99)
    with torch.inference_mode():
        got = model(x, m)
        want = parent(px, pm)
        expect = base(x, m)
    assert not model.launch_fused_used, model.launch_reason
    assert model.fused_ffn_reason == parent.fused_ffn_reason
    # Only the DISPATCH half of attn_reason. It is `f"{why}; {how}"`, where `why` comes
    # from `applies()` (pure, shapes and device properties) and `how` from
    # `autotune_tile` (chosen by TIMING ~1 us kernels against a ~1 us event timer, and
    # measured to flip run to run at these shapes -- see docs/findings/33). Asserting the
    # whole string makes this test fail for a reason that has nothing to do with v34,
    # which is how a test ends up weakened later for the wrong reason.
    assert model.attn_reason.split(";")[0] == parent.attn_reason.split(";")[0], (
        model.attn_reason, parent.attn_reason)

    ours = ref.compare_outputs(expect, got, rtol=RTOL, atol=ATOL)
    theirs = ref.compare_outputs(expect, want, rtol=RTOL, atol=ATOL)
    with capsys.disabled():
        print(f"\n  cfg{cid} declined: v34 max_abs {float(ours.max_abs_error):.3e} vs "
              f"parent {float(theirs.max_abs_error):.3e}; "
              f"v34-vs-parent {(got - want).abs().max().item():.3e}")
    assert ours.passed and theirs.passed
    # Declining must not cost accuracy. Equal in every run observed; the slack is for
    # the JIT/autotune nondeterminism named above, not for a real regression.
    assert float(ours.max_abs_error) <= max(float(theirs.max_abs_error) * 1.05, ATOL / 2)


@cuda
def test_the_fallback_runs_when_the_input_is_masked():
    """A padded input has no proof the kernel can rely on, so the fused path must
    decline and the parent's masked path must still be right. Testing the fallback is
    the point: a dispatch whose fallback is never exercised is untested code."""
    import torch
    ref, base, model, make = _build(2, padding=0.5)
    with torch.inference_mode():
        for t in range(3):
            x, m = make(500 + t)
            res = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
            assert res.passed, (t, float(res.max_abs_error))
    assert not model.launch_fused_used
    assert "masked" in model.launch_reason
    assert model.mask_capture == "copied (masked input)"


@cuda
def test_non_causal_still_delegates_to_the_unmodified_baseline():
    """Finding 32 / L42: the harness's own default is causal=False, and v26's guarantee
    must survive being inherited from."""
    import torch
    ref, base, model, make = _build(2, causal=False)
    with torch.inference_mode():
        x, m = make(11)
        res = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
    assert res.passed, float(res.max_abs_error)
    assert model.causal_path.startswith("baseline")


@cuda
def test_different_inputs_do_not_give_the_same_answer():
    """L23/L25, the crudest and most valuable invariant: this candidate holds static
    CUDA-graph buffers across calls, and a stale one returns the right shape and the
    wrong values while satisfying every reference comparison for the PREVIOUS input."""
    import torch
    _, _, model, make = _build(2)
    with torch.inference_mode():
        for t in range(4):
            model(*make(1234 + t))
        a = model(*make(1)).clone()
        b = model(*make(2)).clone()
        c = model(*make(1)).clone()
    assert not torch.equal(a, b), "output does not depend on the input"
    assert torch.equal(a, c), "the same input gave two different answers"


@cuda
def test_returned_tensor_survives_the_next_call():
    """L25: a static output buffer the next call overwrites is a silent wrong answer
    handed to whoever kept a reference."""
    import torch
    _, _, model, make = _build(2)
    with torch.inference_mode():
        for t in range(4):
            model(*make(1234 + t))
        held = model(*make(1))
        snapshot = held.clone()
        model(*make(2))
    assert torch.equal(held, snapshot), "the returned tensor was mutated by a later call"


@cuda
def test_graph_capture_still_verified():
    """v13's fail-safe capture and v18's insurance must survive the mask elision."""
    import torch
    _, _, model, make = _build(2)
    with torch.inference_mode():
        for t in range(3):
            model(*make(1234 + t))
    assert model.graph_verified, "capture degraded to the compiled callable"
    assert model.capture_source in ("caller", "insurance"), model.capture_source
    assert model._static_m is None, "the dead mask copy is still being captured"
