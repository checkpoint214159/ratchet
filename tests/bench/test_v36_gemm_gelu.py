"""v36: the hand-written projection GEMM, its GELU epilogue, and the predicate that
declines it.

L36 governs the shape of this file: **assert the mechanism engaged before asserting
anything about speed.** This candidate had that exact bug during construction and it was
invisible -- `_core` bailed to the parent whenever v23's attention kernel declined, so
config 9 (`heads=1, head_dim=128`, where v23 declines) ran four cuBLAS calls while
`gemm_reason` reported "triton on out+ffn_in+ffn_out". Every accuracy test passed. What
caught it was counting kernels, so that is what this file does.

The pure tests need no GPU, which is the point: a dispatch predicate that can only be
checked on the device it was tuned on is a hardcoded table wearing a costume (L28). The
one thing that genuinely cannot be pure here is the decision itself -- it is a
MEASUREMENT of the vendor against the sweep, by design (CLAUDE.md rule 2), so what the
pure tests pin is the tile legality, the register budget and the source's freedom from
config ids; and the GPU tests pin that the decision is capable of saying no.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.matrix import BY_ID                                            # noqa: E402
from bench.kernels.proj_gemm import (DECISIVE, MAX_ACC_REGS_PER_THREAD,   # noqa: E402
                                     MMA_MIN, SWEEP_TILES, legal,
                                     probe_rows, viable_tiles)

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

RTOL, ATOL = 0.02, 0.002          # the harness's own defaults. Never widened.


# ======================================================================================
# The tile space, with no GPU
# ======================================================================================

def test_every_swept_tile_respects_the_mma_width():
    """sm_89's instruction is mma.sync.m16n8k16, so `tl.dot` needs every dim >= 16.
    Below that the kernel does not merely run slowly, it fails to compile."""
    for bm, bn, bk, warps, stages in SWEEP_TILES:
        assert bm >= MMA_MIN and bn >= MMA_MIN and bk >= MMA_MIN, (bm, bn, bk)
        assert warps in (1, 2, 4, 8, 16) and stages >= 1, (warps, stages)


def test_legal_rejects_tiles_that_would_index_off_the_edge():
    """The kernel masks only the M edge, so N and K must divide exactly. A tile that
    violated this would read the next row of the weight matrix and be silently wrong."""
    assert not legal(8192, 128, 384, (64, 256, 32, 4, 4))     # N: 384 % 256
    assert not legal(8192, 128, 96, (64, 64, 32, 4, 4))       # N:  96 % 64
    assert not legal(8192, 100, 128, (64, 64, 32, 4, 4))      # K: 100 % 32
    assert not legal(8192, 128, 128, (64, 64, 48, 4, 4))      # K: 128 % 48
    assert legal(8192, 128, 384, (64, 128, 32, 4, 4))         # 384 % 128, 128 % 32
    assert legal(8192, 128, 128, (64, 64, 32, 4, 4))


def test_legal_rejects_a_tile_whose_accumulator_alone_would_spill():
    """A spilling tile is not a candidate, it is a measurement of the spill: the g28
    megakernel read 1.52x spill-free and 2.28x SLOWER once it spilled."""
    # 256x256 fp32 over 4 warps is 512 registers per thread against a hard limit of 255.
    assert not legal(8192, 128, 256, (256, 256, 32, 4, 4))
    assert (128 * 128) / (8 * 32) <= MAX_ACC_REGS_PER_THREAD


def test_legal_declines_a_tile_that_is_mostly_padding():
    """One row of tokens must not be computed on a 256-row tile."""
    assert not legal(64, 128, 128, (256, 64, 32, 8, 3))
    assert legal(8192, 128, 128, (256, 64, 32, 8, 3))


def test_every_announced_shape_has_at_least_one_legal_tile_or_none_at_all():
    """Whatever the sweep decides, it must never crash for want of a candidate: an empty
    tile list has to mean `plan` returns the vendor, not that it raises."""
    for cfg in (BY_ID[i] for i in (1, 2, 4, 5, 8, 9, 12, 13)):
        m, d, f = cfg.tokens, cfg.d_model, cfg.ffn_dim
        assert isinstance(viable_tiles(m, d, 3 * d), list)
        assert viable_tiles(m, d, d), (cfg.id, "d_model")
        assert viable_tiles(m, d, f), (cfg.id, "ffn_dim")


def test_the_probe_is_capped_but_never_above_the_real_shape():
    """v23 caps its probe batch for the same reason: per-program work stops depending on
    M once the grid is several waves deep, and config 14's 3.2M rows at d_model 1024
    would OOM the tuner outright."""
    if not torch.cuda.is_available():
        pytest.skip("probe_rows reads device properties")
    for m, k, n in ((128, 128, 384), (8192, 128, 128), (1_280_000, 128, 128),
                    (3_200_000, 1024, 3072)):
        pm = probe_rows(m, k, n, "cuda")
        assert MMA_MIN <= pm <= m, (m, pm)
    assert probe_rows(128, 128, 384, "cuda") == 128        # small shapes are not capped
    assert probe_rows(8192, 128, 128, "cuda") == 8192
    assert probe_rows(1_280_000, 128, 128, "cuda") < 1_280_000


def test_the_vendor_holds_the_ground_by_a_stated_margin():
    """These kernels run in 6-20 us against a ~1 us event timer, so inside the margin the
    ranking is noise -- and a candidate whose kernel selection varies run to run injects
    that noise into every measurement taken of it (L29)."""
    assert 0.0 < DECISIVE < 0.5


def _executable_source(path: Path) -> str:
    """The module's source with every comment and string literal removed.

    Stripping only `#` lines is not enough and the difference is not academic: the first
    version of this file failed on its own module docstring, which argues AGAINST tanh
    and names the configs the census came from. Prose that cites a config as evidence is
    the point of the docstring; what rule 2 forbids is CODE that looks at one.
    """
    import io
    import tokenize
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL):
                continue
            out.append(tok.string)
    return " ".join(out)


def test_no_config_ids_or_announced_shapes_in_the_predicate_source():
    """CLAUDE.md rule 2, asserted on the source as test_v14_dispatch does."""
    code = _executable_source(REPO / "bench" / "kernels" / "proj_gemm.py").lower()
    for forbidden in ("config", "cfg", "8192", "16384", "10000", "100000", "384"):
        assert forbidden not in code, f"the predicate mentions {forbidden!r}"


def test_the_check_above_can_actually_fail():
    """L38: verify that a check can FAIL before trusting that it passed. The stripper is
    doing real work -- the module docstring contains every forbidden token."""
    raw = (REPO / "bench" / "kernels" / "proj_gemm.py").read_text().lower()
    assert "config" in raw and "8192" in raw, "the docstring should cite its evidence"
    code = _executable_source(REPO / "bench" / "kernels" / "proj_gemm.py")
    assert "roofline" not in code, "a docstring word survived the stripper"
    assert "MAX_ACC_REGS_PER_THREAD" in code, "the stripper ate the code too"


def test_the_gelu_is_the_exact_erf_form():
    """`approximate="none"`. The tanh approximation differs by up to ~1e-3 -- half the
    entire 2e-3 budget spent on an approximation nobody asked for -- and a previous probe
    kernel in this project got it wrong."""
    code = _executable_source(REPO / "bench" / "kernels" / "proj_gemm.py")
    assert "erf" in code
    assert "tanh" not in code
    src = (REPO / "bench" / "kernels" / "proj_gemm.py").read_text()
    assert "0.70710678118654752440" in src, "1/sqrt(2) to full double precision"


# ======================================================================================
# On the device
# ======================================================================================

def _reference(tag="ref_bench_t36"):
    path = REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def _build(config_id, candidate="v36_gemm_gelu", causal=None, padding=0.0):
    from bench.candidates import REGISTRY
    torch._dynamo.reset()          # L36: a shared Dynamo cache silently falls back to eager
    ref = _reference(f"ref_bench_t36_{candidate}_{config_id}")
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
    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        for _ in range(10):
            model(x, m)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                model(x, m)
            torch.cuda.synchronize()
    return sum(1 for ev in prof.events() if str(ev.device_type) == "DeviceType.CUDA") / n


# -------------------------------------------------------------------- the kernel alone

@cuda
@pytest.mark.parametrize("m,k,n,gelu", [(8192, 128, 384, False),
                                        (8192, 128, 128, False),
                                        (8192, 128, 128, True),
                                        (2048, 128, 128, True),
                                        (100, 128, 128, True),
                                        (8192, 32, 32, True)])
def test_kernel_is_never_less_accurate_than_the_vendor(m, k, n, gelu):
    """An IDENTITY argument, not a tolerance argument: same fp16 operands, same fp32
    accumulate, one rounding to fp16. With the GELU in the epilogue it is strictly better
    -- the split path rounds `h` to fp16, writes it to HBM and applies GELU to the
    rounded value, where this applies erf to the fp32 accumulator.

    Checked over EVERY legal tile, not just the one the sweep would pick: the sweep is a
    timing and can land anywhere in this set.
    """
    import torch.nn.functional as F
    from bench.kernels.proj_gemm import proj_gemm
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    bt = torch.randn(k, n, device="cuda", dtype=torch.float16) * 0.1
    bias = torch.randn(n, device="cuda", dtype=torch.float16) * 0.1
    w = bt.t().contiguous()

    exact = F.linear(a.float(), w.float(), bias.float())
    vend = F.linear(a, w, bias)
    if gelu:
        exact = F.gelu(exact, approximate="none")
        vend = F.gelu(vend, approximate="none")
    vend_err = (vend.float() - exact).abs().max().item()

    tiles = viable_tiles(m, k, n)
    assert tiles
    for t in tiles:
        got = proj_gemm(a, bt, bias, t, gelu)
        assert got.shape == (m, n) and got.dtype == torch.float16
        err = (got.float() - exact).abs().max().item()
        assert err <= vend_err * 1.05 + 1e-6, (t, err, vend_err)


@cuda
def test_kernel_reports_its_own_register_and_spill_counts():
    """ncu is unavailable under WSL2 (it denies GPU counters), so the CompiledKernel's
    own numbers are the only occupancy instrument there is. They must be real."""
    from bench.kernels.proj_gemm import kernel_stats
    a = torch.randn(8192, 128, device="cuda", dtype=torch.float16)
    bt = torch.randn(128, 128, device="cuda", dtype=torch.float16)
    bias = torch.randn(128, device="cuda", dtype=torch.float16)
    s = kernel_stats(a, bt, bias, (64, 64, 32, 4, 4), True)
    assert s["n_regs"] and s["n_regs"] > 0
    assert s["n_spills"] == 0, s
    assert s["shared"] and s["shared"] > 0


# ------------------------------------------------------------------ in the model

@cuda
@pytest.mark.parametrize("cid", [9, 10, 1])
def test_mechanism_engages_and_the_gelu_launches_disappear(cid):
    """The four GELU kernels are the part the mechanism GUARANTEES, so they are what is
    asserted -- a DROP against the parent, not an absolute, because both counts undercount
    by a few profiler events and the absolute would pin an unrelated number.

    Configs 1, 9 and 10 all run v34's declined branch with four free-standing `F.linear`
    calls and one free-standing `F.gelu` per layer. Whether the sweep takes the `qkv`
    site varies run to run -- it measures 1.05x-1.18x, straddling the 10% margin -- so
    this asserts the three sites that clear it comfortably, and the count.
    """
    _, _, model, make = _build(cid)
    x, m = make(1234)
    with torch.inference_mode():
        model(x, m)
    assert model.gemm_used, model.gemm_reason
    assert model.gemm_engaged, model.gemm_reason
    for site in ("out", "ffn_in", "ffn_out"):
        assert site in model.gemm_sites, (site, model.gemm_reason)
    assert all(s.get("n_spills") == 0 for s in model.gemm_stats.values()), model.gemm_stats

    _, _, parent, pmake = _build(cid, candidate="v34_launch_bound")
    px, pm = pmake(1234)
    n_parent = _kernels_per_forward(parent, px, pm)
    n_child = _kernels_per_forward(model, x, m)
    # Four GELU launches, one per layer, absorbed into the ffn_in epilogue.
    assert n_parent - n_child >= 3.5, (n_parent, n_child)


@cuda
@pytest.mark.parametrize("cid", [9, 10, 1, 4, 12, 5])
def test_accuracy_at_the_locked_tolerance(cid):
    ref, base, model, make = _build(cid)
    with torch.inference_mode():
        for t in range(3):
            x, m = make(1234 + t)
            res = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
            assert res.passed, (cid, t, float(res.max_abs_error))
    assert model.gemm_used, model.gemm_reason


@cuda
@pytest.mark.parametrize("cid", [9, 12])
def test_margin_is_no_worse_than_the_parents(cid, capsys):
    """L26: a candidate at 94% of the budget and one at 30% are not equally correct.

    The epilogue carries strictly fewer rounding steps than the split path. It does not
    follow that max_abs falls on every shape -- different rounding is not monotonically
    better rounding, and a single seed's max_abs is a noisy order statistic -- so this
    asserts what L4 says the gate actually is (failed_elements) plus a ceiling well
    inside the locked tolerance, and RECORDS the margin so a real drift stays visible.
    """
    ref, base, model, make = _build(cid)
    _, _, parent, pmake = _build(cid, candidate="v34_launch_bound")
    with torch.inference_mode():
        x, m = make(7)
        px, pm = pmake(7)
        ours = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
        theirs = ref.compare_outputs(base(px, pm), parent(px, pm), rtol=RTOL, atol=ATOL)
    with capsys.disabled():
        print(f"\n  cfg{cid} v36 max_abs {float(ours.max_abs_error):.3e} "
              f"({100*float(ours.max_abs_error)/ATOL:.0f}% of the atol budget) "
              f"vs v34 {float(theirs.max_abs_error):.3e} "
              f"({100*float(theirs.max_abs_error)/ATOL:.0f}%)")
    assert ours.passed and int(ours.failed_elements) == 0, int(ours.failed_elements)
    assert float(ours.max_abs_error) <= 0.75 * ATOL, float(ours.max_abs_error)


@cuda
def test_the_predicate_can_say_no():
    """Config 8 is d_model 1024, where cuBLAS selects `cutlass_80_tensorop_f16_s16816gemm`
    and reaches 100.4% of this card's measured peak (F-05). If the predicate cannot
    decline there it is not a predicate, it is a preference -- and CLAUDE.md rule 2 would
    be satisfied only by accident."""
    _, _, model, make = _build(8)
    x, m = make(1234)
    with torch.inference_mode():
        model(x, m)
    assert not model.gemm_used, model.gemm_reason
    assert model.gemm_sites == (), model.gemm_reason
    assert "vendor" in model.gemm_reason


@cuda
def test_declining_costs_no_accuracy_and_takes_the_parents_plan():
    """A declined config must run the parent's code and choose the parent's plan."""
    ref, base, model, make = _build(8)
    _, _, parent, pmake = _build(8, candidate="v34_launch_bound")
    x, m = make(99)
    px, pm = pmake(99)
    with torch.inference_mode():
        got, want, expect = model(x, m), parent(px, pm), base(x, m)
    assert not model.gemm_used, model.gemm_reason
    assert model.launch_reason == parent.launch_reason
    ours = ref.compare_outputs(expect, got, rtol=RTOL, atol=ATOL)
    theirs = ref.compare_outputs(expect, want, rtol=RTOL, atol=ATOL)
    assert ours.passed and theirs.passed
    assert float(ours.max_abs_error) <= float(theirs.max_abs_error) * 1.05 + 1e-9


@cuda
def test_the_fused_branch_leaves_the_ffn_alone():
    """Where v34's megakernel already owns ffn_in, the GELU and ffn_out there is no
    `F.linear` left to replace. Planning one anyway produces a tile that is never
    launched and a reason string that overstates what engaged -- which this candidate did
    until the plan was taught to ask."""
    _, _, model, make = _build(12)
    x, m = make(1234)
    with torch.inference_mode():
        model(x, m)
    assert model.launch_fused_used, model.launch_reason
    assert "ffn_in" not in model.gemm_sites, model.gemm_reason
    assert "ffn_out" not in model.gemm_sites, model.gemm_reason
    assert model._tile_ffn_in is None and model._tile_ffn_out is None


@cuda
def test_the_fallback_runs_when_the_input_is_masked_in_the_middle():
    """A mask that is not right-padded breaks v8's redundancy proof, so `_fastpath` is
    False and the whole substituted `_core` must step aside."""
    ref, base, model, make = _build(9)
    x, m = make(1234)
    m = m.clone()
    m[:, 0] = False                              # a hole, not a suffix
    with torch.inference_mode():
        model(x, m)
        res = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
    assert res.passed, float(res.max_abs_error)


@cuda
def test_non_causal_still_delegates_to_the_unmodified_baseline():
    """v26's fix: the reference benchmark's own default is `causal=False` (L42), and
    every optimization in this lineage was designed under causality."""
    ref, base, model, make = _build(9, causal=False)
    x, m = make(1234)
    with torch.inference_mode():
        res = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
    assert res.passed, float(res.max_abs_error)
    assert model.causal_path.startswith("baseline")


@cuda
def test_different_inputs_do_not_give_the_same_answer():
    """L25/L23: an invariance check catches a class of bug an equivalence check cannot --
    a graph replaying a stale static buffer is correct against itself forever."""
    _, _, model, make = _build(9)
    x1, m1 = make(1)
    x2, m2 = make(2)
    with torch.inference_mode():
        y1 = model(x1, m1).clone()
        y2 = model(x2, m2).clone()
    assert not torch.allclose(y1, y2), "the model ignored its input"


@cuda
def test_repeating_a_call_repeats_its_answer():
    """The other half of L25: after the invariance check above proves the model reads its
    input, this proves it is deterministic across calls -- so a difference measured
    against the reference is a difference in arithmetic, not in scheduling."""
    _, _, model, make = _build(9)
    x1, m1 = make(1)
    x2, m2 = make(2)
    with torch.inference_mode():
        first = model(x1, m1).clone()
        model(x2, m2)
        again = model(x1, m1).clone()
    assert torch.equal(first, again), (first - again).abs().max().item()
