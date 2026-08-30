"""v25: where an fp16 MMA accumulator is affordable, and where it is not.

The point of this file is the BOUNDARY, not a speedup. It pins four things:

  1. the hardware reading is true (the f16-accumulate mma is really emitted, and really
     faster in isolation) -- so the closure is not "we could not get it to work";
  2. the fused FFN is bandwidth-bound, which is WHY the faster instruction buys nothing;
  3. the fp16 accumulator's error exceeds the locked tolerance at every contraction depth
     the hardware can issue, including the shallowest one;
  4. the shipped candidate is numerically identical to its parent, and the fallback path
     is correct.

`test_per_config_margin_report` is the deliverable: max_abs and the fraction of the
locked budget it spends, per config, for the shipped arm and each forced fp16 arm.
"""
import math
import os

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels.ffn_accum import (MEASURED_K16_MAX_ABS, MMA_MIN_K,
                                     accumulator_affordable, affordable_region_is_empty,
                                     arithmetic_intensity, device_roofline,
                                     fused_ffn_accum, max_affordable_k, mma_bound,
                                     min_mma_bound_dim, no_shape_satisfies_both,
                                     ridge_point)
from bench.matrix import MATRIX

ATOL, RTOL = 2e-3, 2e-2
PEAK, BW = 88.2e12, 613.662149685899e9      # ledger/device.json, measured


def _ref():
    import importlib.util
    import sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v25", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v25"] = m
    spec.loader.exec_module(m)
    return m


def _margin(got, want):
    """(max_abs, failing elements) under the harness's OR criterion."""
    d = (got.float() - want.float()).abs()
    ok = (d <= ATOL) | (d <= RTOL * want.float().abs())
    return d.max().item(), int((~ok).sum().item())


# ------------------------------------------------- 1. the hardware reading really is true

def test_fp16_out_dtype_really_emits_an_f16_accumulate_mma():
    """L34: three agents agreeing on a datasheet is one reading, not three votes. Read it
    off the generated PTX instead."""
    import re
    import triton
    import triton.language as tl

    @triton.jit
    def _k(A, B, C, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, F16: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        a = tl.load(A + rm[:, None] * K + tl.arange(0, K)[None, :])
        b = tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
        if F16 == 1:
            acc = tl.dot(a, b, out_dtype=tl.float16).to(tl.float32)
        else:
            acc = tl.dot(a, b, out_dtype=tl.float32)
        tl.store(C + rm[:, None] * N + tl.arange(0, N)[None, :], acc)

    A = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    B = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    C = torch.empty(64, 64, device="cuda", dtype=torch.float32)
    asm = {f: _k[(1,)](A, B, C, K=64, N=64, BM=64, F16=f).asm["ptx"] for f in (0, 1)}

    m32 = set(re.findall(r"mma\.sync[\w.]*", asm[0]))
    m16 = set(re.findall(r"mma\.sync[\w.]*", asm[1]))
    assert any(i.endswith("f32.f16.f16.f32") for i in m32), m32
    assert any(i.endswith("f16.f16.f16.f16") for i in m16), m16


@pytest.mark.gpu
def test_the_f16_accumulate_mma_really_is_faster_in_isolation():
    """If this ever stops holding, the closure below is closing the wrong thing."""
    import triton
    import triton.language as tl
    from triton.testing import do_bench

    @triton.jit
    def _k(A, B, C, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
           REP: tl.constexpr, F16: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        a = tl.load(A + rm[:, None] * K + tl.arange(0, K)[None, :])
        b = tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
        if F16 == 1:
            acc = tl.zeros((BM, N), dtype=tl.float16)
            for _ in tl.static_range(REP):
                acc += tl.dot(a, b, out_dtype=tl.float16)
                a = (acc * 0.0).to(tl.float16) + a
            out = acc.to(tl.float32)
        else:
            acc = tl.zeros((BM, N), dtype=tl.float32)
            for _ in tl.static_range(REP):
                acc += tl.dot(a, b, out_dtype=tl.float32)
                a = (acc * 0.0).to(tl.float16) + a
            out = acc
        tl.store(C + rm[:, None] * N + tl.arange(0, N)[None, :], out)

    M, K, N, BM = 4096, 128, 128, 64
    A = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
    B = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)
    g = (triton.cdiv(M, BM),)
    t = {f: do_bench(lambda f=f: _k[g](A, B, C, K=K, N=N, BM=BM, REP=16, F16=f, num_warps=8),
                     warmup=50, rep=200) for f in (0, 1)}
    print(f"\nMMA-saturated loop: fp32-acc {t[0]*1e3:.1f} us, fp16-acc {t[1]*1e3:.1f} us, "
          f"{t[0]/t[1]:.3f}x")
    assert t[0] / t[1] > 1.2, f"expected the documented ~2x-class win, got {t[0]/t[1]:.3f}x"


# --------------------------------- 2. and the FFN is nowhere near it (the speed closure)

@pytest.mark.gpu
def test_the_fused_ffn_is_bandwidth_bound_not_mma_bound():
    """The reason the 1.57x instruction buys nothing. At config 6's token count the
    kernel already runs at ~99% of the device's MEASURED bandwidth, so there is no MMA
    time left to reclaim."""
    from triton.testing import do_bench

    M, D = 1_280_000, 128
    torch.manual_seed(0)
    xn = torch.randn(M, D, device="cuda", dtype=torch.float16)
    res = torch.randn(M, D, device="cuda", dtype=torch.float32)
    w1 = (torch.randn(D, D, device="cuda") / D ** 0.5).half()
    b1 = (torch.randn(D, device="cuda") / D ** 0.5).half()
    w2 = (torch.randn(D, D, device="cuda") / D ** 0.5).half()
    b2 = (torch.randn(D, device="cuda") / D ** 0.5).half()

    t = do_bench(lambda: fused_ffn_accum(xn, res, w1, b1, w2, b2, 64, 8, 0, 0),
                 warmup=30, rep=100) * 1e-3
    byts = M * (D * 2 + D * 4 + D * 4) + 2 * D * D * 2
    achieved = byts / t
    print(f"\nfused FFN @ {M} tokens: {achieved/1e9:.1f} GB/s = {achieved/BW*100:.1f}% of "
          f"measured bandwidth; intensity {arithmetic_intensity(D, D):.1f} FLOP/B vs "
          f"ridge {ridge_point(PEAK, BW):.1f}")
    assert achieved / BW > 0.90, "kernel is not bandwidth-saturated; re-open the question"
    assert arithmetic_intensity(D, D) < ridge_point(PEAK, BW)


@pytest.mark.gpu
def test_fp16_accumulate_buys_nothing_on_the_shape_the_kernel_exists_for():
    """Config 6 is the largest row in the matrix and ~48s of a 112s sweep. The whole
    proposal lives or dies here."""
    from triton.testing import do_bench

    M, D = 1_280_000, 128
    torch.manual_seed(0)
    xn = torch.randn(M, D, device="cuda", dtype=torch.float16)
    res = torch.randn(M, D, device="cuda", dtype=torch.float32)
    w1 = (torch.randn(D, D, device="cuda") / D ** 0.5).half()
    b1 = (torch.randn(D, device="cuda") / D ** 0.5).half()
    w2 = (torch.randn(D, D, device="cuda") / D ** 0.5).half()
    b2 = (torch.randn(D, device="cuda") / D ** 0.5).half()

    times = {}
    for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
        times[(a, b)] = do_bench(
            lambda a=a, b=b: fused_ffn_accum(xn, res, w1, b1, w2, b2, 64, 8, a, b),
            warmup=30, rep=100)
    base = times[(0, 0)]
    for k, v in times.items():
        print(f"  acc{k}  {v*1e3:8.1f} us  {base/v:.3f}x")
    best = max(base / v for v in times.values())
    assert best < 1.03, (
        f"fp16 accumulate now buys {best:.3f}x on config 6's shape -- it bought 1.000x "
        f"on 2026-08-30. REOPEN docs/findings/30.")


# ------------------------------- 3. the error closure: no legal K fits the locked budget

def test_the_accuracy_ceiling_and_the_speed_floor_are_disjoint():
    """The scissors. Both bounds come from measured device properties and fp16's
    mantissa, so this is a statement about the hardware, not about this matrix."""
    disjoint, k_ceiling, d_floor = no_shape_satisfies_both(PEAK, BW, ATOL)
    print(f"\naccuracy ceiling K <= {k_ceiling:.1f};  speed floor d_model >= {d_floor:.1f}"
          f"  -> gap {d_floor/k_ceiling:.1f}x")
    assert disjoint
    assert not mma_bound(128, 128, PEAK, BW), "d_model=128 is memory-bound"
    assert not accumulator_affordable(128, ATOL), "K=128 does not fit 2e-3"


@pytest.mark.gpu
def test_even_the_shallowest_legal_mma_depth_misses_the_budget():
    """Stronger than the scissors, and what actually closes the direction.

    The model leaves a window exactly one legal value wide (K <= 16.8, and the hardware's
    shallowest MMA is K=16). Measured at that value the accumulator still misses."""
    import triton
    import triton.language as tl

    @triton.jit
    def _g(A, B, C32, C16, K: tl.constexpr, N: tl.constexpr, BM: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        a = tl.load(A + rm[:, None] * K + tl.arange(0, K)[None, :])
        b = tl.load(B + tl.arange(0, K)[:, None] * N + tl.arange(0, N)[None, :])
        o = rm[:, None] * N + tl.arange(0, N)[None, :]
        tl.store(C32 + o, tl.dot(a, b, out_dtype=tl.float32))
        tl.store(C16 + o, tl.dot(a, b, out_dtype=tl.float16).to(tl.float32))

    assert MMA_MIN_K <= max_affordable_k(ATOL) < 2 * MMA_MIN_K, (
        "the model no longer leaves exactly one legal K; re-derive the closure")

    rows = []
    for K in (16, 32, 64, 128, 256):
        torch.manual_seed(0)
        M, N, BM = 2048, 128, 64
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = (torch.randn(K, N, device="cuda") / K ** 0.5).half()   # unit-variance output
        c32 = torch.empty(M, N, device="cuda", dtype=torch.float32)
        c16 = torch.empty(M, N, device="cuda", dtype=torch.float32)
        _g[(triton.cdiv(M, BM),)](a, b, c32, c16, K=K, N=N, BM=BM, num_warps=8)
        ref = a.float() @ b.float()
        e32 = (c32 - ref).abs().max().item()
        e16 = (c16 - ref).abs().max().item()
        rows.append((K, e32, e16))
        print(f"  K={K:>5}  fp32-acc {e32:.3e}   fp16-acc {e16:.3e}  "
              f"({e16/ATOL*100:6.1f}% of budget)")

    k16 = rows[0][2]
    assert k16 > ATOL, (
        f"an fp16 accumulator at K=16 now fits the budget ({k16:.3e} <= {ATOL:g}). "
        f"That is the one result that would REOPEN this direction.")
    assert math.isclose(k16, MEASURED_K16_MAX_ABS, rel_tol=0.5), (
        f"K=16 error moved materially from the recorded {MEASURED_K16_MAX_ABS:.3e}")
    assert affordable_region_is_empty(ATOL, measured_k16_max_abs=k16)
    for _K, e32, e16 in rows:
        assert e32 < ATOL, "the fp32 accumulator must stay far inside budget"


# ------------------------------------------------- 4. the candidate: declines, and is safe

@pytest.mark.gpu
def test_candidate_declines_and_says_why():
    """v14's is_tuned discipline: an untuned path must never be reported as a tuned one,
    and L38 -- a guard that cannot be observed cannot be trusted."""
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=8, seq_len=64, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    cand = REGISTRY["v25_fp16_accum"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    with torch.no_grad():
        cand(torch.randn(8, 64, 128, device="cuda"))
    print(f"\naccum_reason: {cand.accum_reason}")
    assert cand.accum_mode == (0, 0)
    assert "declined" in cand.accum_reason
    assert "memory-bound" in cand.accum_reason


def test_the_predicate_reads_shapes_and_device_properties_only():
    """CLAUDE.md rule 2. No config ids, no announced shape constants."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "bench/candidates/v25_fp16_accum.py").read_text()
    body = src.split('"""', 2)[-1]          # skip the docstring, which quotes measurements
    for c in MATRIX:
        assert f"config {c.id}" not in body.lower()
    for lit in ("batch_size ==", "seq_len ==", "== 10000", "== 1280000"):
        assert lit not in body


def test_the_predicate_responds_to_the_device_not_to_this_matrix():
    """On a card with a higher peak-FLOPs-to-bandwidth ratio the ridge point rises and the
    decline gets STRONGER, and on a hypothetically balanced card it would flip. A
    predicate that cannot flip is a hardcoded answer wearing a costume."""
    assert not mma_bound(128, 128, PEAK, BW)
    assert mma_bound(128, 128, PEAK / 20, BW), "must be able to flip on other hardware"
    assert min_mma_bound_dim(PEAK, BW) > min_mma_bound_dim(PEAK / 4, BW)


@pytest.mark.gpu
def test_shipped_candidate_is_bit_identical_to_its_parent():
    """The predicate declines everywhere, so v25 as shipped IS v18. Anything else means
    the fp32 path drifted."""
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=8, seq_len=64, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    out = {}
    for name in ("v18_capture_insurance", "v25_fp16_accum"):
        torch._dynamo.reset()               # L36: shared Dynamo cache silently degrades
        c = REGISTRY[name].build(ref.BaselineTransformer)(cfg).cuda().eval()
        ref.copy_model_weights(base, c)
        torch.manual_seed(3)
        x = torch.randn(8, 64, 128, device="cuda")
        with torch.no_grad():
            out[name] = c(x).float().clone()
    assert torch.equal(out["v18_capture_insurance"], out["v25_fp16_accum"])


@pytest.mark.gpu
def test_forced_fp16_arms_are_still_wired_to_a_working_kernel():
    """The falsifier must exercise a real path, not a broken one (L36). Forced arms must
    RUN and produce finite output -- their accuracy is the next test's business."""
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=8, seq_len=64, d_model=128, num_heads=4,
                                ffn_dim=128, num_layers=4, causal=True)
    os.environ["RATCHET_FORCE_ACCUM"] = "1,1"
    try:
        torch._dynamo.reset()
        c = REGISTRY["v25_fp16_accum"].build(ref.BaselineTransformer)(cfg).cuda().eval()
        with torch.no_grad():
            y = c(torch.randn(8, 64, 128, device="cuda"))
        assert c.accum_mode == (1, 1) and "forced" in c.accum_reason
        assert torch.isfinite(y).all()
    finally:
        del os.environ["RATCHET_FORCE_ACCUM"]


@pytest.mark.gpu
def test_wide_model_falls_back_and_is_still_correct():
    """d_model=1024 is above the ridge point, so the SPEED predicate would admit it --
    but the fused kernel declines it on shared memory and the accumulator predicate
    declines it on K. The fallback must be correct, not merely taken."""
    ref = _ref()
    cfg = ref.TransformerConfig(batch_size=2, seq_len=32, d_model=1024, num_heads=4,
                                ffn_dim=1024, num_layers=1, causal=True)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    torch._dynamo.reset()
    cand = REGISTRY["v25_fp16_accum"].build(ref.BaselineTransformer)(cfg).cuda().eval()
    ref.copy_model_weights(base, cand)
    x = torch.randn(2, 32, 1024, device="cuda")
    with torch.no_grad():
        want, got = base(x), cand(x)
    assert mma_bound(1024, 1024, PEAK, BW), "d_model=1024 IS compute-bound"
    assert not accumulator_affordable(1024, ATOL), "and its K is hopeless"
    max_abs, bad = _margin(got, want)
    print(f"\nd_model=1024 fallback: max_abs {max_abs:.3e}, {bad} failing")
    assert bad == 0


# ---------------------------------------------------------------- THE DELIVERABLE

@pytest.mark.gpu
@pytest.mark.parametrize("cid", [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13])
def test_per_config_margin_report(cid):
    """Per-config max_abs margin for the shipped arm and each forced fp16 arm.

    The shipped arm must pass. The forced arms are REPORTED, not asserted -- their
    numbers are the evidence that closes the direction, and pinning them as failures
    would make this file fail for the right reason in the wrong way.

    Config 8 is excluded (d_model=1024 exceeds the kernel's shared-memory gate) and is
    covered by test_wide_model_falls_back_and_is_still_correct. Config 6's batch is
    capped -- 10000 batches is minutes per trial -- and the cap is REPORTED, because a
    clamped shape is not the announced shape.
    """
    ref = _ref()
    cfg_row = next(c for c in MATRIX if c.id == cid)
    batch = min(cfg_row.batch_size, 256)
    cfg = ref.TransformerConfig(batch_size=batch, seq_len=cfg_row.seq_len,
                                d_model=cfg_row.d_model, num_heads=cfg_row.heads,
                                ffn_dim=cfg_row.d_model, num_layers=cfg_row.layers,
                                causal=True)
    torch.manual_seed(0)
    base = ref.BaselineTransformer(cfg).cuda().eval()
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
    with torch.no_grad():
        want = base(x).float()

    report = []
    for arm in ("shipped", "1,0", "0,1", "1,1"):
        if arm == "shipped":
            os.environ.pop("RATCHET_FORCE_ACCUM", None)
        else:
            os.environ["RATCHET_FORCE_ACCUM"] = arm
        try:
            torch._dynamo.reset()
            c = REGISTRY["v25_fp16_accum"].build(ref.BaselineTransformer)(cfg).cuda().eval()
            ref.copy_model_weights(base, c)
            with torch.no_grad():
                got = c(x)
            max_abs, bad = _margin(got, want)
            report.append((arm, max_abs, bad, c.accum_mode, c.fused_ffn_used))
        finally:
            os.environ.pop("RATCHET_FORCE_ACCUM", None)

    clamp = "" if batch == cfg_row.batch_size else f"  [batch clamped {cfg_row.batch_size}->{batch}]"
    print(f"\ncfg {cid}  d_model={cfg.d_model} heads={cfg.num_heads} seq={cfg.seq_len} "
          f"B={cfg.batch_size} tokens={batch*cfg_row.seq_len}{clamp}")
    for arm, max_abs, bad, mode, used in report:
        print(f"    {arm:>8} acc={mode} fused={int(used)}  max_abs {max_abs:.3e}  "
              f"{max_abs/ATOL*100:7.1f}% of budget  failing={bad}")

    shipped = report[0]
    assert shipped[3] == (0, 0)
    assert shipped[2] == 0, (
        f"config {cid}: shipped arm has {shipped[2]} elements outside the locked "
        f"tolerance, max_abs={shipped[1]:.3e}")

    # L36: assert the mechanism actually RAN. Without this the forced arms can measure
    # identically to the shipped one because the kernel containing them was skipped, and
    # the report would be a page of numbers about nothing. It happened on the first run.
    for arm, max_abs, _bad, _mode, used in report[1:]:
        assert used, f"config {cid} arm {arm}: fused kernel never ran"
        assert max_abs != shipped[1], (
            f"config {cid} arm {arm}: identical to the fp32 arm ({max_abs:.3e}) -- the "
            f"forced accumulator did not take effect")
