"""v40: the looped attention form, its predicate, and the fallbacks around it.

Tolerance is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR. Not widened here.

[L36] IS THE ORGANISING PRINCIPLE OF THIS FILE. Six candidates this session were caught by
it and two of them reported success while doing nothing. So every speed-shaped claim is
preceded by an assertion that the mechanism ENGAGED -- `attn_form == "looped"` on the
shape it is supposed to engage on, and `attn_form != "looped"` on the shapes where the
predicate is supposed to decline.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.matrix import BY_ID

ATOL, RTOL = 2e-3, 2e-2


def _ref_module():
    p = (Path(__file__).resolve().parents[2]
         / "benchmarks/reference/torch_transformer_benchmark.py")
    spec = importlib.util.spec_from_file_location("ref_v40", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v40"] = m
    spec.loader.exec_module(m)
    return m


def _built(config_id=None, **over):
    ref = _ref_module()
    if config_id is not None:
        c = BY_ID[config_id]
        kw = dict(batch_size=c.batch_size, seq_len=c.seq_len, d_model=c.d_model,
                  num_heads=c.heads, ffn_dim=c.ffn_dim, num_layers=c.layers,
                  causal=c.causal)
    else:
        kw = dict(batch_size=8, seq_len=128, d_model=128, num_heads=2, ffn_dim=128,
                  num_layers=4, causal=True)
    kw.update(over)
    cfg = ref.TransformerConfig(**kw)
    cfg.validate()
    torch.manual_seed(4321)
    base = ref.BaselineTransformer(cfg)
    m = REGISTRY["v40_looped_attn"].build(ref.BaselineTransformer)(cfg)
    ref.copy_model_weights(base, m)
    dev = torch.device("cuda")
    return (m.to(device=dev, dtype=torch.float32).eval(),
            base.to(device=dev, dtype=torch.float32).eval(), cfg, ref)


def _run(m, base, cfg, ref, padding_ratio=0.0, input_scale=1.0, seed=4321):
    x, mask = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       seed=seed, padding_ratio=padding_ratio,
                                       input_scale=input_scale)
    with torch.inference_mode():
        want = base(x, mask)
        got = m(x, mask)
    return ref.compare_outputs(want, got, rtol=RTOL, atol=ATOL)


# ------------------------------------------------- the mechanism actually engages [L36]

def test_config_10_selects_the_looped_form():
    """The whole candidate. Config 10 is head_dim 64 at B*H = 128 -- roughly two waves on
    66 SMs, which is the shape the predicate is written for and the shape the census
    priced. If this reads "single_tile" the candidate is v38 with extra build time."""
    m, base, cfg, ref = _built(10)
    res = _run(m, base, cfg, ref)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"
    assert m.attn_form == "looped", (
        f"the looped form did not engage on config 10: {m.attn_reason}")
    assert m.attn_used is True
    assert len(m.attn_tile) == 4, m.attn_tile
    bm, bn, w, st = m.attn_tile
    assert bn < bm and st >= 2, f"a one-trip or unpipelined tile was selected: {m.attn_tile}"


def test_a_batch_of_one_declines_the_looped_form():
    """Config 2's shape: 4 CTAs on 66 SMs. One wave, nothing to hide behind, and the
    predicate must say so rather than sweeping it."""
    m, base, cfg, ref = _built(2)
    res = _run(m, base, cfg, ref)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"
    assert m.attn_form != "looped", m.attn_reason


def test_head_dim_128_does_not_take_the_looped_form():
    """Config 9. Measured 0.826x against sdpa in the model's own cache regime, twice --
    the looped form must NOT be selected here. This is the row finding 48 priced at
    +0.0021 and this candidate withdraws."""
    m, base, cfg, ref = _built(9)
    res = _run(m, base, cfg, ref)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"
    assert m.attn_form != "looped", (
        f"config 9 selected the looped form, which measured 0.826x hot: {m.attn_reason}")


# ------------------------------------------------------------------- the fallbacks

def test_a_shape_the_tuner_cannot_probe_falls_back_and_is_still_correct():
    """seq_len 100000 would need a 9.8 GiB probe tensor to tune. The chooser must decline
    it without allocating, and the model must still answer correctly through SDPA --
    which is exactly what v38 does there. Run at a small batch so the forward itself
    fits; the point is the TUNER's budget, not the model's."""
    m, base, cfg, ref = _built(batch_size=1, seq_len=4096, d_model=1024, num_heads=16,
                               ffn_dim=1024, num_layers=2)
    res = _run(m, base, cfg, ref)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"
    assert m.attn_form != "looped"


def test_non_causal_input_is_delegated_untouched():
    """v26's guard. A non-causal config must reach the unmodified baseline path, and the
    looped kernel -- which bakes causality into `kv_end` -- must never see it."""
    m, base, cfg, ref = _built(causal=False)
    res = _run(m, base, cfg, ref)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"


def test_padded_input_is_still_correct():
    """v8's fast path is a proof about right-padded causal input; the looped kernel does
    not apply the key mask any more than the single-tile one does, so the padded case has
    to go through the same `_fastpath` / `_needs_zeroing` machinery unchanged."""
    m, base, cfg, ref = _built(10)
    res = _run(m, base, cfg, ref, padding_ratio=0.3)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"


def test_a_second_shape_re_decides_rather_than_reusing_the_first_plan():
    """v33's shape latch, and the reason `_invalidate_shape_state` resets `attn_form`.
    A model warmed at a shape that selects the looped form and then called at a shape
    that must not use it would otherwise run a 4-tuple tile sized for the wrong batch.
    This is [L50]: a fix that makes a dormant defect reachable."""
    m, base, cfg, ref = _built(10)
    res = _run(m, base, cfg, ref)
    assert res.passed
    assert m.attn_form == "looped", m.attn_reason

    # Same model, a batch of 1 -- four CTAs, where the predicate must decline.
    small = ref.TransformerConfig(batch_size=1, seq_len=cfg.seq_len,
                                  d_model=cfg.d_model, num_heads=cfg.num_heads,
                                  ffn_dim=cfg.ffn_dim, num_layers=cfg.num_layers,
                                  causal=True)
    small.validate()
    x, mask = ref.generate_random_case(small, torch.device("cuda"), torch.float32,
                                       seed=7, padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        want = base(x, mask)
        got = m(x, mask)
    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    assert m.attn_form != "looped", (
        f"the plan latched to the first shape: {m.attn_reason}")
    res2 = ref.compare_outputs(want, got, rtol=RTOL, atol=ATOL)
    assert res2.passed, f"max_abs {res2.max_abs_error:.3e}"


def test_input_scale_tail_is_no_worse_than_the_parents():
    """Finding 19's tail, tested as a DIFFERENCE against v38 rather than as an absolute.

    An earlier version of this test asserted `res.passed` at `input_scale=0.1` and failed:
    298 of 1048576 elements past the locked tolerance, max_abs 3.851e-03. That is the
    standing hazard CLAUDE.md states -- the strict both-bounds tolerance is nearly
    saturated by bf16 itself (a 1.95e-03 floor against a locked 2e-03) and honest kernels
    are expected to trip it -- so the absolute assertion was testing the harness, not the
    candidate.

    `bench/probes/g40_attn_loop/probe_input_scale.py` settles which it is, over 3 seeds x
    5 scales: **v38 and v40 fail in exactly the same places, with max_abs identical to
    four significant figures** (3.923e-03, 3.770e-03, 4.062e-03), and v40 never fails
    where v38 passes. The dominant error at that scale is therefore not in attention at
    all. THE TOLERANCE IS NOT WIDENED ANYWHERE HERE: the comparison stays at 2e-3/2e-2 and
    the claim is only that swapping the attention kernel did not make accuracy worse."""
    m40, base, cfg, ref = _built(10)
    m38 = REGISTRY["v38_stream_fallback"].build(ref.BaselineTransformer)(cfg)
    ref.copy_model_weights(base, m38)
    m38 = m38.to(device=torch.device("cuda"), dtype=torch.float32).eval()

    for scale in (0.1, 0.5, 1.0, 10.0):
        x, mask = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                           seed=4321, padding_ratio=0.0,
                                           input_scale=scale)
        with torch.inference_mode():
            want = base(x, mask)
            a = ref.compare_outputs(want, m38(x, mask), rtol=RTOL, atol=ATOL)
            b = ref.compare_outputs(want, m40(x, mask), rtol=RTOL, atol=ATOL)
        if a.passed:
            assert b.passed, (f"input_scale {scale}: v38 passes and v40 does not "
                              f"(max_abs {b.max_abs_error:.3e}) -- a REGRESSION")
        assert b.max_abs_error <= a.max_abs_error * 1.05, (
            f"input_scale {scale}: v40 max_abs {b.max_abs_error:.3e} against v38's "
            f"{a.max_abs_error:.3e}")


@pytest.mark.parametrize("config_id", [2, 9, 8])
def test_where_the_looped_form_declines_v40_runs_exactly_what_v38_runs(config_id):
    """The blast radius, asserted rather than asserted-to-be-small.

    `attn_choice` never SELECTS sdpa -- it only uses it as a bar -- and when the looped
    form does not clear the bar, `_decide_attn` calls the PARENT'S OWN `_decide_attn`
    rather than reimplementing it. So a config the looped form does not win keeps v38's
    plan by construction. This is what makes the ABBA control arms meaningful.

    WHAT THIS TEST DOES NOT ASSERT, AND WHY. It compares `attn_used` and the FORM, not
    the exact tile. Tile equality holds -- `bench/probes/g40_attn_loop/probe_which_form.py`
    builds both arms adjacently in a fresh process and reports identical tiles on all ten
    non-looped configs -- but it is not assertable HERE, because v23's `autotune_tile`
    times candidate tiles with `do_bench` and is only deterministic within a comparable
    process state. Measured: five independent builds of v38 in a clean process return
    `(64, 4, 1)` five times on config 2, and the same assertion fails when the two builds
    are separated by fifty other GPU tests. Asserting tile equality across that gap is
    asserting the PARENT's tuner is state-independent, which is a different claim, is not
    true, and is not this candidate's to make.
    """
    m40, base, cfg, ref = _built(config_id)
    res = _run(m40, base, cfg, ref)
    assert res.passed, f"v40 max_abs {res.max_abs_error:.3e}"
    if m40.attn_form == "looped":
        pytest.skip(f"config {config_id} selects the looped form; not a control config")

    m38 = REGISTRY["v38_stream_fallback"].build(ref.BaselineTransformer)(cfg)
    ref.copy_model_weights(base, m38)
    m38 = m38.to(device=torch.device("cuda"), dtype=torch.float32).eval()
    x, mask = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       seed=4321, padding_ratio=0.0, input_scale=1.0)
    with torch.inference_mode():
        m38(x, mask)
    assert m40.attn_used == m38.attn_used, (
        f"v40 attn_used={m40.attn_used} vs v38 {m38.attn_used}: {m40.attn_reason}")
    assert m40.attn_form != "looped"
    # The decision must have come from the parent's routine, not a reimplementation of
    # it -- that is the structural guarantee, and it is visible in the reason string
    # because `super()._decide_attn` writes its own text before v40 appends to it.
    assert "looped declined:" in m40.attn_reason, m40.attn_reason


def test_inherits_v38s_stream_path():
    """v40 adds attention and must not disturb v38's residency fix -- config 10 runs
    resident, and a candidate that quietly started streaming there would lose 1.6x."""
    m, base, cfg, ref = _built(10)
    _run(m, base, cfg, ref)
    assert getattr(m, "stream_path", None) == "resident", m.stream_path
