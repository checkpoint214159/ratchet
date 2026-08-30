"""v41: the chooser is allowed to hand a shape back to the vendor.

Tolerance is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR. Not widened.

[L36] AND [L38] ARE BOTH ORGANISING PRINCIPLES HERE, and they pull in opposite
directions, which is the point:

  * [L36] the mechanism must be asserted to ENGAGE, not merely to be present. Six
    candidates this session passed every correctness check while doing nothing.
  * [L38] a check must be shown capable of FAILING before a pass from it means anything.
    This candidate's mechanism is a decision rule that, on the announced matrix as
    measured at generation 41, is expected to fire on ZERO configs -- our kernel wins
    everywhere it still applies. A test that only asserts "nothing changed" is
    indistinguishable from a test of an inert candidate.

So the decision rule is tested directly, with the timer scripted in both directions, and
the per-config tests then assert that the rule -- shown to be live -- declines every
announced shape for a stated reason rather than by never running.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels import attn_choice
from bench.matrix import BY_ID

ATOL, RTOL = 2e-3, 2e-2


def _ref_module():
    p = (Path(__file__).resolve().parents[2]
         / "benchmarks/reference/torch_transformer_benchmark.py")
    spec = importlib.util.spec_from_file_location("ref_v41", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v41"] = m
    spec.loader.exec_module(m)
    return m


def _built(name, config_id=None, **over):
    ref = _ref_module()
    if config_id is not None:
        c = BY_ID[config_id]
        kw = dict(batch_size=c.batch_size, seq_len=c.seq_len, d_model=c.d_model,
                  num_heads=c.heads, ffn_dim=c.ffn_dim, num_layers=c.layers,
                  causal=c.causal)
    else:
        kw = dict(batch_size=8, seq_len=128, d_model=128, num_heads=4, ffn_dim=128,
                  num_layers=4, causal=True)
    kw.update(over)
    cfg = ref.TransformerConfig(**kw)
    cfg.validate()
    torch.manual_seed(4321)
    base = ref.BaselineTransformer(cfg)
    m = REGISTRY[name].build(ref.BaselineTransformer)(cfg)
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


# ------------------------------------------------------- the decision rule, both ways
#
# [L38]. `autotune_vendor` is the whole candidate, and on this matrix it is expected to
# decline every shape. These two tests script the timer so the rule is exercised in both
# directions on real hardware -- the kernel is still compiled, launched and correctness-
# checked; only the two timings are substituted.

@pytest.fixture
def scripted_timer(monkeypatch):
    """Replace `_time` with a queue of milliseconds, consumed in call order.

    `autotune_vendor` times the incumbent first and sdpa second, and this asserts that
    ordering by exhausting the queue: a rule that timed them in the other order, or timed
    something extra, fails loudly instead of silently reading the wrong arm.
    """
    def install(values):
        q = list(values)
        def fake(fn, reps):
            assert q, "autotune_vendor timed more arms than the script provides"
            return q.pop(0)
        monkeypatch.setattr(attn_choice, "_time", fake)
        return q
    return install


def test_vendor_is_selected_when_it_wins_decisively(scripted_timer):
    """incumbent 10.0 us, sdpa 8.0 us -> 1.25x, well past DECISIVE. Must return."""
    q = scripted_timer([0.010, 0.008])
    why = attn_choice.autotune_vendor(128, 32, 4, 64, (64, 4, 1))
    assert not q, "an arm went untimed"
    assert "sdpa" in why and "decisively" in why, why
    assert "1.250x" in why, why


def test_vendor_is_refused_inside_the_decisive_margin(scripted_timer):
    """incumbent 10.0 us, sdpa 9.5 us -> 1.053x. Inside 10%: the incumbent holds."""
    q = scripted_timer([0.010, 0.0095])
    with pytest.raises(ValueError, match="did not clear"):
        attn_choice.autotune_vendor(128, 32, 4, 64, (64, 4, 1))
    assert not q


def test_a_tie_goes_to_the_incumbent(scripted_timer):
    """Exactly equal is not a win. The margin exists because the timer cannot resolve
    these kernels below ~10%, and ties must not churn the plan."""
    scripted_timer([0.010, 0.010])
    with pytest.raises(ValueError, match="did not clear"):
        attn_choice.autotune_vendor(128, 32, 4, 64, (64, 4, 1))


def test_an_incorrect_incumbent_is_never_ranked(scripted_timer):
    """Correctness before timing, per arm -- including the incumbent. A tile that does
    not reproduce the reference must not be able to win OR lose a comparison."""
    scripted_timer([])
    # block_m below the MMA width is not a legal tile; the launcher must not produce a
    # matching result, and the routine must refuse before it reaches the timer.
    with pytest.raises(Exception):
        attn_choice.autotune_vendor(128, 32, 4, 64, (8, 4, 1))


def test_the_probe_declines_a_shape_it_cannot_afford():
    """seq_len 100000 is a 9.8 GiB probe tensor on a 16 GiB card. The tuner must not OOM
    the model it is tuning -- config 14's shape declines without allocating."""
    with pytest.raises(ValueError, match="budget"):
        attn_choice.autotune_vendor(100000, 64, 16, 32, (64, 4, 1))


# ------------------------------------------------- the rule is ASKED, and it declines

def test_the_vendor_check_runs_and_declines_on_config_7():
    """head_dim 8, where `attn_single_tile` is at its strongest. The audit measured the
    kernel at 6.556 us against sdpa's 11.537 hot -- 1.76x. The check must RUN (so the
    reason is not "not asked") and must keep the kernel."""
    m, base, cfg, ref = _built("v41_vendor_aware_attn", 7)
    res = _run(m, base, cfg, ref)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"
    assert m.attn_used is True and m.attn_form == "single_tile", m.attn_reason
    assert m.attn_vendor_reason.startswith("kept the kernel"), m.attn_vendor_reason
    assert "did not clear" in m.attn_vendor_reason, m.attn_vendor_reason


def test_config_10_is_not_asked_because_the_looped_form_already_beat_the_vendor():
    """`autotune_looped` carries sdpa as a hard floor, so a shape it wins has already
    beaten the vendor. Re-litigating it would be a second 10% margin on the same
    question -- the bug finding 50 caught in its own first draft."""
    m, base, cfg, ref = _built("v41_vendor_aware_attn", 10)
    res = _run(m, base, cfg, ref)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"
    if m.attn_form == "looped":
        assert m.attn_vendor_reason == "not asked: plan is looped", m.attn_vendor_reason
    else:
        # The looped tuner declined this run; the vendor check is then the safety net
        # this candidate exists to provide, and it must have RUN.
        assert m.attn_vendor_reason != "not asked", m.attn_reason


def test_declined_shapes_are_not_asked():
    """Config 9 is head_dim 128: `attn_single_tile` declines it and the model already
    runs sdpa. There is nothing to hand over, and the check must not spend build time
    proving it."""
    m, base, cfg, ref = _built("v41_vendor_aware_attn", 9)
    res = _run(m, base, cfg, ref)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"
    assert m.attn_used is False
    assert m.attn_vendor_reason.startswith("not asked"), m.attn_vendor_reason


# --------------------------------------------------------------- inherited behaviour

def test_the_reason_is_reset_on_a_shape_change():
    """`attn_vendor_reason` is introduced by v41, so v37's `SHAPE_LATCHED` -- derived at
    v37's build time -- cannot name it. [L50]: this lineage's recurring defect is state
    that outlives the shape it was decided for."""
    m, base, cfg, ref = _built("v41_vendor_aware_attn", 7)
    _run(m, base, cfg, ref)
    assert m.attn_vendor_reason != "not asked"
    m._invalidate_shape_state()
    assert m.attn_vendor_reason == "not asked"


def test_causal_is_exact_and_the_output_still_matches_with_padding():
    """Every announced row is causal (finding 32) and the padded case exercises v8's
    proof that the key mask is redundant. Both must hold whichever path was chosen."""
    m, base, cfg, ref = _built("v41_vendor_aware_attn", 3)
    res = _run(m, base, cfg, ref, padding_ratio=0.3)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"
