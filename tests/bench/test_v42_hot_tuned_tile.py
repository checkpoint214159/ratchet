"""v42: the tile sweep is ranked by an instrument that can resolve the arms it ranks.

Tolerance is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR. Not widened.
Causal stays exact; every config in the matrix is causal and `config.causal` is honoured
by construction (the kernel masks with `rn <= rm`, and these tests build from `BY_ID`).

[L36] IS THE ORGANISING PRINCIPLE. Seven candidates this session were silently inert, and
this one's entire diff is a single class attribute -- the easiest possible thing to leave
unwired. So the mechanism is asserted to ENGAGE, and "engage" here has a specific,
checkable meaning: **`autotune_tile` must return a DIFFERENT tile under v42 than under its
parent, on the shape the candidate was built for, having chosen it from shapes and measured
device properties alone.**

The speed claim is asserted only after that. A test that showed config 2 got faster without
showing the tuner changed its mind would not distinguish this candidate from noise.

[L38] pulls the other way and is served by the scripted-timer tests: the decision rule is
driven in both directions with synthetic timings, so a pass from the on-hardware tests
means something. And the tuner is shown to be capable of NOT changing its mind -- nine of
the ten accepted shapes must select the identical tile under both timers, which is the
candidate's own byte-identical-control claim tested rather than assumed.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels import attn_choice, attn_single_tile as ast
from bench.matrix import BY_ID, MATRIX

ATOL, RTOL = 2e-3, 2e-2

# The shape the candidate was built for, named by its SHAPE and not by its id. Config 2
# is (B=1, D=128, H=4, S=128, L=4) -- the only announced row whose batch is 1, which is
# why the flushed timer's quantum swallowed it. Written out so this file states a shape
# the way the dispatch predicates do, and so it still means something if the matrix moves.
TINY = dict(batch=1, seq_len=128, heads=4, head_dim=32)


def _ref_module():
    p = (Path(__file__).resolve().parents[2]
         / "benchmarks/reference/torch_transformer_benchmark.py")
    spec = importlib.util.spec_from_file_location("ref_v42", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v42"] = m
    spec.loader.exec_module(m)
    return m


def _built(name, config_id):
    ref = _ref_module()
    c = BY_ID[config_id]
    cfg = ref.TransformerConfig(
        batch_size=c.batch_size, seq_len=c.seq_len, d_model=c.d_model,
        num_heads=c.heads, ffn_dim=c.ffn_dim, num_layers=c.layers, causal=c.causal)
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


# --------------------------------------------------------------- the wiring is real
#
# The diff is one class attribute. If it is misspelled, shadowed, or overwritten by an
# ancestor's `_invalidate_shape_state`, everything below would still pass on correctness
# and the candidate would be its parent with extra build time. Check the wire first.

def test_the_candidate_selects_the_hot_timer_and_the_parent_does_not():
    parent = REGISTRY["v41_vendor_aware_attn"].build(object)
    child = REGISTRY["v42_hot_tuned_tile"].build(object)
    assert parent.attn_tile_timer is None, (
        "the parent must keep `autotune_tile`'s own default -- a control arm that has "
        "silently moved to the hot timer measures nothing")
    assert child.attn_tile_timer is ast.hot_time


def test_the_timer_default_is_the_parents_and_is_named():
    """`timer=None` must mean exactly what v23 through v41 shipped, and both timers must
    be identifiable from the reason string [L36]."""
    import inspect
    assert inspect.signature(ast.autotune_tile).parameters["timer"].default is None
    assert ast.flushed_time.__name__ == "flushed_time"
    assert ast.hot_time.__name__ == "hot_time"


# --------------------------------------------------- the decision rule, both ways [L38]

@pytest.fixture
def scripted_timer():
    """Drive `autotune_tile` with a queue of milliseconds, consumed in sweep order.

    The kernels are still compiled, launched and correctness-checked on real hardware;
    only the numbers they are ranked on are substituted. That is what makes the two
    directions below assertions about the RULE rather than about the card.
    """
    def install(values):
        q = list(values)

        def fake(fn, reps):
            assert q, "the sweep timed more arms than the script provides"
            return q.pop(0)
        return fake
    return install


def _grid(seq_len, head_dim):
    props = torch.cuda.get_device_properties("cuda")
    return ast.viable_tiles(seq_len, head_dim, props.regs_per_multiprocessor,
                            props.max_threads_per_multi_processor, props.warp_size)


def test_a_challenger_that_wins_decisively_displaces_the_derived_tile(scripted_timer):
    """The first arm in the grid reads 1.0 us, everything else 10.0. It must win."""
    tiles = _grid(TINY["seq_len"], TINY["head_dim"])
    assert len(tiles) > 1
    ms = [0.001] + [0.010] * (len(tiles) - 1)
    tile, why = ast.autotune_tile(TINY["seq_len"], TINY["head_dim"], TINY["heads"],
                                  TINY["batch"], timer=scripted_timer(ms))
    assert tile == tiles[0], why
    assert "decisively" in why and "hot" not in why.split(":")[0]


def test_a_tie_across_the_whole_grid_keeps_the_derived_tile(scripted_timer):
    """THE EXACT FAILURE THIS CANDIDATE EXISTS TO FIX, reproduced synthetically.

    Under the flushed timer five of config 2's eight arms reported the identical 5.120
    us. This is that table: every arm equal. The rule must fall through to the derived
    tile -- which is correct behaviour given the input, and is precisely why the input
    being a table of ties was the defect rather than the rule being wrong.
    """
    tiles = _grid(TINY["seq_len"], TINY["head_dim"])
    props = torch.cuda.get_device_properties("cuda")
    derived = ast.choose_tile(TINY["seq_len"], TINY["head_dim"],
                              props.regs_per_multiprocessor,
                              props.max_threads_per_multi_processor, props.warp_size)
    tile, why = ast.autotune_tile(TINY["seq_len"], TINY["head_dim"], TINY["heads"],
                                  TINY["batch"],
                                  timer=scripted_timer([0.005] * len(tiles)))
    assert tile == derived, why
    assert "confirmed" in why


def test_a_challenger_inside_the_decisive_margin_does_not_displace(scripted_timer):
    """9.5 against the derived tile's 10.0 is 1.053x. Inside 10%: the incumbent holds."""
    tiles = _grid(TINY["seq_len"], TINY["head_dim"])
    props = torch.cuda.get_device_properties("cuda")
    derived = ast.choose_tile(TINY["seq_len"], TINY["head_dim"],
                              props.regs_per_multiprocessor,
                              props.max_threads_per_multi_processor, props.warp_size)
    ms = [0.0095 if t != derived else 0.010 for t in tiles]
    tile, why = ast.autotune_tile(TINY["seq_len"], TINY["head_dim"], TINY["heads"],
                                  TINY["batch"], timer=scripted_timer(ms))
    assert tile == derived, why


def test_an_arm_that_fails_the_tolerance_is_dropped_before_it_is_timed(monkeypatch):
    """Correctness before timing, per arm -- the gate `autotune_tile` did not have.

    A kernel that returns garbage is made the fastest thing on the card. It must not be
    selected, and it must not even be timed.
    """
    real = ast.single_tile_attention
    poisoned = _grid(TINY["seq_len"], TINY["head_dim"])[0]

    def wrong(qkv, heads, head_dim, scale, bm, w, st=1):
        out = real(qkv, heads, head_dim, scale, bm, w, st)
        return (out + 1.0) if (bm, w, st) == poisoned else out
    monkeypatch.setattr(ast, "single_tile_attention", wrong)

    timed = []

    def watch(fn, reps):
        timed.append(True)
        return 0.001 if len(timed) == 1 else 0.010
    tile, why = ast.autotune_tile(TINY["seq_len"], TINY["head_dim"], TINY["heads"],
                                  TINY["batch"], timer=watch)
    assert tile != poisoned, f"a wrong kernel won the sweep: {why}"
    assert "dropped on tolerance" in why, why


# ---------------------------------------------------------- the mechanism, on hardware

def test_the_hot_timer_changes_the_tile_on_the_tiny_shape():
    """[L36] THE CANDIDATE'S CENTRAL CLAIM, asserted as a change of MIND, not of speed.

    Same routine, same grid, same probe batch, same `DECISIVE` bar, same tiebreak -- the
    only difference is the function that turns an arm into a number. If the two timers
    agree here, this candidate is inert and every number below it is noise.

    Nothing about config 2 is named: the shape is passed in, and `autotune_tile` reaches
    `(16, 4, 1)` from that shape plus `torch.cuda.get_device_properties`.
    """
    kw = dict(seq_len=TINY["seq_len"], head_dim=TINY["head_dim"], heads=TINY["heads"],
              batch=TINY["batch"])
    flushed, why_f = ast.autotune_tile(**kw, timer=ast.flushed_time)
    hot, why_h = ast.autotune_tile(**kw, timer=ast.hot_time)
    assert hot != flushed, (
        f"the mechanism did not engage: both timers chose {hot}\n"
        f"  flushed: {why_f}\n  hot:     {why_h}")
    assert "hot_time" in why_h and "decisively" in why_h, why_h
    # The margin must clear the inherited bar with room, not scrape it -- the whole
    # argument is that this is a 1.28x call the old instrument reported as a tie.
    assert "flushed_time" in why_f


def test_the_hot_timer_ranks_the_chosen_tile_well_clear_of_the_derived_one():
    """The margin, measured directly. Finding 51 read 1.283x; the g42 probe read
    1.280x and 1.278x. Assert well past `DECISIVE` rather than at a fitted number."""
    props = torch.cuda.get_device_properties("cuda")
    s, hd, h, b = (TINY["seq_len"], TINY["head_dim"], TINY["heads"], TINY["batch"])
    derived = ast.choose_tile(s, hd, props.regs_per_multiprocessor,
                              props.max_threads_per_multi_processor, props.warp_size)
    chosen, _ = ast.autotune_tile(s, hd, h, b, timer=ast.hot_time)

    qkv = torch.randn(b, s, 3 * h * hd, device="cuda", dtype=torch.float16)
    scale = hd ** -0.5
    ref = ast.sdpa_reference(qkv, h, hd)
    got = ast.single_tile_attention(qkv, h, hd, scale, *chosen)
    assert torch.allclose(got.float(), ref.float(), atol=ATOL, rtol=RTOL)

    t_der = ast.hot_time(lambda: ast.single_tile_attention(qkv, h, hd, scale, *derived))
    t_new = ast.hot_time(lambda: ast.single_tile_attention(qkv, h, hd, scale, *chosen))
    assert t_new < t_der * (1.0 - ast.DECISIVE), (
        f"{chosen} at {t_new*1e3:.3f} us did not clear {derived} at "
        f"{t_der*1e3:.3f} us by {ast.DECISIVE:.0%}")


def test_every_other_accepted_shape_keeps_its_tile():
    """THE BYTE-IDENTICAL-CONTROL CLAIM, tested rather than assumed -- and [L38]: it is
    what shows the tuner is capable of NOT changing its mind.

    Nine of the ten shapes the kernel accepts must select the identical tile under both
    timers. If this fails the candidate is not what it says it is, and the A/B's control
    configs are not controls.

    Config 3 is EXCLUDED and the exclusion is the evidence, not a concession: the g42
    probe measured the flushed timer picking `(16,4,1)` on it in one run and `(64,4,1)`
    in the next, off a one-quantum difference. An arm that cannot reproduce its own
    answer cannot be asserted equal to anything (finding 50). The hot timer picks
    `(64,4,1)` there in both runs, and that is asserted.
    """
    props = torch.cuda.get_device_properties("cuda")
    moved, unstable = [], BY_ID[3]
    for c in MATRIX:
        ok, _ = ast.applies(c.seq_len, c.head_dim, props)
        if not ok:
            continue
        shape = (c.seq_len, c.head_dim, c.heads, c.batch_size)
        hot, _ = ast.autotune_tile(*shape, timer=ast.hot_time)
        if (c.seq_len, c.head_dim, c.heads) == (unstable.seq_len, unstable.head_dim,
                                                unstable.heads) and \
                c.batch_size == unstable.batch_size:
            derived = ast.choose_tile(c.seq_len, c.head_dim,
                                      props.regs_per_multiprocessor,
                                      props.max_threads_per_multi_processor,
                                      props.warp_size)
            assert hot == derived, f"cfg {c.id}: hot timer moved off the derived tile"
            continue
        flushed, _ = ast.autotune_tile(*shape, timer=ast.flushed_time)
        if hot != flushed:
            moved.append((c.id, flushed, hot))
    assert len(moved) == 1, (
        f"expected exactly one shape to move (the B=1 row); got {moved}")
    assert moved[0][0] == 2


# --------------------------------------------------------------------- correctness
#
# The locked tolerance, on the shapes the mechanism touches and the ones it does not.
# Config 2 is the shape that changes tile; 1, 4 and 12 are controls that must not.

@pytest.mark.parametrize("config_id", [2, 1, 4, 12])
def test_matches_the_reference_at_the_locked_tolerance(config_id):
    m, base, cfg, ref = _built("v42_hot_tuned_tile", config_id)
    res = _run(m, base, cfg, ref)
    assert res.passed, (f"cfg {config_id}: max_abs {res.max_abs_error:.3e} "
                        f"(reason: {getattr(m, 'attn_reason', None)})")


def test_causality_is_exact_on_the_shape_whose_tile_changed():
    """Every announced row is causal and a masked entry carries exactly zero softmax
    weight, so a smaller `block_m` must not leak future tokens. Perturbing a token must
    leave every strictly earlier position bit-identical.
    """
    m, base, cfg, ref = _built("v42_hot_tuned_tile", 2)
    x, mask = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                       seed=99, padding_ratio=0.0, input_scale=1.0)
    cut = cfg.seq_len // 2
    x2 = x.clone()
    x2[:, cut:] += 1.0
    with torch.inference_mode():
        a = m(x, mask)
        b = m(x2, mask)
    assert torch.equal(a[:, :cut], b[:, :cut]), (
        "changing token t moved an output at a position < t: causality is not exact")


@pytest.mark.parametrize("padding_ratio", [0.0, 0.25])
def test_padding_does_not_break_the_smaller_tile(padding_ratio):
    m, base, cfg, ref = _built("v42_hot_tuned_tile", 2)
    res = _run(m, base, cfg, ref, padding_ratio=padding_ratio)
    assert res.passed, f"max_abs {res.max_abs_error:.3e}"


def test_the_model_actually_runs_the_retuned_tile():
    """End of the [L36] chain: the tuner changed its mind, and the MODEL is the thing
    that has to run the result. Assert the built model's plan, not the routine's.

    The plan is latched at PRIME TIME -- `_decide_attn` runs on the first forward, before
    compilation and graph capture -- so both arms are run once first. Asserting on a model
    that has never been called would read `undecided` on both and pass for the wrong
    reason on the day the wiring broke.
    """
    m, base_m, cfg_m, ref_m = _built("v42_hot_tuned_tile", 2)
    parent, base_p, cfg_p, ref_p = _built("v41_vendor_aware_attn", 2)
    assert _run(m, base_m, cfg_m, ref_m).passed
    assert _run(parent, base_p, cfg_p, ref_p).passed
    assert m.attn_used and m.attn_form == "single_tile", m.attn_reason
    assert parent.attn_used and parent.attn_form == "single_tile", parent.attn_reason
    assert m.attn_tile != parent.attn_tile, (
        f"both arms run {m.attn_tile}: the candidate is its parent\n"
        f"  v42: {m.attn_reason}\n  v41: {parent.attn_reason}")
    assert "hot_time" in m.attn_reason and "flushed_time" in parent.attn_reason
