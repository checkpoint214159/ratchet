"""v43: the tile sweep is replicated and reduced by its floor before it displaces.

Tolerance is the LOCKED one -- `atol 2e-3 OR rtol 2e-2`, elementwise OR. Not widened.
Causal stays exact; every config in the matrix is causal and `config.causal` is honoured
by construction (the kernel masks with `rn <= rm`, and these tests build from `BY_ID`).

WHAT THIS CANDIDATE CLAIMS, AND THEREFORE WHAT MUST BE ASSERTED
----------------------------------------------------------------
v42's claim was about a MEAN: a shape got faster. This one's is about a VARIANCE: the
tuner stops changing its mind. A test suite copied from v42's would be measuring the
wrong thing entirely -- every correctness and speed assertion in it could pass while the
selection rule stayed exactly as unstable as before.

So [L36] is applied to the property that is actually claimed. The mechanism is not "a
tile changed"; the mechanism is **the tuner selects the SAME tile every time it is asked,
on the shape whose margin is small enough that the parent moved three ways** -- and
**does not lose the shape the parent won**, whose margin is large. Both are asserted
directly, on hardware, in the model.

WHY IN THE MODEL AND NOT IN A STANDALONE SWEEP
------------------------------------------------
Finding 53 caught itself on exactly this: it validated a regime-mismatch fix with a probe
that had a regime mismatch. On B=4 the model's own prime-time sweep reported one tile
beating another **1.460x** where a standalone probe measured the same two arms at
**0.98x** -- same arms, same timer, 1.49x apart.

Generation 43 measured the same trap twice more, and both are why the on-hardware tests
below are written the way they are:

  * `bench/probes/g43_stable_tiles/prime_stability.py` in a ONE-ARM process reports both
    v42 and v43 selecting one plan in 8 of 8 on both shapes. The instability does not
    exist there. It appears only once a second model is built, primed and resident --
    which is what `bench/abba.py` does and therefore what every ranking of these two
    candidates is taken in. A one-arm probe would have concluded "nothing to fix".

  * A sweep run OUTSIDE `torch.inference_mode()` after any model has been run inside one
    makes `do_bench_cudagraph` raise, and `hot_time`'s bare `except` silently returns
    `do_bench`'s number instead -- so the whole grid comes back quantized to 1.024 us,
    the exact instrument v42 was built to remove, wearing v42's name. `_decide_attn` runs
    inside `inference_mode` at the real call site, so this is a probe hazard rather than a
    shipping one, but it is a live trap for anything that measures this tuner.

[L38] IS SERVED BY THE SCRIPTED-TIMER TESTS
---------------------------------------------
The decision rule is driven in every direction with synthetic timings -- a margin that
holds up, a challenger contaminated in one sweep, an incumbent contaminated in one sweep,
a margin inside `DECISIVE`, an arm that cannot be timed throughout -- so a pass from the
on-hardware tests means something. Two of them are paired with a `replicates=1` control
running the identical script and getting the OPPOSITE answer, which is what makes "the
replication is what did it" a measurement rather than a hope.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.kernels import attn_single_tile as ast
from bench.matrix import BY_ID

ATOL, RTOL = 2e-3, 2e-2

# The two shapes this candidate is about, named by their SHAPE and not by their id, the
# way the dispatch predicates are. Config 2 is the only announced row whose batch is 1 --
# the shape v42 won, whose true margin is 28%, and which v43 must not lose. Config 3 is
# B=4, whose two leading tiles are ~2% apart, which is why the parent's sweep moved three
# ways across six runs. Written out so these tests still mean something if the matrix
# moves, and so no config id drives a decision here.
BIG_MARGIN = dict(batch=1, seq_len=128, heads=4, head_dim=32)
SMALL_MARGIN = dict(batch=4, seq_len=128, heads=4, head_dim=32)


def _ref_module():
    p = (Path(__file__).resolve().parents[2]
         / "benchmarks/reference/torch_transformer_benchmark.py")
    spec = importlib.util.spec_from_file_location("ref_v43", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v43"] = m
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


def _case(cfg, ref, seed=4321):
    return ref.generate_random_case(cfg, torch.device("cuda"), torch.float32,
                                    seed=seed, padding_ratio=0.0, input_scale=1.0)


def _reprime(model, x, n):
    """Ask the model's OWN tuner, `n` times, with the model resident.

    This is prime time reproduced, not simulated: `_decide_attn` is the routine the model
    runs once before compilation and graph capture, and re-running it on a model that is
    already built and on the device puts it in the allocator and cache state finding 53
    measured a 1.49x regime gap against a standalone probe.

    INSIDE `torch.inference_mode()`, WHICH IS NOT A DETAIL. That is where the real call
    site runs -- `bench/abba.py` primes every arm inside one, and so does the graded
    harness. OUTSIDE it, once any model has been run under inference mode,
    `do_bench_cudagraph` raises `Inplace update to inference tensor outside InferenceMode`
    and `hot_time`'s bare `except` silently returns `do_bench`'s number instead: the whole
    grid comes back quantized to the 1.024 us event tick, which is the exact instrument
    generation 42 removed. The first draft of this test omitted the context manager and
    duly reported the PARENT as unstable on the shape it wins 10 of 10 on.
    """
    picks = []
    with torch.inference_mode():
        for _ in range(n):
            model.attn_reason = "undecided"
            model._decide_attn(x)
            picks.append((model.attn_form, tuple(model.attn_tile or ())))
    return picks


# --------------------------------------------------------------- the wiring is real
#
# The diff is one class attribute, the easiest possible thing to leave unwired. Seven
# candidates this session were silently inert. Check the wire before anything else.

def test_the_candidate_replicates_and_the_parent_does_not():
    parent = REGISTRY["v42_hot_tuned_tile"].build(object)
    child = REGISTRY["v43_replicated_tile"].build(object)
    assert parent.attn_tile_replicates == 1, (
        "the parent must keep `autotune_tile`'s own default -- a control arm that has "
        "silently started replicating measures nothing")
    assert child.attn_tile_replicates == 2


def test_the_timer_is_inherited_unchanged():
    """v43 changes the STABILITY of v42's decision, not its instrument. If the timer
    moved too, nothing below would be attributable to replication."""
    parent = REGISTRY["v42_hot_tuned_tile"].build(object)
    child = REGISTRY["v43_replicated_tile"].build(object)
    assert child.attn_tile_timer is parent.attn_tile_timer is ast.hot_time


def test_the_replicate_default_is_the_parents():
    """`replicates=1` must mean exactly what v23 through v42 shipped."""
    import inspect
    assert inspect.signature(ast.autotune_tile).parameters["replicates"].default == 1


# --------------------------------------------------- the decision rule, all ways [L38]

@pytest.fixture
def scripted_timer():
    """Drive `autotune_tile` with a queue of milliseconds, consumed in sweep order.

    The kernels are still compiled, launched and correctness-checked on real hardware;
    only the numbers they are ranked on are substituted. That is what makes the cases
    below assertions about the RULE rather than about the card.
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


def _derived(seq_len, head_dim):
    props = torch.cuda.get_device_properties("cuda")
    return ast.choose_tile(seq_len, head_dim, props.regs_per_multiprocessor,
                           props.max_threads_per_multi_processor, props.warp_size)


def test_a_margin_that_replicates_still_displaces(scripted_timer):
    """A challenger that is fastest in BOTH sweeps must still displace the derived tile.

    This is the assertion that v42's win is not collateral damage of the stabilisation.
    """
    tiles = _grid(BIG_MARGIN["seq_len"], BIG_MARGIN["head_dim"])
    assert len(tiles) > 1
    one = [0.001] + [0.010] * (len(tiles) - 1)
    tile, why = ast.autotune_tile(
        BIG_MARGIN["seq_len"], BIG_MARGIN["head_dim"], BIG_MARGIN["heads"],
        BIG_MARGIN["batch"], timer=scripted_timer(one + one), replicates=2)
    assert tile == tiles[0], why
    assert "decisively" in why and "1 distinct per-sweep winners" in why, why


def test_a_challenger_contaminated_in_one_sweep_still_displaces(scripted_timer):
    """THE CASE THAT KILLED THE PRE-REGISTERED VOTE RULE, in synthetic form.

    The true winner reads 1.0 us in one sweep and 20.0 in the other; the derived tile
    reads 10.0 in both. Contamination on this harness is ONE-SIDED, so the challenger's
    floor is the statistic about the code and its slow reading is a statistic about the
    machine. A rule that voted per sweep would split 1-1 and hold the derived tile --
    which is exactly how the vote rule lost B=1 in 5 of 10 fresh processes.
    """
    s, hd = BIG_MARGIN["seq_len"], BIG_MARGIN["head_dim"]
    tiles = _grid(s, hd)
    fast = [0.001 if t == tiles[0] else 0.010 for t in tiles]
    slow = [0.020 if t == tiles[0] else 0.010 for t in tiles]
    tile, why = ast.autotune_tile(s, hd, BIG_MARGIN["heads"], BIG_MARGIN["batch"],
                                  timer=scripted_timer(fast + slow), replicates=2)
    assert tile == tiles[0], why
    assert "2 distinct per-sweep winners" in why, why


def test_a_margin_carried_by_a_contaminated_incumbent_does_not_displace(scripted_timer):
    """THE FAILURE THIS CANDIDATE EXISTS TO FIX, in the form the grids measured it in.

    The derived tile is genuinely level with the challenger (10.0 against 9.9, 1.01x),
    and in ONE of the two sweeps it reads 40.0 -- the incumbent contaminated, which is
    what `sweep_grids.py` measured at B=4 (3.752 and 4.245 against a 2.517 floor). The
    parent's single sweep would see a 4.04x margin there and displace. Reduced by the
    floor, the incumbent is back at 10.0 and nothing clears the bar.
    """
    s, hd = SMALL_MARGIN["seq_len"], SMALL_MARGIN["head_dim"]
    tiles = _grid(s, hd)
    derived = _derived(s, hd)
    clean = [0.010 if t == derived else 0.0099 for t in tiles]
    hit = [0.040 if t == derived else 0.0099 for t in tiles]
    tile, why = ast.autotune_tile(s, hd, SMALL_MARGIN["heads"], SMALL_MARGIN["batch"],
                                  timer=scripted_timer(clean + hit), replicates=2)
    assert tile == derived, why
    assert "inside" in why, why


def test_one_sweep_on_the_same_script_displaces_instead(scripted_timer):
    """THE CONTROL FOR THE TEST ABOVE, and what makes it a measurement rather than a
    hope. Identical FIRST sweep, identical rule, `replicates=1` -- and the contaminated
    incumbent is never seen, so the parent's rule displaces on a 1.01x tile dressed up
    as 4.04x. The difference between the two answers is the whole candidate.
    """
    s, hd = SMALL_MARGIN["seq_len"], SMALL_MARGIN["head_dim"]
    tiles = _grid(s, hd)
    derived = _derived(s, hd)
    hit = [0.040 if t == derived else 0.0099 for t in tiles]
    tile, why = ast.autotune_tile(s, hd, SMALL_MARGIN["heads"], SMALL_MARGIN["batch"],
                                  timer=scripted_timer(hit), replicates=1)
    assert tile != derived, why
    assert "decisively" in why, why


def test_agreement_inside_the_decisive_margin_does_not_displace(scripted_timer):
    """Replication reduces the estimator; it does not replace `DECISIVE`.

    Both sweeps put the challenger 1.053x ahead -- inside the inherited 10% bar. The
    incumbent still holds, exactly as it did under the parent.
    """
    s, hd = BIG_MARGIN["seq_len"], BIG_MARGIN["head_dim"]
    tiles = _grid(s, hd)
    derived = _derived(s, hd)
    one = [0.0095 if t != derived else 0.010 for t in tiles]
    tile, why = ast.autotune_tile(s, hd, BIG_MARGIN["heads"], BIG_MARGIN["batch"],
                                  timer=scripted_timer(one + one), replicates=2)
    assert tile == derived, why
    assert "inside" in why, why


def test_every_arm_gets_the_same_trial_budget_in_every_sweep(scripted_timer):
    """Equal budget per arm, per sweep, and the floor taken over the same count for
    each. An arm reduced over fewer trials than its rivals is finding 47's
    best-of-N-against-best-of-1 handicap, inverted."""
    s, hd = BIG_MARGIN["seq_len"], BIG_MARGIN["head_dim"]
    tiles = _grid(s, hd)
    calls = []

    def counting(fn, reps):
        calls.append(reps)
        return 0.005
    ast.autotune_tile(s, hd, BIG_MARGIN["heads"], BIG_MARGIN["batch"],
                      timer=counting, replicates=2)
    assert len(calls) == 2 * len(tiles), (
        f"{len(calls)} timings for {len(tiles)} tiles over 2 sweeps")
    assert len(set(calls)) == 1, "arms were timed with different repeat counts"


def test_an_arm_that_could_not_be_timed_in_every_sweep_is_dropped(scripted_timer):
    """An arm present in one sweep and not the other must not be reduced over fewer
    readings than its rivals -- it is dropped instead. Here the first tile raises in the
    second sweep, and its otherwise-winning 1.0 us reading must not select it."""
    s, hd = BIG_MARGIN["seq_len"], BIG_MARGIN["head_dim"]
    tiles = _grid(s, hd)
    seen = []

    def flaky(fn, reps):
        seen.append(1)
        if len(seen) == 1:
            return 0.001                      # the first arm, first sweep: fastest
        if len(seen) == len(tiles) + 1:
            raise RuntimeError("this arm could not be timed in the second sweep")
        return 0.010
    tile, why = ast.autotune_tile(s, hd, BIG_MARGIN["heads"], BIG_MARGIN["batch"],
                                  timer=flaky, replicates=2)
    assert tile != tiles[0], why


def test_a_wrong_arm_is_still_dropped_before_it_is_timed(monkeypatch):
    """Correctness before timing survives the refactor into sweeps, and the drop is
    reported once rather than once per replicate."""
    s, hd = BIG_MARGIN["seq_len"], BIG_MARGIN["head_dim"]
    real = ast.single_tile_attention
    poisoned = _grid(s, hd)[0]

    def wrong(qkv, heads, head_dim, scale, bm, w, st=1):
        out = real(qkv, heads, head_dim, scale, bm, w, st)
        return (out + 1.0) if (bm, w, st) == poisoned else out
    monkeypatch.setattr(ast, "single_tile_attention", wrong)

    timed = []

    def watch(fn, reps):
        timed.append(True)
        return 0.010
    tile, why = ast.autotune_tile(s, hd, BIG_MARGIN["heads"], BIG_MARGIN["batch"],
                                  timer=watch, replicates=2)
    assert tile != poisoned, f"a wrong kernel won the sweep: {why}"
    assert "(1 dropped on tolerance)" in why, why


# ------------------------------------------------- the mechanism, on hardware, in model
#
# THE WHOLE CLAIM OF THIS CANDIDATE. Read the module docstring for why these are made by
# re-priming a resident model rather than by calling `autotune_tile` on an idle card.

def test_the_small_margin_shape_selects_one_tile_across_repeated_priming():
    """The parent picked three different tiles in six sweeps here. v43 must pick one.

    The shape is passed in by its dimensions; nothing names config 3. What is asserted is
    a property of the TUNER -- that its output is a function of the shape and the device
    rather than of when it happened to run. The parent is primed alongside as an in-test
    reference and REPORTED rather than asserted: its instability is a property of the
    machine on the day, and a test that required the parent to misbehave would fail for
    the wrong reason on a quiet card.
    """
    child, _, cfg, ref = _built("v43_replicated_tile", 3)
    x, mask = _case(cfg, ref)
    with torch.inference_mode():
        child(x, mask)                               # prime, compile, capture
    picks = _reprime(child, x, 8)
    assert len(set(picks)) == 1, (
        f"the tuner selected {len(set(picks))} different plans across 8 primings of the "
        f"same resident model: {picks}\n  reason: {child.attn_reason}")


def test_the_big_margin_shape_keeps_the_tile_its_parent_won_with():
    """v43 must not cost v42 its win, and must not wobble on it either.

    THE PRE-REGISTERED VOTE RULE FAILED EXACTLY HERE, which is why this test exists in
    this form. Requiring two sweeps to agree on a winner lost this shape in 5 of 10 fresh
    processes -- always the 5 in which the tuner primed second -- because the one-sided
    contamination lands on whichever arm it lands on and a vote is then decided by that.
    The floor reduction is what fixed it, so the assertion is BOTH that the plan is stable
    AND that it is not the derived tile: a rule that reverted to the derived tile every
    time would pass a stability test and throw the whole of generation 42 away.
    """
    child, _, cfg, ref = _built("v43_replicated_tile", 2)
    x, mask = _case(cfg, ref)
    with torch.inference_mode():
        child(x, mask)
    picks = _reprime(child, x, 8)
    assert len(set(picks)) == 1, (
        f"the tuner selected {len(set(picks))} different plans across 8 primings: "
        f"{picks}\n  reason: {child.attn_reason}")
    form, tile = picks[0]
    assert form == "single_tile", picks
    derived = _derived(cfg.seq_len, cfg.d_model // cfg.num_heads)
    assert tile != derived, (
        f"the replicated sweep fell back to the derived tile {derived}; v42's win on "
        f"this shape is gone\n  reason: {child.attn_reason}")


# --------------------------------------------------------------------- correctness
#
# The locked tolerance, on the shape whose plan is stabilised and on the ones that carry
# the score. Config 3 is the shape this candidate is about; 2, 1 and 12 are controls.

@pytest.mark.parametrize("config_id", [3, 2, 1, 12])
def test_matches_the_reference_at_the_locked_tolerance(config_id):
    m, base, cfg, ref = _built("v43_replicated_tile", config_id)
    x, mask = _case(cfg, ref)
    with torch.inference_mode():
        res = ref.compare_outputs(base(x, mask), m(x, mask), rtol=RTOL, atol=ATOL)
    assert res.passed, (f"cfg {config_id}: max_abs {res.max_abs_error:.3e} "
                        f"(reason: {getattr(m, 'attn_reason', None)})")


def test_causality_is_exact_on_the_stabilised_shape():
    """Every announced row is causal and a masked entry carries exactly zero softmax
    weight. Perturbing a token must leave every strictly earlier position bit-identical,
    whichever tile the replicated sweep settled on.
    """
    m, _, cfg, ref = _built("v43_replicated_tile", 3)
    x, mask = _case(cfg, ref, seed=99)
    cut = cfg.seq_len // 2
    x2 = x.clone()
    x2[:, cut:] += 1.0
    with torch.inference_mode():
        a = m(x, mask)
        b = m(x2, mask)
    assert torch.equal(a[:, :cut], b[:, :cut]), (
        "a future token changed an earlier output: causality is not exact")
