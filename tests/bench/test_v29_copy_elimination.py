"""v29: the graph's output copy, removed without reintroducing finding 24.

Ported from `tests/bench/test_v21_double_buffer.py` on `cand/g21/double-buffered` and
extended. Five groups, and the middle three are the ones that matter:

  * the DEFECT -- the parent copies its whole output on every call, whether or not anybody
    wanted a copy. If a future parent stops, v29's rationale is gone and this file says so
    loudly instead of leaving dead machinery in the frontier.
  * the SAFETY -- a candidate that returns a static buffer is the exact shape of
    finding 24. These tests hold the returned tensor across later calls, alias it, keep
    only its STORAGE, and check the guard is capable of FIRING (L38), not merely of
    passing. v29 has ONE buffer, so nothing here can pass by rotation.
  * the SENSOR -- the alias check is calibrated at runtime rather than hardcoded, so its
    semantics are pinned here. If `_storage_use_count` ever stops responding, the
    candidate must REFUSE, not silently clobber.
  * the RECOVERY -- v29's improvement over g21: an alias event costs the parent's clone,
    not the compiled fallback, once the alias is released.
  * the MECHANISM -- L36: a test whose subject was never built is worse than no test.
    Every test asserts the graph was captured and zero-copy is engaged before it asserts
    anything about the result.

Plus causality (finding 32): v29 overrides `forward`, so it would bypass v26's own check.
"""
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY
from bench.candidates.v29_copy_elimination import _storage_use_count

NAME = "v29_copy_elimination"
PARENT = "v26_causal_correct"

ATOL, RTOL = 2e-3, 2e-2          # LOCKED. Never widen these (CLAUDE.md rule 1).


def _ref():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v29", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v29"] = m
    spec.loader.exec_module(m)
    return m


def _cfg(m, causal=True):
    return m.TransformerConfig(batch_size=8, seq_len=64, d_model=128, num_heads=4,
                               ffn_dim=128, num_layers=2, causal=causal)


def _build(name, with_baseline=False, cls=None, causal=True):
    # Dynamo's cache_size_limit is 8 and shared per process. Without this reset a later
    # test silently runs EAGER, which allocates a fresh tensor every call and therefore
    # passes every static-buffer assertion vacuously -- finding 24's exact failure.
    torch._dynamo.reset()
    m = _ref()
    cfg = _cfg(m, causal)
    torch.manual_seed(0)
    base = m.BaselineTransformer(cfg)
    build_cls = cls if cls is not None else REGISTRY[name].build(m.BaselineTransformer)
    cand = build_cls(cfg)
    m.copy_model_weights(base, cand)
    cand = cand.cuda().eval()
    if with_baseline:
        return m, cand, base.cuda().eval()
    return m, cand


def _x():
    return torch.randn(8, 64, 128, device="cuda")


def _engaged(cand):
    """L36. Refuse to draw a conclusion from a candidate whose mechanism never ran."""
    assert cand.graph_verified is True, (
        f"no CUDA graph (capture_source={cand.capture_source}); every assertion below "
        f"would pass vacuously against the compiled fallback")
    assert cand.zero_copy == "on", cand.zero_copy_reason


# ------------------------------------------------------------------- the DEFECT

@pytest.mark.gpu
def test_the_parent_copies_its_entire_output_on_every_call():
    """PINS THE DEFECT v29 exists to remove.

    v13 through v26 end `forward` with `return self._static_y.clone()`. That clone is a
    full device-to-device copy of the output, paid unconditionally; a fresh profile of v26
    at config 6's shape bills `Memcpy DtoD` at 7.2% of forward, two calls per forward, and
    this is one of them. The evidence that it happens is that the returned tensor never
    shares storage with the static buffer the graph writes.

    If a future parent stops copying, this fails and v29 is either obsolete or dangerous.
    """
    m, parent = _build(PARENT)
    x = _x()
    with torch.inference_mode():
        parent(x, None)                       # first call primes, compiles and captures
        assert parent.graph_verified is True, "parent never captured; nothing is pinned"
        for _ in range(4):
            out = parent(x, None)
            assert out.data_ptr() != parent._static_y.data_ptr(), (
                "the parent no longer copies its output -- v29's whole premise is gone")
            assert out.untyped_storage().data_ptr() != \
                parent._static_y.untyped_storage().data_ptr()


@pytest.mark.gpu
def test_no_copy_on_the_pattern_both_harnesses_use():
    """THE FIX. Both timing loops call `model(x, mask)` as a bare statement and discard the
    result before the next call (reference benchmark lines 494 and 504;
    `bench/run_matrix.py` median_ms lines 90 and 96). Nothing holds the previous output, so
    nothing has to be preserved, so no copy is made.

    `output_copies` is the counter that keeps this from rotting: reinstate the clone and it
    stops being zero. It would also have caught g21's own liveness bug, where binding the
    handout to a local before `sys.getrefcount` made the count read one too high and every
    call took the copy path -- a candidate that measures as its own parent, silently.
    """
    m, cand = _build(NAME)
    x = _x()
    with torch.inference_mode():
        cand(x, None)
        _engaged(cand)
        before = cand.output_copies
        for _ in range(10):
            cand(x, None)                     # discarded, exactly like the harness
    assert cand.output_copies == before == 0, (
        f"{cand.output_copies} copies on the discard pattern -- the liveness check is "
        f"seeing a reference that does not exist")
    assert cand.zero_copy_returns >= 10
    assert cand.preserve_rebinds == 0
    assert cand.fallback_calls == 0


@pytest.mark.gpu
def test_the_returned_tensor_really_is_the_static_buffer():
    """Not merely 'no copy was counted' -- the tensor handed back must actually alias the
    buffer the graph wrote, or the counter is measuring nothing."""
    m, cand = _build(NAME)
    x = _x()
    with torch.inference_mode():
        cand(x, None)
        _engaged(cand)
        out = cand(x, None)
        assert out.data_ptr() == cand._static_y.data_ptr(), (
            "v29 returned something that is not the static buffer")


# ------------------------------------------------------------------- the SAFETY

@pytest.mark.gpu
@pytest.mark.parametrize("depth", [1, 2, 3, 6])
def test_a_held_output_survives_arbitrarily_many_later_calls(depth):
    """Finding 24, generalised past the depth the shared sweep checks.

    `tests/bench/test_lineage_invariants.py` holds the result across exactly ONE later
    call. v29 has a SINGLE buffer, so there is no rotation to hide behind at any depth --
    the liveness check is the only thing that can make this pass, and `preserve_rebinds`
    asserts it is what did.
    """
    m, cand = _build(NAME)
    x1, x2 = _x(), _x()
    with torch.inference_mode():
        cand(x1, None)
        _engaged(cand)
        held = cand(x1, None)
        snapshot = held.clone()
        for _ in range(depth):
            cand(x2, None)
        assert torch.equal(held, snapshot), (
            f"the tensor handed to the caller was mutated {depth} calls later")
    assert cand.preserve_rebinds >= 1, (
        "nothing was preserved, so the invariant held by accident -- this test is not "
        "exercising the guard it was written for")
    assert cand.alias_events == 0, "a plain held tensor must be rebound, not treated as an alias"


@pytest.mark.gpu
def test_a_view_of_the_output_is_not_corrupted_either():
    """The case the rebind cannot fix, and therefore the case the candidate must refuse.

    `Tensor.set_` repoints the tensor we handed out; it cannot repoint a view the caller
    made of it. So when an un-rebindable alias exists the buffer stops being handed out,
    and the call is served from the compiled callable. Slower and correct.
    """
    m, cand = _build(NAME)
    x1, x2 = _x(), _x()
    with torch.inference_mode():
        cand(x1, None)
        _engaged(cand)
        out = cand(x1, None)
        view = out[2:5]                       # an alias we cannot rebind
        snapshot = view.clone()
        for _ in range(6):
            cand(x2, None)
        assert torch.equal(view, snapshot), (
            "a view of the returned tensor was overwritten -- finding 24, again")
    assert cand.alias_events >= 1, "the guard never fired; the test proves nothing"
    assert cand.zero_copy == "clone"
    assert cand.fallback_calls >= 6, (
        "the buffer was replayed into while the caller's view still aliased it")


@pytest.mark.gpu
def test_a_caller_who_kept_only_the_storage_is_seen():
    """The hole g21 declared open. It is not.

    g21's finding says 'a caller retaining `untyped_storage()` rather than a tensor is
    invisible to the check'. MEASURED here: a held `UntypedStorage` raises the storage use
    count exactly like a view does, so the guard fires and the buffer is never clobbered.

    This test discriminates: if the sensor were blind to it, the handout's Python refcount
    would read 'free' after `del out` and the buffer WOULD be overwritten.
    """
    m, cand = _build(NAME)
    x1, x2 = _x(), _x()
    with torch.inference_mode():
        cand(x1, None)
        _engaged(cand)
        out = cand(x1, None)
        storage = out.untyped_storage()
        before = storage.data_ptr()
        del out                               # only the raw storage survives
        for _ in range(3):
            cand(x2, None)
    assert cand.alias_events >= 1, (
        "the candidate did not notice a caller holding the storage, and clobbered it")
    assert cand.zero_copy == "clone"
    assert storage.data_ptr() == before
    del storage


@pytest.mark.gpu
def test_two_outputs_held_at_once_are_two_different_answers():
    """L23/L25 combined: staleness and clobbering in one assertion. Two different inputs,
    both results held simultaneously, must differ from each other AND must each still hold
    their own answer."""
    m, cand = _build(NAME)
    x1, x2 = _x(), _x()
    with torch.inference_mode():
        cand(x1, None)
        _engaged(cand)
        y1 = cand(x1, None)
        s1 = y1.clone()
        y2 = cand(x2, None)
        s2 = y2.clone()
        y3 = cand(x1, None)                   # a third call, both earlier results alive
        assert torch.equal(y1, s1), "y1 was clobbered while the caller held it"
        assert torch.equal(y2, s2), "y2 was clobbered while the caller held it"
        assert not torch.allclose(y1, y2), "different inputs gave identical output"
        assert torch.allclose(y1, y3, rtol=1e-3, atol=1e-4)
    assert cand.preserve_rebinds >= 2


@pytest.mark.gpu
def test_equivalence_against_the_reference_at_the_locked_tolerance():
    """The tolerance is locked at atol 2e-3 / rtol 2e-2, OR criterion. v29 changes no
    arithmetic, so this must pass with the same margin as the parent."""
    m, cand, base = _build(NAME, with_baseline=True)
    with torch.inference_mode():
        for _ in range(4):
            x = _x()
            want = base(x)
            got = cand(x, None)
            _engaged(cand)
            d = (got.float() - want.float()).abs()
            assert ((d <= ATOL) | (d <= RTOL * want.float().abs())).all(), \
                f"max_abs {d.max():.3e}"


@pytest.mark.gpu
def test_numerics_are_bit_identical_to_the_parent(monkeypatch):
    """v29 replays the same compiled callable over the same buffer. It may not move one bit
    of the answer; if it does, something other than the copy changed.

    ONE CONFOUND, MEASURED, AND IT BELONGS TO v23 RATHER THAN TO EITHER ARM. v23 picks its
    attention tile by TIMING candidate tiles at prime time, so two instances built in one
    process legitimately disagree: observed (16, 4, 1) for the parent and (32, 4, 1) for
    the child on the same shape, max_abs 2.366e-04 apart -- three orders inside the
    tolerance, and nothing to do with the output copy. g21's version of this test could
    assert bit-identity outright because v18 has no autotuned kernel.

    So the tile is pinned to whatever the first arm measures, and both arms then run the
    same kernel. Anything left is v29's, and there must be nothing left.
    """
    import bench.candidates.v23_single_tile_attn as v23mod
    real_autotune = v23mod.autotune_tile
    memo = {}

    def once(*a, **k):
        if "tile" not in memo:
            memo["tile"] = real_autotune(*a, **k)
        return memo["tile"]

    monkeypatch.setattr(v23mod, "autotune_tile", once)

    torch._dynamo.reset()
    m = _ref()
    cfg = _cfg(m)
    torch.manual_seed(0)
    base = m.BaselineTransformer(cfg)
    parent = REGISTRY[PARENT].build(m.BaselineTransformer)(cfg)
    child = REGISTRY[NAME].build(m.BaselineTransformer)(cfg)
    m.copy_model_weights(base, parent)
    m.copy_model_weights(base, child)
    parent, child = parent.cuda().eval(), child.cuda().eval()
    x = _x()
    with torch.inference_mode():
        a = parent(x, None).clone()
        b = child(x, None).clone()
    assert parent.graph_verified and child.graph_verified
    assert parent.attn_tile == child.attn_tile, (
        f"the tile pin did not hold ({parent.attn_tile} vs {child.attn_tile}); the "
        f"comparison below would be measuring v23's autotuner, not v29")
    assert torch.equal(a, b), "v29 changed the answer; it was supposed to change a copy"


@pytest.mark.gpu
def test_the_safety_assertions_can_actually_fail():
    """L38, applied to this file itself: a guard is only evidence if it is capable of
    firing, and so is a test.

    Wedge `_verdict` to always answer 'free' -- which is precisely the naive
    delete-the-clone candidate, i.e. finding 24 -- and re-run the three assertions the
    three tests above rest on. ALL THREE must report corruption. If any of them stays
    green here, the corresponding test is passing for a reason other than the liveness
    check, and it is not protecting anything.
    """
    torch._dynamo.reset()
    m = _ref()
    cls = REGISTRY[NAME].build(m.BaselineTransformer)

    class Blind(cls):
        def _verdict(self):
            return "free"                     # never ask, always clobber

    _m, cand = _build(NAME, cls=Blind)
    x1, x2 = _x(), _x()
    with torch.inference_mode():
        cand(x1, None)
        assert cand.zero_copy == "on", cand.zero_copy_reason

        held = cand(x1, None)
        snap = held.clone()
        for _ in range(6):
            cand(x2, None)
        assert not torch.equal(held, snap), (
            "a blind candidate did NOT corrupt a held output -- "
            "test_a_held_output_survives_arbitrarily_many_later_calls proves nothing")

        out = cand(x1, None)
        view = out[2:5]
        vsnap = view.clone()
        for _ in range(6):
            cand(x2, None)
        assert not torch.equal(view, vsnap), (
            "a blind candidate did NOT corrupt a view -- "
            "test_a_view_of_the_output_is_not_corrupted_either proves nothing")

        out = cand(x1, None)
        storage = out.untyped_storage()
        osnap = out.clone()
        del out
        for _ in range(3):
            cand(x2, None)
        reread = torch.empty(0, dtype=osnap.dtype, device=osnap.device)
        reread.set_(storage, 0, osnap.shape)
        assert not torch.equal(reread, osnap), (
            "a blind candidate did NOT corrupt a retained storage -- "
            "test_a_caller_who_kept_only_the_storage_is_seen proves nothing")


# ------------------------------------------------------------------- the SENSOR

@pytest.mark.gpu
def test_the_storage_use_count_means_what_the_candidate_thinks_it_means():
    """Pins the sensor's semantics, measured rather than assumed. Everything v29 does rests
    on these four numbers; a torch upgrade that changes any of them must fail HERE, loudly,
    rather than in a silent clobber."""
    t = torch.randn(16, device="cuda")
    base = _storage_use_count(t)
    d = t.detach()
    assert _storage_use_count(t) == base + 1, "a second TensorImpl must be countable"
    v = d[2:5]
    assert _storage_use_count(t) == base + 2, "a view must be countable"
    del v
    assert _storage_use_count(t) == base + 1, "the count must fall when an alias dies"
    s = d.untyped_storage()
    del d
    assert _storage_use_count(t) > base, (
        "a held UntypedStorage is invisible -- v29's guard has a hole and the "
        "kept-only-the-storage test above is passing for the wrong reason")
    del s
    assert _storage_use_count(t) == base


@pytest.mark.gpu
def test_arming_records_the_calibration_it_measured():
    """g21 hardcoded the threshold at 2. v29 measures it, and says what it measured."""
    m, cand = _build(NAME)
    with torch.inference_mode():
        cand(_x(), None)
    _engaged(cand)
    assert cand._base_use == _storage_use_count(cand._static_y)
    assert "calibrated" in cand.zero_copy_reason
    assert "verified able to fire" in cand.zero_copy_reason


@pytest.mark.gpu
def test_a_sensor_that_cannot_fire_is_refused_and_the_parent_clone_stands(monkeypatch):
    """L38: a guard is only evidence if it is capable of firing, and v29 checks that at arm
    time against an alias it creates itself. Wedge the sensor so it cannot respond, and the
    candidate must decline zero-copy entirely and go on being the parent -- not assume the
    buffer is free forever, which is finding 24 with extra steps."""
    import bench.candidates.v29_copy_elimination as mod
    monkeypatch.setattr(mod, "_storage_use_count", lambda t: 1)

    m, cand, base = _build(NAME, with_baseline=True)
    x = _x()
    with torch.inference_mode():
        cand(x, None)
        assert cand.graph_verified is True, "no graph; the refusal path was never reached"
        assert cand.zero_copy == "refused", cand.zero_copy_reason
        assert "did not respond" in cand.zero_copy_reason
        held = cand(x, None)
        snapshot = held.clone()
        for _ in range(4):
            cand(x, None)
        assert torch.equal(held, snapshot), "a refused candidate must still be safe"
        want, got = base(x), cand(x, None)
    assert cand.output_copies >= 5, "a refused candidate must be paying the parent's clone"
    assert cand.zero_copy_returns == 0
    d = (got.float() - want.float()).abs()
    assert ((d <= ATOL) | (d <= RTOL * want.float().abs())).all()


# ----------------------------------------------------------------- the RECOVERY

@pytest.mark.gpu
def test_the_graph_comes_back_once_the_alias_is_released():
    """v29's improvement over g21, which retired the buffer FOREVER and served every later
    call from the compiled callable -- the whole +7.9% that owning the graph bought (L20).

    A caller who slices the output once should not cost the graph permanently. While the
    alias lives, the compiled callable serves; the moment it dies, the buffer is replayed
    into and cloned out, which is exactly what the parent does.
    """
    m, cand, base = _build(NAME, with_baseline=True)
    x1, x2 = _x(), _x()
    with torch.inference_mode():
        cand(x1, None)
        _engaged(cand)
        out = cand(x1, None)
        view = out[2:5]
        cand(x2, None)                        # trips the alias guard
        assert cand.zero_copy == "clone"
        fell_back = cand.fallback_calls
        assert fell_back >= 1
        cand(x2, None)
        assert cand.fallback_calls == fell_back + 1, "still aliased; must not replay"
        assert cand.output_copies == 0

        del view, out                         # the alias is gone
        copies = cand.output_copies
        for _ in range(3):
            got = cand(x1, None)
        assert cand.fallback_calls == fell_back + 1, (
            "the graph did not come back after the alias was released -- this is g21's "
            "permanent retirement, which v29 exists to avoid")
        assert cand.output_copies == copies + 3, "the recovered path must be replay+clone"
        want = base(x1)
    d = (got.float() - want.float()).abs()
    assert ((d <= ATOL) | (d <= RTOL * want.float().abs())).all()


# ------------------------------------------------------------------ the MECHANISM

@pytest.mark.gpu
def test_the_degradation_is_reported_not_silent():
    """L36 again: a fallback nobody can observe is one nobody will notice. `zero_copy` and
    `zero_copy_reason` are what make it visible in the ledger notes."""
    torch._dynamo.reset()
    m = _ref()
    cand = REGISTRY[NAME].build(m.BaselineTransformer)(_cfg(m)).cuda().eval()
    assert cand.zero_copy == "unbuilt" and cand.zero_copy_reason == "not built"
    with torch.inference_mode():
        cand(_x(), None)
    assert cand.zero_copy in ("unbuilt", "on", "clone", "refused")
    assert cand.zero_copy_reason != "not built"


@pytest.mark.gpu
def test_no_config_ids_or_shape_literals_in_the_executable_body():
    """Rule 2, and L28: a mechanism that does not respond to the device is a hardcoded
    table wearing a costume. v29's only runtime constant is measured from the buffer it is
    about to hand out."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "bench/candidates/v29_copy_elimination.py").read_text()
    code = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
    code = code.split('"""', 2)[-1]           # drop the module docstring
    for literal in ("config 6", "config_id", "seq_len ==", "d_model =="):
        assert literal not in code, f"{literal!r} appears in v29's executable body"
    assert "_base_use = n0" in code, "the alias threshold must be measured, not written down"


# ------------------------------------------------------------------- CAUSALITY

@pytest.mark.gpu
@pytest.mark.parametrize("causal", [True, False])
def test_causal_is_honoured_on_both_settings(causal):
    """Finding 32. v29 overrides `forward` on the graph path, so it bypasses v26's own
    check and must restate it. The reference benchmark's DEFAULT is causal=False."""
    m, cand, base = _build(NAME, with_baseline=True, causal=causal)
    x = _x()
    with torch.inference_mode():
        want, got = base(x), cand(x, None)
    assert cand.causal_path.startswith("optimized" if causal else "baseline"), \
        cand.causal_path
    d = (got.float() - want.float()).abs()
    ok = (d <= ATOL) | (d <= RTOL * want.float().abs())
    assert ok.all(), f"causal={causal}: {(~ok).sum().item()} failed, max_abs {d.max():.3e}"


@pytest.mark.gpu
def test_a_non_causal_config_never_arms_zero_copy():
    """Belt and braces: the delegation must happen BEFORE any buffer machinery, so a
    non-causal run cannot end up handing out a static buffer at all."""
    m, cand = _build(NAME, causal=False)
    x = _x()
    with torch.inference_mode():
        a = cand(x, None)
        b = cand(x, None)
        snapshot = a.clone()
        cand(_x(), None)
    assert cand.zero_copy == "unbuilt", cand.zero_copy_reason
    assert cand.zero_copy_returns == 0
    assert a.data_ptr() != b.data_ptr(), "the non-causal path returned a reused buffer"
    assert torch.equal(a, snapshot)
