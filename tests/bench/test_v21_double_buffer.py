"""v21: the graph's output copy, removed without reintroducing finding 24.

Three groups of tests, and the middle group is the one that matters most:

  * the DEFECT -- the parent copies its whole output on every call, whether or not
    anybody wanted a copy. If a future parent stops doing that, v21's rationale is gone
    and this file says so loudly instead of leaving dead machinery in the frontier.
  * the SAFETY -- a candidate that returns a static buffer is the exact shape of
    finding 24. These tests hold the returned tensor deeper than the buffer count, alias
    it, and check the guard is capable of FIRING (L38), not merely of passing.
  * the MECHANISM -- L36: a test whose subject was never built is worse than no test.
    Every test here asserts the graph was captured and zero-copy is actually engaged
    before it asserts anything about the result.
"""
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.candidates import REGISTRY

NAME = "v21_double_buffered"
PARENT = "v18_capture_insurance"

ATOL, RTOL = 2e-3, 2e-2          # LOCKED. Never widen these (CLAUDE.md rule 1).


def _ref():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_v21", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_v21"] = m
    spec.loader.exec_module(m)
    return m


def _cfg(m):
    return m.TransformerConfig(batch_size=8, seq_len=64, d_model=128, num_heads=4,
                               ffn_dim=128, num_layers=2, causal=True)


def _build(name, with_baseline=False):
    # Dynamo's cache_size_limit is 8 and shared per process. Without this reset a later
    # test silently runs EAGER, which allocates a fresh tensor every call and therefore
    # passes every static-buffer assertion vacuously -- finding 24's exact failure.
    torch._dynamo.reset()
    m = _ref()
    cfg = _cfg(m)
    torch.manual_seed(0)
    base = m.BaselineTransformer(cfg)
    cand = REGISTRY[name].build(m.BaselineTransformer)(cfg)
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
    assert cand.buffering in ("single", "double"), cand.buffering_reason
    assert cand._zero_copy is True, cand.buffering_reason


# ------------------------------------------------------------------ the DEFECT

@pytest.mark.gpu
def test_the_parent_copies_its_entire_output_on_every_call():
    """PINS THE DEFECT v21 exists to remove.

    v13-v18 end `forward` with `return self._static_y.clone()`. That clone is a full
    device-to-device copy of the output, paid unconditionally, and a profile of the
    frontier at config 6's shape bills `Memcpy DtoD` at 6.6% of forward -- roughly half
    of it this copy. The evidence that it happens at all is that the returned tensor
    never shares storage with the static buffer the graph writes.

    If a future parent stops copying, this test fails and v21 is either obsolete or
    dangerous -- either way somebody has to look.
    """
    m, parent = _build(PARENT)
    x = _x()
    with torch.inference_mode():
        parent(x, None)                       # first call captures
        assert parent.graph_verified is True, "parent never captured; nothing is pinned"
        for _ in range(4):
            out = parent(x, None)
            assert out.data_ptr() != parent._static_y.data_ptr(), (
                "the parent no longer copies its output -- v21's whole premise is gone")
            assert out.untyped_storage().data_ptr() != \
                parent._static_y.untyped_storage().data_ptr()


@pytest.mark.gpu
def test_v21_does_not_copy_on_the_pattern_both_harnesses_use():
    """THE FIX. Both timing loops call `model(x, mask)` as a bare statement and discard
    the result before the next call (reference benchmark lines 494 and 504;
    bench/run_matrix.py `median_ms`). Nothing is holding the previous output, so nothing
    has to be preserved, so no copy is made.

    `output_copies` is the counter that keeps this from rotting: reinstate the clone and
    it stops being zero.
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
        f"seeing a reference that does not exist (a local binding of the handout inside "
        f"the check itself will do this)")
    assert cand.zero_copy_returns >= 10
    assert cand.preserve_rebinds == 0


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
        ptrs = {b.data_ptr() for _g, b in cand._bufs}
    assert out.data_ptr() in ptrs, "v21 returned something that is not a static buffer"


# ------------------------------------------------------------------ the SAFETY

@pytest.mark.gpu
@pytest.mark.parametrize("depth", [1, 2, 3, 6])
def test_a_held_output_survives_more_calls_than_there_are_buffers(depth):
    """Finding 24, generalised past the depth the shared sweep checks.

    `tests/bench/test_lineage_invariants.py` holds the result across exactly ONE later
    call. With two buffers that passes by rotation alone, which is precisely the accident
    L24 warns about -- so this goes deeper than the rotation can cover and forces the
    liveness check to do the work.
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
    if depth >= len(cand._bufs):
        assert cand.preserve_rebinds >= 1, (
            "nothing was preserved, so the invariant held by rotation alone -- this test "
            "is not exercising the guard it was written for")


@pytest.mark.gpu
def test_a_view_of_the_output_is_not_corrupted_either():
    """The case the rebind cannot fix, and therefore the case the candidate must refuse.

    `Tensor.set_` repoints the tensor we handed out; it cannot repoint a view the caller
    made of it. So when an un-rebindable alias exists, the buffer is retired instead of
    overwritten and the call is served from the compiled callable. Slower and correct.
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
    assert cand.retired_buffers >= 1, "the guard never fired; the test proves nothing"
    assert cand._zero_copy is False
    assert "retired" in cand.buffering_reason


@pytest.mark.gpu
def test_two_outputs_held_at_once_are_two_different_answers():
    """L23/L25 combined: staleness and clobbering in one assertion. Two different inputs,
    both results held simultaneously, must differ from each other AND must each still
    hold their own answer."""
    m, cand, base = _build(NAME, with_baseline=True)
    x1, x2 = _x(), _x()
    with torch.inference_mode():
        cand(x1, None)
        _engaged(cand)
        y1 = cand(x1, None)
        s1 = y1.clone()
        y2 = cand(x2, None)
        s2 = y2.clone()
        y3 = cand(x1, None)                   # a third call, past the buffer count
        assert torch.equal(y1, s1), "y1 was clobbered while the caller held it"
        assert torch.equal(y2, s2), "y2 was clobbered while the caller held it"
        assert not torch.allclose(y1, y2), "different inputs gave identical output"
        assert torch.allclose(y1, y3, rtol=1e-3, atol=1e-4)


@pytest.mark.gpu
def test_equivalence_against_the_reference_at_the_locked_tolerance():
    """The tolerance is locked at atol 2e-3 / rtol 2e-2, OR criterion. v21 changes no
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
def test_numerics_are_bit_identical_to_the_parent():
    """v21 replays the same compiled callable over the same buffers. It may not move one
    bit of the answer; if it does, something other than the copy changed."""
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
    assert torch.equal(a, b), "v21 changed the answer; it was supposed to change a copy"


# --------------------------------------------------------------- the MECHANISM

@pytest.mark.gpu
def test_the_two_buffers_are_actually_two_buffers():
    """If the allocator reused the first output address, double buffering is a costume
    and the safety margin it claims does not exist."""
    m, cand = _build(NAME)
    with torch.inference_mode():
        cand(_x(), None)
    _engaged(cand)
    if cand.buffering != "double":
        pytest.skip(f"declined a second buffer: {cand.buffering_reason}")
    ptrs = [b.data_ptr() for _g, b in cand._bufs]
    assert len(set(ptrs)) == len(ptrs), "two graphExecs share one output address"


@pytest.mark.gpu
def test_the_pool_overlap_guard_can_fire():
    """L38: a guard is only evidence if it is capable of firing.

    Capturing the second graph into the FIRST graph's memory pool is the cheap version of
    this candidate -- it would duplicate only the output buffer instead of a whole working
    set. It was tried, and MEASURED to be wrong on this workload: the allocator gives
    capture 1's output an address capture 0 uses for an intermediate, so replaying graph 0
    destroys graph 1's result. This test re-creates that exact configuration and asserts
    `_verify_pair` rejects it, so the guard is never trusted on the quiet case alone.
    """
    m, cand = _build(NAME)
    with torch.inference_mode():
        cand(_x(), None)
    _engaged(cand)

    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(g, pool=cand._graph.pool()):
            y = cand._compiled_core(cand._static_x, cand._static_m)
        ok, why = cand._verify_pair(g, y)
    assert ok is False, (
        "sharing the graph pool no longer aliases the output buffer. That would make the "
        "cheap version of this candidate viable -- go take it, and delete this test.")
    assert "shared pool" in why or "no-op" in why, why


@pytest.mark.gpu
def test_the_memory_gate_declines_and_the_candidate_still_works():
    """The gate is a ratio of two measured byte counts, so it will decline on the largest
    shapes -- which are the shapes where the copy is worth the most. That must cost
    correctness nothing and the zero-copy win nothing: with one buffer the liveness check
    alone still covers the discard pattern both harnesses use.
    """
    torch._dynamo.reset()
    m = _ref()
    cls = REGISTRY[NAME].build(m.BaselineTransformer)

    class Tight(cls):
        MEM_HEADROOM = 1e9            # nothing will ever satisfy this

    cfg = _cfg(m)
    torch.manual_seed(0)
    base = m.BaselineTransformer(cfg)
    cand = Tight(cfg)
    m.copy_model_weights(base, cand)
    cand, base = cand.cuda().eval(), base.cuda().eval()
    x = _x()
    with torch.inference_mode():
        cand(x, None)
        assert cand.buffering == "single", cand.buffering_reason
        assert "free" in cand.buffering_reason and "reserved" in cand.buffering_reason
        _engaged(cand)
        for _ in range(6):
            cand(x, None)
        assert cand.output_copies == 0, "one buffer must still be copy-free when discarded"
        want, got = base(x), cand(x, None)
    d = (got.float() - want.float()).abs()
    assert ((d <= ATOL) | (d <= RTOL * want.float().abs())).all()


@pytest.mark.gpu
def test_the_degradation_is_reported_not_silent():
    """L36 again: a fallback nobody can observe is one nobody will notice. `buffering`
    and `buffering_reason` are what make it visible in the ledger notes."""
    torch._dynamo.reset()
    m = _ref()
    cand = REGISTRY[NAME].build(m.BaselineTransformer)(_cfg(m)).cuda().eval()
    assert cand.buffering == "none" and cand.buffering_reason == "not built"
    with torch.inference_mode():
        cand(_x(), None)
    assert cand.buffering in ("none", "single", "double")
    assert cand.buffering_reason != "not built"


@pytest.mark.gpu
def test_no_config_ids_or_shape_literals_in_the_predicate():
    """Rule 2. The memory gate must be a function of measured device state, not of the
    announced matrix (L28: a dispatch that does not respond to the device is a hardcoded
    table wearing a costume)."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2]
           / "bench/candidates/v21_double_buffered.py").read_text()
    code = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
    code = code.split('"""', 2)[-1]           # drop the module docstring
    for literal in ("10000", "655", "config 6", "config_id"):
        assert literal not in code, f"{literal!r} appears in v21's executable body"
    assert "mem_get_info" in code and "memory_reserved" in code
