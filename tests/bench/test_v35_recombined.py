"""v35: the recombination, and specifically THE PART NEITHER PARENT CONTAINS.

v33's tests already prove streaming and the shape fix; v34's already prove the kernel
count and the disjoint predicate. Both files still run here unchanged and are the
regression surface for the halves. What this file has to establish is the merge itself:

  1. the shape fix SURVIVES the extra layer (the two-shape probe, against the same
     control v33 used -- the parent must still exhibit the bug being fixed);
  2. the kernel reduction SURVIVES (counted, not assumed -- L36 -- and counted against
     a freshly reset Dynamo, because `cache_size_limit` is 8 and shared per process);
  3. the three predicates coexist without overlapping or contradicting;
  4. v34's five shape-latched attributes are RESET when the shape changes, which is the
     one thing composing the two candidates actually requires -- plus the mask-derived
     state, which v33's fix makes reachable for the first time.

Every GPU test is preceded by an assertion that the mechanism it is about engaged. A
merge test that quietly ran the parent's path asserts only that v33 equals v33.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.candidates import REGISTRY                                     # noqa: E402
from bench.matrix import MATRIX, BY_ID                                    # noqa: E402
from bench.kernels.ffn_fused import (MIN_TILE_ROWS, amortizes, fits,      # noqa: E402
                                     launch_tile, one_wave)
from bench.candidates.v14_dispatch import (RESIDENT_BUDGET, choose,       # noqa: E402
                                           estimate_working_set_bytes)

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# This card, from ledger/device.json. Used only to evaluate the predicates as pure
# functions; the candidate reads them off get_device_properties at run time.
SM_COUNT = 66
SMEM_PER_SM = 102400
SMEM_OPTIN = 101376

RTOL, ATOL = 0.02, 0.002          # the harness's own defaults. Never widened.


def _reference():
    path = REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_bench_v35", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench_v35"] = m
    spec.loader.exec_module(m)
    return m


# ======================================================================================
# Lineage, and the merge's own bookkeeping. No GPU.
# ======================================================================================

class TestLineage:
    def test_parent_is_v33_and_v34_is_a_registered_contributor(self):
        assert REGISTRY["v35_recombined"].parent == "v33_streamed_long"
        # The contributor has to EXIST as a candidate in its own right; a recombination
        # that absorbs a sibling without registering it destroys the sibling's clade.
        assert REGISTRY["v34_launch_bound"].parent == "v26_causal_correct"
        assert "v34_launch_bound" in REGISTRY["v35_recombined"].summary

    def test_the_merge_did_not_drop_a_registered_candidate(self):
        """[L14]. A regex resolution broke this registry twice in one session."""
        for name, spec in REGISTRY.items():
            assert spec.name == name, f"{name} keyed against spec named {spec.name}"
            assert spec.parent is None or spec.parent in REGISTRY, (name, spec.parent)
        assert {"v26_causal_correct", "v33_streamed_long", "v34_launch_bound",
                "v35_recombined"} <= set(REGISTRY)

    def test_neither_mechanism_was_retyped(self):
        """Both halves must be the SAME objects the siblings use, not copies.

        v33 makes this argument about v14's predicate already: two copies of a threshold
        drift apart, one copy cannot. The merge is where that stops being rhetorical.
        """
        from bench.candidates import v33_streamed_long as S
        from bench.candidates import v34_launch_bound as L
        from bench.candidates import v35_recombined as M
        assert M.build_streaming_on is S.build_on
        assert M.build_v34 is L.build
        assert M.DERIVED_WARPS == L.DERIVED_WARPS

    def test_v33s_own_entry_point_is_unchanged_by_the_factoring(self):
        """`build_on` was extracted so v35 could stack the streaming layer on a sibling
        of v26. v33 itself must still be v33: same layer, same base."""
        from bench.candidates import v33_streamed_long as S
        from bench.candidates.v26_causal_correct import build as b26
        ref = _reference()
        direct = S.build_on(b26(ref.BaselineTransformer))
        viabuild = S.build(ref.BaselineTransformer)
        assert [c.__name__ for c in viabuild.__mro__] == \
               [c.__name__ for c in direct.__mro__]


def _chain():
    ref = _reference()
    from bench.candidates.v26_causal_correct import build as b26
    from bench.candidates.v34_launch_bound import build as b34
    return (b26(ref.BaselineTransformer), b34(ref.BaselineTransformer),
            REGISTRY["v35_recombined"].build(ref.BaselineTransformer))


def _own_state(cls):
    """Non-callable, non-dunder attributes a class introduces in its own body."""
    return {k for k, v in vars(cls).items()
            if not k.startswith("_") and not callable(v)
            and not isinstance(v, (classmethod, staticmethod, property))}


class TestTheMergeCoversWhatItInherited:
    def test_the_layering_order_puts_streaming_above_the_fusion(self):
        _v26, _v34, v35 = _chain()
        names = [c.__name__ for c in v35.__mro__]
        assert names[:4] == ["CandidateV35", "CandidateV33", "CandidateV34",
                             "CandidateV26"], names
        # One copy of each layer. A diamond here would mean a mechanism was applied twice.
        assert len(names) == len(set(names)), names

    def test_the_reset_list_is_derived_from_v34_not_typed_out(self):
        """THE MERGE'S ONE LOAD-BEARING CLAIM.

        v33's `_invalidate_shape_state` enumerates the state latched to an input shape.
        v34 adds five more attributes to that set and v33 cannot know their names. This
        computes the difference from the classes, so a generation 36 that latches a sixth
        attribute fails here instead of silently running a stale plan.
        """
        v26, v34, v35 = _chain()
        added = _own_state(v34) - _own_state(v26)
        declared = set(v35.SHAPE_LATCHED_BY_V34)
        assert added == declared, (
            f"v34 introduces {sorted(added)} but v35's reset covers {sorted(declared)}. "
            f"Anything in the first set and not the second survives a shape change and "
            f"is then read against the wrong shape.")

    def test_every_declared_attribute_is_actually_reset(self):
        """Read the source of the override rather than trusting the tuple."""
        src = (REPO / "bench" / "candidates" / "v35_recombined.py").read_text()
        body = src.split("def _invalidate_shape_state", 1)[1].split(
            "def _settle_slice_decisions", 1)[0]
        _v26, _v34, v35 = _chain()
        for attr in v35.SHAPE_LATCHED_BY_V34:
            assert re.search(rf"self\.{attr}\s*=", body), f"{attr} is never reset"
        assert "self._prime(" in body, "the mask-derived state is never re-derived"

    def test_the_streamed_path_settles_the_launch_decision(self):
        """The streamed path never reaches v34's `forward`, so the hook has to."""
        src = (REPO / "bench" / "candidates" / "v35_recombined.py").read_text()
        hook = src.split("def _settle_slice_decisions", 1)[1]
        assert "_decide_launch" in hook


# ======================================================================================
# Three predicates, no GPU. A dispatch predicate that needs a GPU to check is a
# hardcoded table wearing a costume (L28).
# ======================================================================================

def _fused(cfg, sm=SM_COUNT, smem_sm=SMEM_PER_SM, smem_opt=SMEM_OPTIN, tokens=None):
    tokens = cfg.tokens if tokens is None else tokens
    bm = launch_tile(tokens, sm)
    if not fits(cfg.d_model, cfg.ffn_dim, 2, bm, smem_opt):
        return False
    if amortizes(tokens, cfg.d_model, cfg.ffn_dim, 2):
        return False
    return one_wave(tokens, cfg.d_model, cfg.ffn_dim, 2, bm, sm, smem_sm)


class TestThreePredicatesCoexist:
    """MEASURED, not assumed: the two occupancy predicates are NOT disjoint as functions.

    v34's docstring says the two sets are disjoint on the announced matrix, and the sets
    it SELECTS are. The raw predicates are not: config 7 (d_model = ffn_dim = 32) has
    weights so small that they amortize at 8192 tokens AND so small that eight blocks fit
    per SM, so the whole grid is one wave. Both are true of it.

    v34 resolves that by PRECEDENCE -- `_decide_launch` asks `amortizes` first and returns
    -- not by construction. That is a sound resolution and the sets come out disjoint, but
    it is an ordering rule inside one function rather than a property of the predicates,
    and the merge is the moment to write that down rather than inherit the stronger claim.
    """

    def test_the_predicates_overlap_on_exactly_one_announced_shape(self):
        both = set()
        for cfg in MATRIX:
            bm = launch_tile(cfg.tokens, SM_COUNT)
            if not fits(cfg.d_model, cfg.ffn_dim, 2, bm, SMEM_OPTIN):
                continue
            if (amortizes(cfg.tokens, cfg.d_model, cfg.ffn_dim, 2)
                    and one_wave(cfg.tokens, cfg.d_model, cfg.ffn_dim, 2, bm,
                                 SM_COUNT, SMEM_PER_SM)):
                both.add(cfg.id)
        assert both == {7}, (
            f"the overlap moved: {sorted(both)}. v34 resolves an overlap by giving "
            f"`amortizes` precedence; a NEW overlap has not been reasoned about.")

    def test_precedence_makes_the_decision_single_valued(self):
        """Whatever the overlap, exactly one regime may claim a shape at the end."""
        selected = {c.id for c in MATRIX if _fused(c)}
        assert selected == {2, 3, 4, 12}, selected
        for cfg in MATRIX:
            if cfg.id in selected:
                assert not amortizes(cfg.tokens, cfg.d_model, cfg.ffn_dim, 2), cfg.id
        assert 7 not in selected, "the overlapping shape must fall to `amortizes`"

    def test_the_third_predicate_is_on_a_different_axis_and_never_overrides(self):
        """`choose` asks about CAPACITY -- does the working set fit the memory the device
        reports free -- and the other two about OCCUPANCY. It does not select a kernel; it
        decides how many times `_core` is called, and each of those calls then asks the
        other two about its own SLICE.

        Thirteen announced shapes are resident, so their slice IS the whole input and the
        occupancy answer is unchanged by streaming existing at all. The fourteenth streams,
        and the point below is that its slice still gets an answer.
        """
        big = 15 * 2**30
        resident = [c.id for c in MATRIX
                    if choose(c.batch_size, c.seq_len, c.d_model, c.heads, c.layers,
                              4, big)[0] == "resident"]
        assert len(resident) == len(MATRIX) - 1 and 14 not in resident, resident
        for cid in resident:
            cfg = BY_ID[cid]
            assert _fused(cfg) == _fused(cfg, tokens=cfg.tokens)

    def test_the_streamed_config_declines_the_fusion_on_its_slice_too(self):
        """And it declines for the SAME reason at both shapes, which is the check that
        matters: streaming must not turn a shape the kernel cannot hold into one it
        thinks it can.

        Config 14 is d_model = ffn_dim = 1024, so both weight matrices are 4.25 MB of
        shared memory against 99 KB opt-in. `fits` refuses it whole and refuses it sliced.
        """
        cfg = BY_ID[14]
        per_row = estimate_working_set_bytes(1, cfg.seq_len, cfg.d_model, cfg.heads,
                                             cfg.layers, 4)
        free = 15 * 2**30
        path, _tuned = choose(cfg.batch_size, cfg.seq_len, cfg.d_model, cfg.heads,
                              cfg.layers, 4, free)
        assert path == "streamed", path
        stream_slice = max(1, min(cfg.batch_size,
                                  int(free * RESIDENT_BUDGET // per_row)))
        assert stream_slice < cfg.batch_size
        tokens = stream_slice * cfg.seq_len
        bm = launch_tile(tokens, SM_COUNT)
        assert not fits(cfg.d_model, cfg.ffn_dim, 2, bm, SMEM_OPTIN)
        assert not _fused(cfg, tokens=tokens)
        assert not _fused(cfg)

    def test_the_predicates_respond_to_the_device_not_to_the_matrix(self):
        """Same shapes, a different card: the answer must move."""
        cfg = BY_ID[2]
        assert _fused(cfg, sm=SM_COUNT)
        # A machine with one SM cannot hold config 2's grid in one wave.
        assert not _fused(cfg, sm=1, smem_sm=SMEM_PER_SM)

    def test_no_config_ids_or_announced_shapes_in_the_candidate_source(self):
        """CLAUDE.md rule 2. The merge adds no new constant; this proves it."""
        src = (REPO / "bench" / "candidates" / "v35_recombined.py").read_text()
        code = src.split('"""', 2)[-1]                    # drop the module docstring
        code = "\n".join(l for l in code.split("\n")
                          if not l.lstrip().startswith("#"))
        for bad in ("config_id", "BY_ID", "MATRIX", "100000"):
            assert bad not in code, f"{bad} appears in v35's executable source"


# ======================================================================================
# GPU. Build the way run_matrix builds, warm the way the harness warms.
# ======================================================================================

def _build(config_id, candidate="v35_recombined", causal=None, padding=0.0,
           batch=None, seq=None):
    from bench.candidates import REGISTRY as R
    torch._dynamo.reset()          # L36: a shared Dynamo cache silently falls back to eager
    ref = _reference()
    cfg = BY_ID[config_id]
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    tcfg = ref.TransformerConfig(
        batch_size=cfg.batch_size if batch is None else batch,
        seq_len=cfg.seq_len if seq is None else seq,
        d_model=cfg.d_model, num_heads=cfg.heads, ffn_dim=cfg.ffn_dim,
        num_layers=cfg.layers, causal=cfg.causal if causal is None else causal)
    tcfg.validate()
    torch.manual_seed(1234)
    base = ref.BaselineTransformer(tcfg)
    model = R[candidate].build(ref.BaselineTransformer)(tcfg)
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
    k = sum(1 for ev in prof.events() if str(ev.device_type) == "DeviceType.CUDA")
    return k / n


# -------------------------------------------------------------- VERIFICATION 1: shape
@cuda
def test_the_shape_fix_survives_the_merge():
    """THE PROBE, re-run on the recombination.

    Warm at batch 8, then call at batch 1. v13's `_static_x.copy_(x)` BROADCASTS the
    smaller input and returns eight rows. Measured on the frontier: v26 hands back
    (8, 128, 128) for a (1, 128, 128) input.

    The parent is asserted here on purpose, exactly as v33's own test does it. Without
    the control this file would only show that v35 works, never that there was anything
    to fix -- and an assurance nobody arranged to be capable of failing is [L40].
    """
    shapes, answers = {}, {}
    for name in ("v26_causal_correct", "v35_recombined"):
        _ref, base, cand, make = _build(2, candidate=name, batch=8, seq=128)
        x8, m8 = make(1234)
        x1, m1 = x8[3:4].contiguous(), m8[3:4].contiguous()
        with torch.inference_mode():
            cand(x8, m8)                                 # warm and latch at batch 8
            try:
                y = cand(x1, m1)
                shapes[name] = tuple(y.shape)
                answers[name] = (y.clone(), base(x1, m1).clone())
            except RuntimeError:
                shapes[name] = "raised"
    assert shapes["v35_recombined"] == (1, 128, 128), shapes
    y, e = answers["v35_recombined"]
    assert (y - e).abs().max().item() < ATOL, "re-decided path gave a wrong answer"
    assert shapes["v26_causal_correct"] != (1, 128, 128), (
        "the parent no longer exhibits the shape latch this candidate inherits a fix "
        "for; if that is a real change, this test needs rewriting")


@cuda
def test_the_reshape_re_decides_the_launch_plan_against_the_new_shape():
    """THE MERGE ITSELF. Without v35's reset, `launch_bm` and `launch_reason` survive a
    shape change and the megakernel runs a tile sized for a batch that is no longer there.

    1024 tokens -> a 16-row tile; 2048 tokens -> a 32-row tile. Both are fused, so a
    stale plan would not announce itself by declining; it would just be wrong about the
    machine it is filling.
    """
    _ref, base, cand, make = _build(2, batch=8, seq=128)
    x8, m8 = make(1234)
    with torch.inference_mode():
        cand(x8, m8)
    assert cand.launch_fused_used, cand.launch_reason
    first_bm, first_reason = cand.launch_bm, cand.launch_reason
    assert first_bm == launch_tile(8 * 128, SM_COUNT)

    _ref2, base2, _c, make2 = _build(2, batch=16, seq=128)
    x16, m16 = make2(4321)
    with torch.inference_mode():
        y = cand(x16, m16)
        e = base2(x16, m16)          # _build seeds before construction: same weights
    assert cand.launch_fused_used, cand.launch_reason
    assert cand.launch_bm == launch_tile(16 * 128, SM_COUNT), (
        f"tile not re-decided: {cand.launch_bm} (was {first_bm}) for 2048 tokens")
    assert cand.launch_bm != first_bm, ("pick two shapes whose tiles differ, or this "
                                        "test cannot see a stale plan")
    assert cand.launch_reason != first_reason
    assert y.shape == x16.shape
    assert (y - e).abs().max().item() < ATOL


@cuda
def test_a_padded_mask_at_a_new_shape_does_not_run_the_maskless_kernel():
    """THE HAZARD v33's FIX CREATES AND v35 CLOSES.

    `_nomask` is derived once, in `_prime`. v34's `_core` gates only on
    `launch_fused_used` -- the mask check lives in `_decide_launch`, which without a reset
    runs once ever. And v34's `_try_capture` ELIDES the mask buffer entirely when
    `_nomask` is True.

    In v34 alone that is unreachable: the model raises on the second shape before it can
    matter. v33 removes the raise. So the merge is the first configuration in which a
    model warmed all-True can be re-called, at a new shape, with a mask that must be
    honoured -- and the fused kernel does no masking. v35 re-derives the mask state on the
    shape change, `_decide_launch` re-runs and declines.
    """
    _ref, _b, cand, make = _build(2, batch=8, seq=128, padding=0.0)
    x8, m8 = make(1234)
    with torch.inference_mode():
        cand(x8, m8)
    assert cand.launch_fused_used, "warm-up did not take the fused path; test is vacuous"
    assert cand._nomask

    _ref2, base2, _c, make2 = _build(2, batch=16, seq=128, padding=0.4)
    xp, mp = make2(99)
    assert not bool(mp.all().item()), "the padded case produced an all-true mask"
    with torch.inference_mode():
        y = cand(xp, mp)
        e = base2(xp, mp)
    assert not cand._nomask, "mask state was not re-derived on the shape change"
    assert not cand.launch_fused_used, cand.launch_reason
    assert "masked" in cand.launch_reason, cand.launch_reason
    assert (y - e).abs().max().item() < ATOL, "the maskless kernel ran on a masked input"


# ------------------------------------------------------ VERIFICATION 2: kernel count
@cuda
@pytest.mark.parametrize("cid", [2, 12])
def test_the_kernel_reduction_survives_the_merge(cid):
    """L36, and counted rather than argued. The parent replays 36 nodes per forward on
    every config; v35 must engage the fused path AND actually replay fewer.

    Speed is not claimed here. The count is what the launch-floor argument rests on, and
    the count is the thing the merge could have silently lost.
    """
    _ref, _b, model, make = _build(cid)
    x, m = make(1234)
    with torch.inference_mode():
        model(x, m)
    assert model.launch_fused_used, model.launch_reason
    assert model.launch_bm >= MIN_TILE_ROWS
    assert model.mask_capture.startswith("elided"), model.mask_capture
    assert model.stream_path == "resident", model.stream_reason

    _r2, _b2, parent, pmake = _build(cid, candidate="v33_streamed_long")
    px, pm = pmake(1234)
    n_parent = _kernels_per_forward(parent, px, pm)
    n_child = _kernels_per_forward(model, x, m)
    print(f"\nconfig {cid}: v33 {n_parent:.1f} kernels -> v35 {n_child:.1f} "
          f"(drop {n_parent - n_child:.1f})")

    # THE ASSERTION IS ON THE DROP, NOT THE ABSOLUTE COUNT, and that is a measured
    # choice rather than a loosened one. Run alone, this counts 36.0 and 20.0 five times
    # out of five on both configs. Run inside the full suite it counts 35.3 and 19.3 --
    # a CONSTANT deficit of 7 events per profiled window, identical for both models,
    # which the profiler drops in a loaded process. The difference is unchanged at
    # exactly 16.0 in both environments, so the difference is what the mechanism claim
    # rests on and the absolute number is what the environment perturbs.
    # The DROP is the invariant, not the absolute count. Measured alone the counts are
    # 36.0 / 20.0; inside the full suite both undercount by a constant ~7 profiler events
    # (35.3 / 19.3) as CUDA module and context state accumulates across the 36 candidates
    # compiled in one process. The difference is 16.0 in both contexts. A test whose
    # verdict depends on what ran before it is not testing the candidate (L36).
    assert n_parent >= 30.0, f"parent count implausible: {n_parent}"
    assert n_parent - n_child == pytest.approx(16.0, abs=1.5), (n_parent, n_child)


# ------------------------------------------------------- VERIFICATION 3: streaming
@cuda
def test_streaming_still_engages_and_settles_the_launch_decision():
    """v33's mechanism, plus the merge's obligation: on the streamed path v34's `forward`
    is never reached, so without the hook `launch_reason` would still read "undecided"
    at the moment `_core` runs. Not wrong -- the flag defaults to False -- but silent,
    which is the failure mode finding 18 is about."""
    _ref, base, cand, make = _build(2, batch=8, seq=256)
    x, m = make(1234)
    real = torch.cuda.mem_get_info
    torch.cuda.mem_get_info = lambda *a, **k: (1_000_000, real()[1])
    try:
        with torch.inference_mode():
            y = cand(x, m).clone()
    finally:
        torch.cuda.mem_get_info = real
    assert cand.stream_path == "streamed", cand.stream_reason
    assert cand.stream_slice < 8, "streaming with a full-batch slice is not streaming"
    assert cand.launch_reason != "undecided", (
        "the streamed path ran with the launch decision unsettled")
    with torch.inference_mode():
        e = base(x, m)
    assert (y - e).abs().max().item() < ATOL


@cuda
def test_streamed_and_resident_agree_inside_the_locked_tolerance():
    _ref, _b, resident, make = _build(2, batch=8, seq=256)
    _r2, _b2, streamed, _m2 = _build(2, batch=8, seq=256)
    x, m = make(1234)
    real = torch.cuda.mem_get_info
    torch.cuda.mem_get_info = lambda *a, **k: (1_000_000, real()[1])
    try:
        with torch.inference_mode():
            ys = streamed(x, m).clone()
    finally:
        torch.cuda.mem_get_info = real
    with torch.inference_mode():
        yr = resident(x, m)
    assert streamed.stream_path == "streamed" and resident.stream_path == "resident"
    assert (ys - yr).abs().max().item() < ATOL


# ------------------------------------------------------ VERIFICATION 4: correctness
@cuda
@pytest.mark.parametrize("cid", [2, 4, 12, 8, 10])
def test_accuracy_at_the_locked_tolerance(cid):
    """2, 4 and 12 take the fused path; 8 and 10 decline it. Both must be right, and the
    declines must be observably declines rather than silent failures to engage."""
    _ref, base, model, make = _build(cid)
    x, m = make(7)
    with torch.inference_mode():
        y, e = model(x, m), base(x, m)
    assert torch.allclose(y, e, rtol=RTOL, atol=ATOL), \
        f"config {cid}: max_abs {(y - e).abs().max().item():.3e}"
    assert model.launch_reason != "undecided"
    assert model.causal_path.startswith("optimized")


@cuda
def test_the_fp32_residual_is_preserved():
    """Finding 08. The residual stream must not be fp16 anywhere in the merged path."""
    _ref, _b, model, make = _build(2)
    x, m = make(1234)
    with torch.inference_mode():
        y = model(x, m)
    assert model.launch_fused_used, model.launch_reason
    assert y.dtype == torch.float32
    src = (REPO / "bench" / "kernels" / "ffn_fused.py").read_text()
    assert "tl.float32" in src.split("attention residual add", 1)[1][:600]


@cuda
def test_non_causal_still_delegates_to_the_unmodified_baseline():
    """Finding 32 / [L42]. The harness's OWN default is causal=False, so this is the path
    an unflagged grader run takes -- through THREE dispatch layers now."""
    _ref, base, model, make = _build(2, causal=False)
    x, m = make(1234)
    with torch.inference_mode():
        y, e = model(x, m), base(x, m)
    assert model.causal_path.startswith("baseline"), model.causal_path
    assert not model.launch_fused_used
    assert torch.allclose(y, e, rtol=RTOL, atol=ATOL)


@cuda
def test_different_inputs_do_not_give_the_same_answer():
    """The stale-buffer control. A graph that replays nothing passes every tolerance test
    against a cached output."""
    _ref, _b, model, make = _build(2)
    x1, m1 = make(1)
    x2, _m2 = make(2)
    with torch.inference_mode():
        y1 = model(x1, m1).clone()
        y2 = model(x2, m1).clone()
    assert (y1 - y2).abs().max().item() > 1e-2
