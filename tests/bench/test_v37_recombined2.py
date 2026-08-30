"""v37: the second recombination, and specifically THE PART NEITHER PARENT CONTAINS.

`test_v35_recombined.py` and `test_v36_gemm_gelu.py` both still run here unchanged and are
the regression surface for the two halves. What this file has to establish is the merge:

  1. v33's SHAPE FIX survives -- the two-shape probe, against the same v26 control v33 and
     v35 used, because an assurance nobody arranged to be capable of failing is [L40];
  2. v35's RESET survives and now covers NINE more attributes, with the set computed FROM
     THE CLASSES rather than typed out, so a generation 38 that latches a fifteenth is
     caught here instead of shipping a stale plan ([L14]);
  3. v34's KERNEL-COUNT CUT survives -- the DROP, not an absolute count;
  4. v36's PROJECTION GEMMS actually FIRE, asserted by name in the device profile, because
     v36's own first draft silently did nothing on config 9 while `gemm_reason` claimed
     Triton and every accuracy test passed ([L36]);
  5. config 3 -- v35's unique win over v34 -- still takes the path that produced it.

Every GPU test asserts that the mechanism it is about engaged before it asserts anything
else. A merge test that quietly ran a parent's path asserts only that the parent equals
itself.
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
from bench.matrix import BY_ID                                            # noqa: E402
from bench.kernels.ffn_fused import MIN_TILE_ROWS, launch_tile            # noqa: E402

torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# This card, from ledger/device.json. Used only to evaluate a predicate as a pure
# function; the candidate reads it off get_device_properties at run time.
SM_COUNT = 66

RTOL, ATOL = 0.02, 0.002          # the harness's own defaults. Never widened.

CANDIDATE = "v37_recombined2"


def _reference(tag="ref_bench_t37"):
    path = REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def _chain():
    """v26, v34, v36, v35, v37 -- all built off one reference class."""
    ref = _reference("ref_bench_t37_chain")
    from bench.candidates.v26_causal_correct import build as b26
    from bench.candidates.v34_launch_bound import build as b34
    from bench.candidates.v36_gemm_gelu import build as b36
    B = ref.BaselineTransformer
    return (b26(B), b34(B), b36(B),
            REGISTRY["v35_recombined"].build(B),
            REGISTRY[CANDIDATE].build(B))


# ======================================================================================
# Lineage and the merge's bookkeeping. No GPU.
# ======================================================================================

class TestLineage:
    def test_parent_is_v36_and_v35_is_a_named_contributor(self):
        spec = REGISTRY[CANDIDATE]
        assert spec.parent == "v36_gemm_gelu"
        assert spec.generation == 37
        # The contributor has to EXIST as a candidate in its own right. A recombination
        # that absorbs a sibling without registering it destroys the sibling's clade.
        assert "v35_recombined" in spec.summary
        assert REGISTRY["v35_recombined"].parent == "v33_streamed_long"
        assert REGISTRY["v36_gemm_gelu"].parent == "v34_launch_bound"

    def test_neither_line_is_an_ancestor_of_the_other(self):
        """The premise of the whole file. If one already contained the other this would
        be an inheritance, not a recombination."""
        def anc(name):
            out, cur = set(), REGISTRY[name].parent
            while cur is not None:
                out.add(cur)
                cur = REGISTRY[cur].parent if cur in REGISTRY else None
            return out
        assert "v36_gemm_gelu" not in anc("v35_recombined")
        assert "v35_recombined" not in anc("v36_gemm_gelu")
        assert "v26_causal_correct" in anc("v35_recombined") & anc("v36_gemm_gelu")

    def test_the_merge_did_not_drop_a_registered_candidate(self):
        """[L14]. A regex resolution broke this registry twice in one session."""
        for name, spec in REGISTRY.items():
            assert spec.name == name, f"{name} keyed against spec named {spec.name}"
            assert spec.parent is None or spec.parent in REGISTRY, (name, spec.parent)
        assert {"v26_causal_correct", "v33_streamed_long", "v34_launch_bound",
                "v35_recombined", "v36_gemm_gelu", CANDIDATE} <= set(REGISTRY)

    def test_no_mechanism_was_retyped(self):
        """Both halves must be the SAME objects the siblings use, not copies. Two copies
        of a threshold drift apart; one copy cannot."""
        from bench.candidates import v33_streamed_long as S
        from bench.candidates import v36_gemm_gelu as G
        from bench.candidates import v37_recombined2 as M
        assert M.build_streaming_on is S.build_on
        assert M.build_v36 is G.build


class TestTheLayering:
    def test_streaming_sits_above_the_gemm_which_sits_above_the_fusion(self):
        _v26, _v34, _v36, _v35, v37 = _chain()
        names = [c.__name__ for c in v37.__mro__]
        assert names[:5] == ["CandidateV37", "CandidateV33", "CandidateV36",
                             "CandidateV34", "CandidateV26"], names
        # One copy of each layer. A diamond here would mean a mechanism applied twice.
        assert len(names) == len(set(names)), names

    def test_v35s_own_layering_is_untouched_by_this_generation(self):
        """v35 is PORTED here byte for byte, not adapted. If this file's existence had
        changed it, v35's own measured rows would no longer describe the code."""
        _v26, _v34, _v36, v35, _v37 = _chain()
        names = [c.__name__ for c in v35.__mro__]
        assert names[:4] == ["CandidateV35", "CandidateV33", "CandidateV34",
                             "CandidateV26"], names


# ======================================================================================
# VERIFICATION 2 (static half): the reset set is DERIVED, and it covers v35's.
# ======================================================================================

class TestTheResetIsDerivedFromTheClasses:
    def test_the_derivation_reproduces_v35s_hand_written_list_exactly(self):
        """The cross-check that the generic rule is the same rule v35 applied by hand.

        v35 typed out five names. Run over the same two classes, `shape_latched_over`
        must return exactly those five -- otherwise the generalization is a different
        claim wearing v35's clothes.
        """
        from bench.candidates.v37_recombined2 import shape_latched_over
        v26, v34, _v36, v35, _v37 = _chain()
        derived = set(shape_latched_over(v34, v26))
        assert derived == set(v35.SHAPE_LATCHED_BY_V34), (
            sorted(derived), sorted(v35.SHAPE_LATCHED_BY_V34))

    def test_v36_latches_nine_more_and_all_of_them_are_covered(self):
        """THE MERGE'S ONE LOAD-BEARING CLAIM, one generation on from v35's.

        v35's test was written so that "a generation 36 that latches a sixth attribute
        fails a test instead of shipping a wrong answer". Generation 36 latched NINE --
        five `gemm_*` flags and four private tile tuples, every one computed in
        `_decide_gemm` from `m = b * s`. This asserts the derived set covers v35's five
        AND names the nine, so a mechanism silently dropped from the reset fails here.
        """
        from bench.candidates.v37_recombined2 import shape_latched_over
        v26, _v34, v36, v35, v37 = _chain()
        derived = set(shape_latched_over(v36, v26))
        assert derived == set(v37.SHAPE_LATCHED), "the class does not use its own rule"
        assert set(v35.SHAPE_LATCHED_BY_V34) < derived, (
            f"v35's reset covered {sorted(v35.SHAPE_LATCHED_BY_V34)} and this covers "
            f"{sorted(derived)}; anything in the first set and not the second survives a "
            f"shape change and is then read against the wrong shape.")
        assert derived - set(v35.SHAPE_LATCHED_BY_V34) == {
            "gemm_used", "gemm_reason", "gemm_sites", "gemm_engaged", "gemm_stats",
            "_tile_qkv", "_tile_out", "_tile_ffn_in", "_tile_ffn_out"}, sorted(derived)

    def test_the_private_tile_tuples_are_not_skipped(self):
        """v35's own helper excluded underscore-prefixed names, which was sound for v34
        (it latches nothing private) and would have silently dropped four of v36's nine.
        This is the line where that exclusion had to go."""
        _v26, _v34, _v36, _v35, v37 = _chain()
        assert {"_tile_qkv", "_tile_out", "_tile_ffn_in",
                "_tile_ffn_out"} <= set(v37.SHAPE_LATCHED)

    def test_the_defaults_are_the_class_body_defaults(self):
        """The reset restores each attribute to the value its own class declares, so a
        re-decision starts from the same state a fresh model does."""
        _v26, _v34, v36, _v35, v37 = _chain()
        for name, default in v37.SHAPE_LATCHED.items():
            assert getattr(v36, name) == default, name
        assert v37.SHAPE_LATCHED["gemm_reason"] == "undecided"
        assert v37.SHAPE_LATCHED["launch_reason"] == "undecided"

    def test_the_derivation_can_actually_fail(self):
        """[L38]: verify a check is CAPABLE of failing before trusting that it passed.

        A subclass that latches a fifteenth attribute must be seen by the same rule,
        with no edit to this file or to the candidate.
        """
        from bench.candidates.v37_recombined2 import shape_latched_over
        v26, _v34, v36, _v35, _v37 = _chain()

        class Generation38(v36):
            new_shape_latched_thing = 0
            def _decide_something(self):        # a method must NOT be picked up
                return None

        derived = set(shape_latched_over(Generation38, v26))
        assert "new_shape_latched_thing" in derived
        assert "_decide_something" not in derived

    def test_every_derived_attribute_is_actually_reset_by_the_override(self):
        """Read the source of the override rather than trusting the tuple."""
        src = (REPO / "bench" / "candidates" / "v37_recombined2.py").read_text()
        body = src.split("def _invalidate_shape_state", 1)[1].split(
            "def _settle_slice_decisions", 1)[0]
        assert "self.SHAPE_LATCHED.items()" in body, "the reset does not use the set"
        assert re.search(r"setattr\(self, name,", body), "nothing is assigned"
        assert "self._prime(" in body, "the mask-derived state is never re-derived"
        assert "_proj_t" in body, "the instance-level projection cache is never dropped"

    def test_the_streamed_path_settles_both_new_decisions_in_order(self):
        """`_decide_gemm` READS `launch_fused_used`, so planning before it plans FFN
        sites a megakernel already owns."""
        src = (REPO / "bench" / "candidates" / "v37_recombined2.py").read_text()
        # The executable body only: the docstring names them in the other order, on
        # purpose, because it is explaining the dependency rather than obeying it.
        hook = src.split("def _settle_slice_decisions", 1)[1].split('"""', 2)[-1]
        assert "_decide_launch" in hook and "_decide_gemm" in hook
        assert hook.index("_decide_launch") < hook.index("_decide_gemm"), (
            "the GEMM plan is settled before the fusion decision it reads")

    def test_no_config_ids_or_announced_shapes_in_the_candidate_source(self):
        """CLAUDE.md rule 2. The merge adds no new constant; this proves it."""
        src = (REPO / "bench" / "candidates" / "v37_recombined2.py").read_text()
        code = src.split('"""', 2)[-1]                    # drop the module docstring
        code = "\n".join(l for l in code.split("\n")
                         if not l.lstrip().startswith("#"))
        for bad in ("config_id", "BY_ID", "MATRIX", "100000", "8192"):
            assert bad not in code, f"{bad} appears in v37's executable source"


# ======================================================================================
# GPU. Build the way run_matrix builds, warm the way the harness warms.
# ======================================================================================

def _build(config_id, candidate=CANDIDATE, causal=None, padding=0.0,
           batch=None, seq=None):
    torch._dynamo.reset()          # L36: a shared Dynamo cache silently falls back to eager
    ref = _reference(f"ref_bench_t37_{candidate}_{config_id}_{batch}_{seq}")
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
    model = REGISTRY[candidate].build(ref.BaselineTransformer)(tcfg)
    ref.copy_model_weights(base, model)
    base = base.to(device=dev, dtype=torch.float32).eval()
    model = model.to(device=dev, dtype=torch.float32).eval()

    def make(seed):
        return ref.generate_random_case(tcfg, dev, torch.float32, seed=seed,
                                        padding_ratio=padding, input_scale=1.0)
    return ref, base, model, make


def _device_events(model, x, m, n=10):
    """(kernels per forward, the set of device kernel names seen)."""
    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        for _ in range(10):
            model(x, m)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n):
                model(x, m)
            torch.cuda.synchronize()
    evs = [ev for ev in prof.events() if str(ev.device_type) == "DeviceType.CUDA"]
    return len(evs) / n, {ev.name for ev in evs}


def _kernels_per_forward(model, x, m, n=10):
    return _device_events(model, x, m, n)[0]


# --------------------------------------------------- VERIFICATION 1: the shape fix
@cuda
def test_the_shape_fix_survives_the_second_merge():
    """THE PROBE, re-run on the recombination of the recombination.

    Warm at batch 8, then call at batch 1. v13's `_static_x.copy_(x)` BROADCASTS the
    smaller input and returns eight rows. Measured on the frontier: v26 hands back
    (8, 128, 128) for a (1, 128, 128) input.

    The v26 control is asserted on purpose, exactly as v33's and v35's tests do it.
    Without it this test would only show that v37 works, never that there was anything to
    fix -- and an assurance nobody arranged to be capable of failing is [L40].
    """
    shapes, answers = {}, {}
    for name in ("v26_causal_correct", CANDIDATE):
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
    assert shapes[CANDIDATE] == (1, 128, 128), shapes
    y, e = answers[CANDIDATE]
    assert (y - e).abs().max().item() < ATOL, "the re-decided path gave a wrong answer"
    assert shapes["v26_causal_correct"] != (1, 128, 128), (
        "the control no longer exhibits the shape latch this candidate inherits a fix "
        "for; if that is a real change, this test needs rewriting")


# ------------------------------------- VERIFICATION 2 (live half): the reset fires
@cuda
def test_a_reshape_re_decides_the_launch_plan():
    """v35's half of the reset, inherited. 1024 tokens -> a 16-row tile; 2048 tokens ->
    a 32-row tile. Both are fused, so a stale plan would not announce itself by
    declining; it would just be wrong about the machine it is filling."""
    _ref, _b, cand, make = _build(2, batch=8, seq=128)
    x8, m8 = make(1234)
    with torch.inference_mode():
        cand(x8, m8)
    assert cand.launch_fused_used, cand.launch_reason
    first_bm = cand.launch_bm
    assert first_bm == launch_tile(8 * 128, SM_COUNT)

    _r2, base2, _c, make2 = _build(2, batch=16, seq=128)
    x16, m16 = make2(4321)
    with torch.inference_mode():
        y = cand(x16, m16)
        e = base2(x16, m16)          # _build seeds before construction: same weights
    assert cand.launch_fused_used, cand.launch_reason
    assert cand.launch_bm == launch_tile(16 * 128, SM_COUNT), (
        f"tile not re-decided: {cand.launch_bm} (was {first_bm}) for 2048 tokens")
    assert cand.launch_bm != first_bm, ("pick two shapes whose tiles differ, or this "
                                        "test cannot see a stale plan")
    assert y.shape == x16.shape
    assert (y - e).abs().max().item() < ATOL


@cuda
def test_a_reshape_re_decides_the_gemm_plan_and_the_control_shows_it_would_not():
    """THE HALF v35 COULD NOT HAVE: v36's plan is latched to `m = b * s` too.

    The control is the point ([L40]). A subclass that keeps v33's reset and drops v37's
    extension is exactly the naive composition of the two lines, and it must be SEEN to
    keep the old shape's plan -- otherwise this test proves nothing about the reset.
    """
    from bench.candidates.v33_streamed_long import build_on as stream_on
    from bench.candidates.v36_gemm_gelu import build as b36

    def probe(cls_builder, tag):
        _ref, _b, cand, make = _build(9, candidate=CANDIDATE, batch=8, seq=128)
        if cls_builder is not None:                       # swap in the naive composition
            ref2 = _reference(f"ref_bench_t37_naive_{tag}")
            cfg = BY_ID[9]
            tcfg = ref2.TransformerConfig(
                batch_size=8, seq_len=128, d_model=cfg.d_model, num_heads=cfg.heads,
                ffn_dim=cfg.ffn_dim, num_layers=cfg.layers, causal=True)
            torch.manual_seed(1234)
            cand = cls_builder(ref2.BaselineTransformer)(tcfg).to(
                device="cuda", dtype=torch.float32).eval()
            make = lambda seed, t=tcfg, r=ref2: r.generate_random_case(  # noqa: E731
                t, torch.device("cuda"), torch.float32, seed=seed,
                padding_ratio=0.0, input_scale=1.0)
        x8, m8 = make(1234)
        with torch.inference_mode():
            cand(x8, m8)
        before = cand.gemm_reason
        _r2, _b2, _c2, make2 = _build(9, batch=64, seq=128)
        x64, m64 = make2(4321)
        with torch.inference_mode():
            cand(x64, m64)
        return before, cand.gemm_reason

    naive_before, naive_after = probe(lambda B: stream_on(b36(B)), "ctl")
    assert naive_before == naive_after, (
        "the control did not exhibit the stale plan this candidate resets; if v36 stopped "
        "latching its plan to the input shape, this test needs rewriting")

    before, after = probe(None, "v37")
    assert before != after, (
        f"the GEMM plan was not re-derived at the new shape: {before!r}")
    assert "probe M" in after or "vendor" in after or "triton" in after, after


@cuda
def test_a_padded_mask_at_a_new_shape_does_not_run_the_maskless_kernel():
    """THE WRONG ANSWER v33's FIX CREATES AND v35 CLOSES, re-verified one layer up.

    v34's `_try_capture` ELIDES the mask buffer when `_nomask`; the mask check lives in
    `_decide_launch`, which without a reset runs once ever. In v34 alone that is
    unreachable -- the model raises on the second shape. v33 removes the raise. v35
    measured 69407/262144 elements past the locked tolerance with the reset neutered.
    """
    _ref, _b, cand, make = _build(2, batch=8, seq=128, padding=0.0)
    x8, m8 = make(1234)
    with torch.inference_mode():
        cand(x8, m8)
    assert cand.launch_fused_used, "warm-up did not take the fused path; test is vacuous"
    assert cand._nomask

    _r2, base2, _c, make2 = _build(2, batch=16, seq=128, padding=0.4)
    xp, mp = make2(99)
    assert not bool(mp.all().item()), "the padded case produced an all-true mask"
    with torch.inference_mode():
        y = cand(xp, mp)
        e = base2(xp, mp)
    assert not cand._nomask, "the mask state was not re-derived on the shape change"
    assert not cand.launch_fused_used, cand.launch_reason
    assert "masked" in cand.launch_reason, cand.launch_reason
    assert (y - e).abs().max().item() < ATOL, "the maskless kernel ran on a masked input"


# ------------------------------------ VERIFICATION 3: v34's kernel-count cut survives
@cuda
@pytest.mark.parametrize("cid", [2, 12])
def test_the_kernel_reduction_survives_the_second_merge(cid):
    """[L36], counted rather than argued. v33 replays 36 nodes per forward on every
    config (identical to v26 -- the streaming layer adds no kernel on the resident path,
    which is the negative control for the merge); v37 must engage the fused path AND
    actually replay sixteen fewer.

    THE ASSERTION IS ON THE DROP, NOT THE ABSOLUTE COUNT, and that is a measured choice
    rather than a loosened one. Run alone these count 36.0 and 20.0; run inside the full
    suite they count ~35.3 and ~19.3 -- a constant deficit of ~7 events per profiled
    window, identical for both models, which the profiler drops in a loaded process (and
    which has grown as CUDA context state accumulates across 37 candidates). The
    DIFFERENCE is what the mechanism claims; the absolute is what the environment moves.
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
    print(f"\nconfig {cid}: v33 {n_parent:.1f} kernels -> v37 {n_child:.1f} "
          f"(drop {n_parent - n_child:.1f})")
    assert n_parent - n_child == pytest.approx(16.0, abs=1.5), (n_parent, n_child)


# ---------------------------------- VERIFICATION 4: v36's projection GEMMs actually fire
@cuda
@pytest.mark.parametrize("cid", [9, 10, 1])
def test_the_projection_gemms_fire_and_are_visible_by_name(cid):
    """[L36] in its purest form, and the reason this assertion is by NAME.

    v36's first draft bailed to the parent whenever v23's attention kernel declined.
    Config 9 is `heads=1, head_dim=128`, exactly where v23 declines -- so the #1 headroom
    row ran four cuBLAS calls while `gemm_reason` reported "triton on
    out+ffn_in+ffn_out", and every accuracy test passed. A reason string is a claim; the
    device profile is the evidence.

    Two independent witnesses are required here: the Triton kernel must appear in the
    device events BY NAME, and the four free-standing GELU launches must disappear.
    """
    _r, _b, model, make = _build(cid)
    x, m = make(1234)
    with torch.inference_mode():
        model(x, m)
    assert model.gemm_used, model.gemm_reason
    assert model.gemm_engaged, model.gemm_reason
    for site in ("out", "ffn_in", "ffn_out"):
        assert site in model.gemm_sites, (site, model.gemm_reason)
    assert all(s.get("n_spills") == 0 for s in model.gemm_stats.values()), model.gemm_stats

    n_child, child_names = _device_events(model, x, m)
    assert any("proj_gemm" in n for n in child_names), (
        f"`gemm_reason` says {model.gemm_reason!r} but no _proj_gemm kernel reached the "
        f"device: {sorted(child_names)}")
    assert not any("gelu" in n.lower() for n in child_names), (
        f"a free-standing GELU is still launching: {sorted(child_names)}")

    _r2, _b2, parent, pmake = _build(cid, candidate="v34_launch_bound")
    px, pm = pmake(1234)
    n_parent, parent_names = _device_events(parent, px, pm)
    assert any("gelu" in n.lower() for n in parent_names), (
        "the control does not launch a free-standing GELU, so absorbing it proves nothing")
    print(f"\nconfig {cid}: v34 {n_parent:.1f} kernels -> v37 {n_child:.1f} "
          f"(drop {n_parent - n_child:.1f})")
    # Four GELU launches, one per layer, absorbed into the ffn_in epilogue.
    assert n_parent - n_child >= 3.5, (n_parent, n_child)


# ------------------------------------------- VERIFICATION 5: config 3, v35's unique win
@cuda
def test_config_3_still_takes_the_path_that_produced_v35s_win():
    """Config 3 is the row v35 wins outright -- 0.0553 ms against v34's 0.0886, where v34
    is WORSE than the v26 it descends from. This file cannot rank two candidates (the
    graded harness's baseline arm spreads 39% here -- finding 42), so what it pins is the
    MECHANISM: the one-wave fusion must still claim config 3, the kernel count must still
    fall, and the GEMM plan must not have quietly stolen the FFN sites the megakernel
    already owns. The timing is in the deliverable, under a stated protocol.
    """
    _ref, base, model, make = _build(3)
    x, m = make(1234)
    with torch.inference_mode():
        y, e = model(x, m), base(x, m)
    assert model.stream_path == "resident", model.stream_reason
    assert model.launch_fused_used, model.launch_reason
    assert "fused" in model.launch_reason
    assert model.gemm_reason != "undecided"
    # Where v34's megakernel owns ffn_in, the GELU and ffn_out there is no `F.linear`
    # left to replace; planning one produces a tile that is never launched.
    assert "ffn_in" not in model.gemm_sites, model.gemm_reason
    assert "ffn_out" not in model.gemm_sites, model.gemm_reason
    assert model._tile_ffn_in is None and model._tile_ffn_out is None
    assert (y - e).abs().max().item() < ATOL

    _r2, _b2, parent, pmake = _build(3, candidate="v33_streamed_long")
    px, pm = pmake(1234)
    n_parent = _kernels_per_forward(parent, px, pm)
    n_child = _kernels_per_forward(model, x, m)
    print(f"\nconfig 3: v33 {n_parent:.1f} kernels -> v37 {n_child:.1f} "
          f"(drop {n_parent - n_child:.1f}); gemm sites {model.gemm_sites}")
    assert n_parent - n_child == pytest.approx(16.0, abs=1.5), (n_parent, n_child)


# ----------------------------------------------------------------- streaming survives
@cuda
def test_streaming_still_engages_and_settles_all_three_decisions():
    """On the streamed path v36's `forward` is never reached, so without the hook
    `launch_reason` AND `gemm_reason` would still read "undecided" at the moment `_core`
    runs. Not wrong -- both flags default to False -- but silent, which is the failure
    mode finding 18 is about."""
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
    assert cand.gemm_reason != "undecided", (
        "the streamed path ran with the GEMM plan unsettled")
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


# ------------------------------------------------------------------- correctness
@cuda
@pytest.mark.parametrize("cid", [1, 2, 3, 4, 5, 8, 9, 10, 11, 12])
def test_accuracy_at_the_locked_tolerance(cid):
    """Both halves of the dispatch on both branches of `_core`, plus config 8 where the
    GEMM predicate declines outright and must be observably a decline."""
    ref, base, model, make = _build(cid)
    with torch.inference_mode():
        for t in range(3):
            x, m = make(1234 + t)
            res = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
            assert res.passed, (cid, t, float(res.max_abs_error))
    assert model.launch_reason != "undecided"
    assert model.gemm_reason != "undecided"
    assert model.causal_path.startswith("optimized")


@cuda
def test_the_gemm_predicate_can_still_say_no():
    """Config 8 is d_model 1024, where cuBLAS selects
    `cutlass_80_tensorop_f16_s16816gemm` and reaches 100.4% of this card's measured peak.
    If the predicate cannot decline there it is not a predicate, it is a preference."""
    _r, _b, model, make = _build(8)
    x, m = make(1234)
    with torch.inference_mode():
        model(x, m)
    assert not model.gemm_used, model.gemm_reason
    assert model.gemm_sites == (), model.gemm_reason
    assert "vendor" in model.gemm_reason


@cuda
def test_the_fp32_residual_is_preserved():
    """Finding 08. The residual stream must not be fp16 anywhere in the merged path."""
    _r, _b, model, make = _build(2)
    x, m = make(1234)
    with torch.inference_mode():
        y = model(x, m)
    assert model.launch_fused_used, model.launch_reason
    assert y.dtype == torch.float32
    src = (REPO / "bench" / "kernels" / "ffn_fused.py").read_text()
    assert "tl.float32" in src.split("attention residual add", 1)[1][:600]


def _executable_source(path: Path) -> str:
    """The module's source with every comment and string literal removed.

    Stripping only `#` lines is not enough, and the difference is not academic: the first
    version of this check failed on `proj_gemm.py`'s own module docstring, which argues
    AGAINST tanh by name. Prose that cites the thing it is refusing is the point of a
    docstring; what matters is what the CODE computes. (Borrowed from
    `test_v36_gemm_gelu.py`, which learned the same lesson about config ids.)
    """
    import tokenize
    out = []
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL):
                continue
            out.append(tok.string)
    return " ".join(out)


def test_the_gelu_is_the_exact_erf_form_in_every_kernel_that_computes_it():
    """`approximate="none"`. The tanh approximation differs by up to ~1e-3 -- half the
    entire 2e-3 budget spent on an approximation nobody asked for -- and both of the
    merged lines compute a GELU: v36's in the `ffn_in` epilogue, v34's inside the
    megakernel. One is not enough."""
    for name in ("proj_gemm.py", "ffn_fused.py"):
        code = _executable_source(REPO / "bench" / "kernels" / name)
        assert "erf" in code, name
        assert "tanh" not in code, name
    # [L38]: the stripper must be doing real work, or this check passes vacuously.
    assert "tanh" in (REPO / "bench" / "kernels" / "proj_gemm.py").read_text(), (
        "the docstring should still argue against the approximation it refuses")


@cuda
def test_non_causal_still_delegates_to_the_unmodified_baseline():
    """Finding 32 / [L42]. The harness's OWN default is causal=False, so this is the path
    an unflagged grader run takes -- through FOUR dispatch layers now."""
    ref, base, model, make = _build(9, causal=False)
    x, m = make(1234)
    with torch.inference_mode():
        res = ref.compare_outputs(base(x, m), model(x, m), rtol=RTOL, atol=ATOL)
    assert res.passed, float(res.max_abs_error)
    assert model.causal_path.startswith("baseline"), model.causal_path
    assert not model.launch_fused_used
    assert not model.gemm_used


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
def test_different_inputs_do_not_give_the_same_answer():
    """[L25]/[L23]: the stale-buffer control. A graph that replays nothing is correct
    against a cached output forever."""
    _r, _b, model, make = _build(9)
    x1, m1 = make(1)
    x2, m2 = make(2)
    with torch.inference_mode():
        y1 = model(x1, m1).clone()
        y2 = model(x2, m2).clone()
    assert not torch.allclose(y1, y2), "the model ignored its input"


@cuda
def test_repeating_a_call_repeats_its_answer():
    """The other half of [L25]: a difference measured against the reference is then a
    difference in arithmetic, not in scheduling."""
    _r, _b, model, make = _build(9)
    x1, m1 = make(1)
    x2, m2 = make(2)
    with torch.inference_mode():
        first = model(x1, m1).clone()
        model(x2, m2)
        again = model(x1, m1).clone()
    assert torch.equal(first, again), (first - again).abs().max().item()
