"""The matrix is the single source of truth; pin the facts everything else derives from."""
from bench.matrix import MATRIX, BY_ID, REGIMES, regime_of, weighted_score


def test_fourteen_configs_with_dense_ids():
    assert len(MATRIX) == 14
    assert sorted(BY_ID) == list(range(1, 15))


def test_every_announced_config_is_causal():
    # The whole causal-skipping argument rests on this; if a non-causal row ever appears,
    # the "half the score matrix is structurally zero" reasoning stops holding.
    assert all(c.causal for c in MATRIX)


def test_ffn_dim_equals_model_dim_on_every_row():
    # Differs from the reference benchmark's own 4x default, and is why attention weighs
    # more here than profiling the reference's defaults suggests.
    assert all(c.ffn_dim == c.d_model for c in MATRIX)


def test_head_dim_spans_the_awkward_cases():
    dims = {c.id: c.head_dim for c in MATRIX}
    assert dims[7] == 8 and dims[11] == 8      # vendor fast paths may refuse these
    assert dims[8] == 256 and dims[9] == 128
    assert set(dims.values()) == {8, 32, 64, 128, 256}


def test_every_config_has_exactly_one_regime():
    ids = [i for group in REGIMES.values() for i in group]
    assert sorted(ids) == list(range(1, 15)), "regimes must partition the matrix"
    assert regime_of(14) == "extreme"


def test_config_14_cannot_materialize_its_scores():
    # 20 TB of score matrix: the reference implementation cannot run this config at any
    # batch size, which makes it a feasibility result rather than a speed one.
    assert BY_ID[14].dense_scores_bytes() > 1e13


def test_weighted_score_clips_and_penalizes_unmeasured():
    # One spectacular regime must not carry a submission that is mediocre elsewhere.
    assert weighted_score({1: 100.0}) < weighted_score({i: 2.0 for i in range(1, 15)})
    # An unmeasured config scores 1.0, not "skipped" -- skipping would reward not running.
    assert weighted_score({1: 3.0}) == (3.0 + 13 * 1.0) / 14


def test_chunk_solver_is_calibrated_not_hardcoded():
    # The measured optimum for config 6 on a 48 MB L2 card was 125; the solver must land
    # near it WITHOUT that number appearing anywhere in the code.
    from bench.candidates.v3_chunked import solve_chunk
    L2_48MB = 50_331_648
    assert 96 <= solve_chunk(10000, 128, 128, L2_48MB) <= 160

    # And it must respond to the device, not ignore it: 4x the cache should raise the
    # chunk roughly 4x. A solver that does not move with L2 is a hardcoded table.
    small = solve_chunk(10000, 128, 128, L2_48MB)
    large = solve_chunk(10000, 128, 128, L2_48MB * 4)
    assert large > 3 * small

    # Never chunk beyond the batch, never to zero.
    assert solve_chunk(8, 128, 128, L2_48MB) == 8
    assert solve_chunk(10000, 100000, 1024, L2_48MB) >= 1


def test_every_registered_candidate_actually_builds():
    # A merge-conflict resolution once dropped `return build(baseline_cls)` from one
    # builder. The function stayed syntactically valid, imported fine, and returned None
    # -- failing only at call time, inside a subprocess, as an opaque TypeError.
    from bench.candidates import REGISTRY

    class Dummy:
        pass

    for name, spec in REGISTRY.items():
        assert spec.build(Dummy) is not None, f"{name}.build() returned None"
        assert spec.parent is None or spec.parent in REGISTRY, \
            f"{name} names parent {spec.parent!r} which is not registered"
