"""Where a persistent FFN block can and cannot have a mechanism, as arithmetic.

Proposal F-03 asks for `_ffn_block_normed` to be rewritten in persistent grid-stride form
-- grid clamped to a multiple of the SM count, the row tile turned into a loop -- so that
`w1`/`w2` are loaded once per program instead of once per row tile. It measured that form
at 0.918x-0.977x of the frontier at 128, 2048 and 8192 tokens and priced it at +0.0138 of
`weighted_score` on configs 2, 4 and 12.

The saving is proportional to `(trips per program - 1)`, and the launcher runs
`min(grid, ntiles)` programs. Three facts, in decreasing strength:

  1. AT THE DERIVED TILE THE LOOP NEVER RUNS. `one_wave` is True iff
     `ntiles <= sm_count * blocks_per_sm`, and both weight matrices are a 64 KB on-chip
     image, so `blocks_per_sm == 1` on every announced config that reaches the branch and
     `one_wave` degenerates to `ntiles <= sm_count`. F-03's grids are `sm_count * k`, so
     `min(grid, ntiles) == ntiles` and every program takes exactly one trip. This is the
     tile all four of its winning arms used, which is why its numbers carry no mechanism.

  2. ON CONFIGS 2 AND 3 THE LOOP CANNOT RUN AT ANY TILE. `MIN_TILE_ROWS` is 16 because
     sm_89's `mma.sync` is m16n8k16 -- a hardware floor. At 128 and 512 tokens even that
     floor yields 8 and 32 tiles against 66 SMs. No grid is smaller than the tile count.

  3. ON CONFIGS 4 AND 12 IT CAN, BUT ONLY AT HALF THE DERIVED TILE. At 2048 tokens BM=16
     gives 128 tiles on 66 SMs. F-03 measured BM=32 and so never sampled its own mechanism
     on the only two rows that have one. Measured correctly it is worth +0.0063 of
     `weighted_score`, not +0.0138 -- see the finding.

This is arithmetic over `bench.kernels.ffn_fused`'s own public functions, so it needs no
GPU and cannot rot silently the way a prose closure can [L40]. If a future card, width or
tile rule moves any of the three boundaries, the matching test fails and says which.

See `docs/findings/47-the-persistent-block-had-no-trips.md` and the two probes under
`bench/probes/g39_persistent_ffn/`.
"""
from __future__ import annotations

import math

import pytest

from bench.kernels.ffn_fused import (MAX_TILE_ROWS, MIN_TILE_ROWS, amortizes,
                                     blocks_per_sm, fits, launch_tile, one_wave,
                                     smem_bytes)
from bench.matrix import MATRIX

# The measured device (docs/00-mission.md; ledger/device.json). Named here as data so the
# arithmetic can be re-run for another card by changing these two numbers.
SM_COUNT = 66
SMEM_PER_SM = 101376          # 99 KB
SMEM_PER_BLOCK_OPTIN = 101376
ELEM = 2                      # fp16 weights

# F-03's own grid multipliers, from bench/probes/ffn_persistent.py.
GRID_MULTIPLIERS = (1, 2, 4)


def _fused_configs():
    """Every announced config whose shape reaches the `one_wave` branch of the launcher.

    Mirrors `v34_launch_bound._decide_launch` in order -- `fits`, then `amortizes`, then
    `one_wave` -- minus the one run-time gate it cannot evaluate here (a masked input).
    The `amortizes` gate is load-bearing rather than cosmetic: config 7 (d_model 32) has
    `blocks_per_sm == 8` and would satisfy `one_wave` with 128 tiles, but it is taken by
    the throughput branch before `one_wave` is ever asked.

    Selects {2, 3, 4, 12}, which is the set v34's own docstring names.
    """
    out = []
    for cfg in MATRIX:
        d, f, tokens = cfg.d_model, cfg.ffn_dim, cfg.tokens
        bm = launch_tile(tokens, SM_COUNT)
        if not fits(d, f, ELEM, bm, SMEM_PER_BLOCK_OPTIN):
            continue
        if amortizes(tokens, d, f, ELEM):
            continue
        if not one_wave(tokens, d, f, ELEM, bm, SM_COUNT, SMEM_PER_SM):
            continue
        out.append((cfg, bm))
    return out


def test_the_fused_block_holds_exactly_one_program_per_sm():
    """F-03's stated occupancy diagnosis, checked: 64 KB of weight image per program.

    This is the fact that makes the closure below unconditional rather than incidental --
    with `blocks_per_sm == 1`, `one_wave` degenerates to `ntiles <= sm_count`.
    """
    for cfg, bm in _fused_configs():
        per_sm = blocks_per_sm(cfg.d_model, cfg.ffn_dim, ELEM, bm, SMEM_PER_SM)
        assert per_sm == 1, (
            f"config {cfg.id}: blocks_per_sm={per_sm}; the closure argument assumes 1")
        assert smem_bytes(cfg.d_model, cfg.ffn_dim, ELEM, bm) > SMEM_PER_SM // 2


def test_some_announced_config_actually_dispatches_the_fused_block():
    """L36: prove the set under test is non-empty before asserting anything about it.

    A closure that holds vacuously because the filter matched nothing is not a closure.
    """
    got = _fused_configs()
    assert got, "no announced config reaches the one_wave branch -- the filter is wrong"
    assert {c.id for c, _ in got} == {2, 3, 4, 12}, (
        f"expected v34's stated set, got {sorted(c.id for c, _ in got)}")


@pytest.mark.parametrize("gmul", GRID_MULTIPLIERS)
def test_at_the_derived_tile_every_persistent_arm_takes_one_trip(gmul):
    """The load-bearing assertion: at the tile the launcher derives, the loop never runs.

    `persistent_ffn` launches `min(grid, ntiles)` programs over `ntiles` tiles, so trips
    per program is `ntiles / min(grid, ntiles)`. At `launch_tile`'s own answer, under
    `one_wave`, that is exactly 1.0 for every grid at or above the SM count -- which is
    every grid F-03 swept, and the tile every one of its winning arms used at 128, 2048
    and 8192 tokens.

    This is what makes F-03's reported 0.918x-0.977x carry no mechanism. It is NOT a
    claim that persistence is unreachable in general -- see the two tests below, which
    state exactly where it is and is not.
    """
    for cfg, bm in _fused_configs():
        ntiles = math.ceil(cfg.tokens / bm)
        grid = SM_COUNT * gmul
        trips = ntiles / min(grid, ntiles)
        assert trips == 1.0, (
            f"config {cfg.id} (tokens={cfg.tokens}, derived BM={bm}, ntiles={ntiles}) "
            f"takes {trips:.2f} trips at grid={grid}: F-03's arms were not vacuous after "
            f"all -- re-open it")


def _legal_tiles():
    bm = MIN_TILE_ROWS
    while bm <= MAX_TILE_ROWS:
        yield bm
        bm *= 2


def test_configs_2_and_3_cannot_be_persistent_at_any_legal_tile():
    """The strong half of the closure: two of the four rows have fewer tiles than SMs.

    `MIN_TILE_ROWS` is 16 because sm_89's `mma.sync` is m16n8k16 and `tl.dot` needs every
    dimension >= 16 -- so 16 rows per program is a hardware floor, not a tuning choice.
    At 128 and 512 tokens even that floor yields 8 and 32 tiles against 66 SMs, so no
    grid can be smaller than the tile count and no program can ever take a second trip.
    Measured: the symmetric sweep found ZERO persistent arms at 128 tokens.
    """
    for cfg, _ in _fused_configs():
        if cfg.tokens > MIN_TILE_ROWS * SM_COUNT:
            continue                                    # configs 4 and 12; see below
        for bm in _legal_tiles():
            ntiles = math.ceil(cfg.tokens / bm)
            assert ntiles <= SM_COUNT, (
                f"config {cfg.id} at BM={bm} has {ntiles} tiles on {SM_COUNT} SMs -- "
                f"persistence is reachable here after all")


def test_configs_4_and_12_reach_persistence_only_by_halving_the_derived_tile():
    """The weak half, stated so the finding's +0.0063 is not mistaken for zero.

    At 2048 tokens persistence IS reachable -- but only at BM=16, which is half the tile
    `launch_tile` derives. F-03 measured BM=32, so it never sampled its own mechanism on
    the only two rows where the mechanism exists.
    """
    reachable = {}
    for cfg, derived in _fused_configs():
        tiles = [bm for bm in _legal_tiles()
                 if math.ceil(cfg.tokens / bm) > SM_COUNT]
        if tiles:
            reachable[cfg.id] = (derived, tiles)
    assert set(reachable) == {4, 12}, f"expected only 4 and 12, got {sorted(reachable)}"
    for cid, (derived, tiles) in reachable.items():
        assert tiles == [MIN_TILE_ROWS], (
            f"config {cid}: expected persistence only at the MMA floor, got {tiles}")
        assert derived == 2 * MIN_TILE_ROWS, (
            f"config {cid}: derived tile {derived} is not twice the only persistent tile")


def test_the_only_persistent_token_counts_are_ones_the_predicate_declines():
    """The complement, stated as a prediction rather than left implicit.

    F-03's one genuinely persistent winning arm was at 16384 tokens (15.5 trips per
    program). That token count is config 5's, and `one_wave` declines it. So the
    mechanism engages only where the kernel is never dispatched -- which is the whole
    finding, and it is checked here rather than asserted in prose.
    """
    persistent_somewhere = []
    for cfg in MATRIX:
        d, f, tokens = cfg.d_model, cfg.ffn_dim, cfg.tokens
        bm = launch_tile(tokens, SM_COUNT)
        if not fits(d, f, ELEM, bm, SMEM_PER_BLOCK_OPTIN):
            continue
        if amortizes(tokens, d, f, ELEM):
            continue                               # taken by the throughput branch first
        ntiles = math.ceil(tokens / bm)
        if ntiles > SM_COUNT:                      # a grid of SM programs would loop
            persistent_somewhere.append(cfg.id)
            assert not one_wave(tokens, d, f, ELEM, bm, SM_COUNT, SMEM_PER_SM), (
                f"config {cfg.id} is BOTH persistent-capable and one_wave -- the two "
                f"predicates are supposed to be complements")
    assert persistent_somewhere, "expected at least one persistent-capable shape"
    # configs 1, 5, 9, 10, 11: 128 or 256 tiles against 66 SMs -- persistence would bite,
    # and `one_wave` declines every one of them.
    assert set(persistent_somewhere) == {1, 5, 9, 10, 11}, persistent_somewhere


def test_launch_tile_stays_inside_its_own_bounds():
    """A guard on the tile rule the arithmetic above depends on."""
    for cfg in MATRIX:
        bm = launch_tile(cfg.tokens, SM_COUNT)
        assert MIN_TILE_ROWS <= bm <= MAX_TILE_ROWS
        assert bm & (bm - 1) == 0, f"BM={bm} is not a power of two"
