"""Pick between the two attention tile shapes by TIMING THEM SYMMETRICALLY.

WHY THIS FILE EXISTS AND NOT AN `if head_dim == 64`
---------------------------------------------------
`attn_single_tile.autotune_tile` already sweeps the loop-free kernel's own legal grid at
prime time. This routine widens that sweep to include `attn_looped`'s legal grid and
returns whichever form and tile is fastest -- **with the same protocol, the same timer and
the same number of trials applied to both**.

That symmetry is the whole design constraint, and it is here because the project measured
what happens without it. Finding 47 measured a **4.5% best-of-N-against-best-of-1
handicap**: applying a challenger's selection protocol to the *incumbent* made the
incumbent 4.5% faster than itself. Finding 48 then committed exactly that error one file
later -- 180 arms swept for the looped form against 4 arms at a single warp count for the
single-tile form, on a shape whose best tile finding 31 had already recorded as 8 warps.
Re-run symmetrically, its headline moved from 1.206x to 1.200x and its *baseline* moved by
2.1x.

So: both forms are enumerated over their complete legal grid on the measured device, every
arm is checked for correctness before it is timed, every arm is timed by the same function
with the same repeat count, and the winner must clear `DECISIVE` against the incumbent's
DERIVED tile before it displaces anything.

    AMENDED AT GENERATION 42. "The same timer" was true WITHIN this file and false across
    the package: `attn_single_tile.autotune_tile`, the routine this one falls back to,
    had ranked with the L2-flushed `do_bench` since generation 23. So the symmetry held
    between the arms of one sweep and broke between the two sweeps -- and the flushed
    timer's 1.024 us quantum cannot resolve a 1.9 us kernel at all, which cost config 2 a
    1.28x tile on a table of ties (finding 53). The timer is now one function,
    `attn_single_tile.hot_time`, called by both routines; `_time` below is an alias for
    it. Correctness-before-timing likewise now holds in both, not only here.

`DECISIVE` IS INHERITED, NOT RELAXED
------------------------------------
`attn_single_tile.DECISIVE = 0.10` exists because these kernels run in 1-13 us against a
~1 us event timer and a project noise floor of +/-7% (L29). A challenger that wins by less
than 10% has not been shown to win. That bar is applied here unchanged and it is applied
to the *cross-form* comparison too -- the looped kernel does not get an easier test for
being new.

CORRECTNESS BEFORE TIMING, PER ARM
-----------------------------------
Every tile is checked against `F.scaled_dot_product_attention` at the locked tolerance
(2e-3 / 2e-2, never widened) on the probe shape before it is admitted to the timing set.
An arm that does not match is dropped, not reported -- a fast wrong kernel must not be
able to win a sweep.

SDPA IS A BAR, NOT AN ASSUMPTION AND NOT A SELECTABLE ARM
----------------------------------------------------------
The first draft of this file compared the two Triton forms to each other and to the
derived single tile. That is wrong wherever `attn_single_tile` DECLINES -- configs 8, 9
and 13 -- because there `derived` is None, and any looped tile that compiles would have
won the sweep by default while the thing it actually has to beat, `SDPA` plus the
head-major repack, was never timed. Finding 48 measured that the looped form beats SDPA
by 1.006x on config 8: a tie, which under the first draft would have shipped as a win.

So `sdpa+repack` is swept as a third arm, and a Triton form must clear `DECISIVE` against
the BETTER of the derived single tile and SDPA. But SDPA is never *selected*: where
nothing clears the bar, the shape keeps precisely what the parent ran. That bounds this
file's blast radius to "shapes where the looped kernel demonstrably wins", which is what
makes the candidate's A/B attributable and supplies its byte-identical control configs.
It also means a genuine second result -- SDPA beating the incumbent single-tile kernel on
config 10 in the hot regime -- is deliberately left on the table for a separate change
with its own evidence, rather than smuggled in on this one's measurement.

    AMENDED AT GENERATION 41. That separate change is `autotune_vendor` below, and the
    heading above now describes `autotune_looped` alone. The split is what keeps both
    A/Bs attributable: `autotune_looped` still cannot select SDPA, and a shape it
    declines still falls through to the parent's routine unchanged. The vendor became
    selectable only through a SECOND routine, which times exactly two arms -- the tile
    the parent already chose, and sdpa+repack -- and is asked only where the looped form
    has already lost. The g41 audit measured all three paths on all thirteen runnable
    configs, twice: the vendor wins on exactly one shape (config 10, 1.119x over
    `single_tile`), and the looped form beats it there by a further 1.099x.

THE PROBE ALLOCATION IS BOUNDED
--------------------------------
This sweep runs at prime time with the model already resident, and it allocates a real
`[probe_batch, S, 3*d_model]` tensor. At S=100000 that is 9.8 GiB on a 16 GiB card --
the tuner would OOM the model it is tuning. The budget is a fraction of the device's
MEASURED `total_memory`, and a shape over it declines the sweep rather than attempting
it (no config id appears; a card with more memory sweeps more shapes).
"""

from __future__ import annotations

import torch

from . import attn_looped, attn_single_tile
from .attn_single_tile import ATOL, DECISIVE, RTOL   # one definition, in the kernel module

# The tuning probe may allocate at most this fraction of the device's measured total
# memory. It runs with the model resident, so it must not be able to disturb -- let alone
# OOM -- the thing it is tuning. 1/64 of a 16 GiB card is 256 MiB, which admits every
# announced shape except seq_len 100000 (9.8 GiB), and admits that one on a big enough
# card without being retuned.
PROBE_MEMORY_FRACTION = 1.0 / 64.0


def _reference(qkv: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """What both forms must reproduce: SDPA plus the head-major repack.

    ONE definition, in `attn_single_tile`, next to the kernel whose contract it states.
    Kept as a module-level name here because probes and tests import it from this module.
    """
    return attn_single_tile.sdpa_reference(qkv, heads, head_dim)


def probe_batch(batch: int, heads: int, multi_processor_count: int) -> int:
    """`autotune_tile`'s cap, reused verbatim so both forms are probed at one shape.

    Per-program work does not depend on batch once the grid fills the machine, so timing
    config 6's 10000-row batch would allocate 983 MB to learn what 66 rows already say.

    CAVEAT, stated because it is a real limitation of this routine: `attn_looped.pays`
    is a statement about the grid, and the cap shrinks the grid. The predicate is
    therefore evaluated on the REAL batch (which is what will run) while the timing
    happens at the capped batch. Where the cap bites, the ranking is measured at fewer
    waves than the model will have -- which is conservative for the looped form, not
    generous to it.
    """
    return max(1, min(batch, 4 * multi_processor_count // max(1, heads)))


def _time(fn, reps: int) -> float:
    """Time an arm IN THE REGIME THE CALL SITE RUNS IN, which is L2-hot inside a graph.

    ONE TIMER FOR EVERY TUNER IN THIS PACKAGE. The body moved to
    `attn_single_tile.hot_time` at generation 42, when v23's `autotune_tile` -- which had
    been ranking with the L2-flushed `do_bench` since generation 23 -- was switched onto
    it. Sharing the function rather than the reasoning is the point: the two routines now
    rank on the same instrument by construction, which is the symmetry argument this
    module's docstring already makes about the ARMS, applied to the ROUTINES.

    The evidence for the choice is in `hot_time`'s docstring (the g42 quantization grids)
    and here (the g40 regime table):

        cfg 10   flushed 1.222x   hot 1.228x     same decision
        cfg  9   flushed 0.955x   hot 0.826x     same decision
        cfg  1   flushed 1.176x   hot 1.098x     DIFFERENT decision

    On config 1 the flushed timer clears the 10% `DECISIVE` margin and the hot timer does
    not. A tuner running the flushed timer would therefore switch config 1 -- a scoring
    row -- to a form its own steady state measures as inside the noise. The flushed
    numbers were also the less stable of the two: 1.176x and 1.375x across two runs of
    the identical sweep, against 1.098x/1.131x hot.

    Kept as a module-level name -- rather than importing `hot_time` directly at the call
    sites -- because `tests/bench/test_v41_vendor_aware_attn.py` monkeypatches it to
    drive `autotune_vendor` past a decision without a GPU sweep.
    """
    return attn_single_tile.hot_time(fn, reps)


def autotune_looped(seq_len: int, head_dim: int, heads: int, batch: int, device="cuda",
                    reps: int = 2, collect: list | None = None):
    """Return `(block_m, block_n, num_warps, num_stages), reason` -- or RAISE.

    The only thing this routine can say is "use the looped form with this tile". Every
    other outcome is a `ValueError`, and the caller answers it by running v23's
    `autotune_tile` unchanged, so a shape the looped kernel does not win is left exactly
    as the parent had it. See "the decision" below for why that asymmetry is deliberate.

    `collect`, if given, receives one row per timed arm: (form, tile, ms, n_regs,
    n_spills, smem), including the single-tile arms and sdpa. The sweep is symmetric even
    though the decision is not -- that is what lets a probe or a test check the ranking
    rather than trust it.
    """
    props = torch.cuda.get_device_properties(device)
    dm = heads * head_dim
    pb = probe_batch(batch, heads, props.multi_processor_count)

    # The probe tensor, before anything is compiled. A shape this cannot afford declines
    # the sweep entirely rather than OOMing the resident model.
    probe_bytes = pb * seq_len * 3 * dm * 2
    budget = int(props.total_memory * PROBE_MEMORY_FRACTION)
    if probe_bytes > budget:
        raise ValueError(
            f"tuning probe would allocate {probe_bytes/2**20:.0f} MiB against a "
            f"{budget/2**20:.0f} MiB budget ({PROBE_MEMORY_FRACTION:.4f} of "
            f"{props.total_memory/2**30:.1f} GiB); not sweeping")

    single_tiles = attn_single_tile.viable_tiles(
        seq_len, head_dim, props.regs_per_multiprocessor,
        props.max_threads_per_multi_processor, props.warp_size)
    # The predicate is asked about the REAL batch; the timing runs at the capped one.
    looped_tiles = (attn_looped.viable_tiles(batch, heads, seq_len, head_dim, props)
                    if attn_looped.applies(batch, heads, seq_len, head_dim, props)[0]
                    else [])
    if not single_tiles and not looped_tiles:
        raise ValueError("no viable tile in either form")

    derived = attn_single_tile.choose_tile(
        seq_len, head_dim, props.regs_per_multiprocessor,
        props.max_threads_per_multi_processor, props.warp_size)

    qkv = torch.randn(pb, seq_len, 3 * dm, device=device, dtype=torch.float16)
    scale = head_dim ** -0.5
    ref = _reference(qkv, heads, head_dim)

    timed: dict[tuple[str, tuple], float] = {}

    def _admit(form, tile, call):
        """Correctness first, then time. Identical protocol for both forms."""
        try:
            out, h = call()
            torch.cuda.synchronize()
        except Exception:
            return
        if not torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL):
            return
        if getattr(h, "n_spills", 0):
            return          # a spilling tile is never the answer; do not even time it
        try:
            ms = _time(lambda: call()[0], reps)
        except Exception:
            return
        timed[(form, tile)] = ms
        if collect is not None:
            collect.append((form, tile, ms, h.n_regs, h.n_spills,
                            h.metadata.shared))

    for bm, w, st in single_tiles:
        _admit("single_tile", (bm, w, st),
               lambda bm=bm, w=w, st=st: _single_with_handle(
                   qkv, heads, head_dim, scale, bm, w, st))
    for bm, bn, w, st in looped_tiles:
        _admit("looped", (bm, bn, w, st),
               lambda bm=bm, bn=bn, w=w, st=st: attn_looped.looped_attention(
                   qkv, heads, head_dim, scale, bm, bn, w, st, _return_handle=True))

    # THE THIRD ARM: what the model runs today. Where `attn_single_tile` applies that is
    # its derived tile; where it declines it is SDPA plus the head-major repack. Timed by
    # the same function with the same repeat count as every Triton arm.
    try:
        sdpa_ms = _time(lambda: _reference(qkv, heads, head_dim), reps)
        timed[("sdpa", ())] = sdpa_ms
        if collect is not None:
            collect.append(("sdpa", (), sdpa_ms, 0, 0, 0))
    except Exception:
        sdpa_ms = None

    del qkv, ref
    if not timed:
        raise ValueError("no tile in either form both matched and timed")

    n = (f"{len(single_tiles)} single-tile + {len(looped_tiles)} looped tiles + sdpa "
         f"at batch {pb}")

    # ------------------------------------------------------------------ the decision
    #
    # THIS ROUTINE MAY ONLY EVER SELECT THE LOOPED FORM. On any other outcome it raises,
    # and the caller falls back to v23's `autotune_tile` verbatim -- so every shape the
    # looped kernel does not win keeps the parent's plan BY CONSTRUCTION rather than by
    # coincidence. Two reasons, and the second was measured rather than anticipated:
    #
    #  1. It bounds the blast radius to "shapes where the looped kernel demonstrably
    #     wins", which is what makes the candidate's A/B attributable and what supplies
    #     its byte-identical control configs (finding 49's addendum).
    #
    #  2. The first draft did not do this, and it silently re-tuned the SINGLE-TILE form
    #     as a side effect of changing the timer. On config 2 it returned `(16, 2, 1)`
    #     where v38 runs `(64, 4, 1)`, because the hot timer sees that tile win by 1.124x
    #     and the flushed timer v23 uses does not. That may well be an improvement -- but
    #     it is a DIFFERENT improvement, it would have ridden into the ledger on this
    #     candidate's measurement, and it would have destroyed the control arm that
    #     measurement depends on. It is written up as a proposal instead.
    #
    # THE BAR IS THE INCUMBENT, NOT THE BEST ALTERNATIVE. `DECISIVE` asks "is the
    # improvement over the status quo bigger than the noise", so it is applied against
    # what the model runs today: the derived single tile, or sdpa where no tile is
    # viable. Applying it against `min(derived, sdpa)` instead -- which the first draft
    # did -- asks a different and much stricter question, and it declined config 10 at
    # 1.228x over the incumbent because that same arm is only 1.096x over sdpa.
    #
    # SDPA REMAINS A HARD GUARD, just not a 10% one: a Triton kernel slower than the
    # vendor call it replaces is never shipped, whatever its margin over our own
    # incumbent. That is the config-8 case finding 48 measured at 1.006x.
    looped_timed = {k: m for k, m in timed.items() if k[0] == "looped"}
    if not looped_timed:
        raise ValueError(f"no looped tile matched and timed; swept {n}")
    (_, tile), best_ms = min(looped_timed.items(), key=lambda kv: kv[1])

    derived_ms = timed.get(("single_tile", derived)) if derived else None
    incumbent_ms = derived_ms if derived_ms is not None else sdpa_ms
    if incumbent_ms is None:
        raise ValueError(f"no incumbent could be timed; swept {n}")
    incumbent_name = f"single_tile{derived}" if derived_ms is not None else "sdpa"

    if best_ms >= incumbent_ms * (1.0 - DECISIVE):
        raise ValueError(
            f"looped{tile} at {best_ms*1e3:.3f} us did not clear {incumbent_name} at "
            f"{incumbent_ms*1e3:.3f} us by {DECISIVE:.0%}; swept {n}")
    if sdpa_ms is not None and best_ms > sdpa_ms:
        raise ValueError(
            f"looped{tile} at {best_ms*1e3:.3f} us is slower than sdpa at "
            f"{sdpa_ms*1e3:.3f} us; swept {n}")

    why = (f"swept {n}: looped{tile} cleared {incumbent_name} decisively "
           f"({incumbent_ms / best_ms:.3f}x)")
    if sdpa_ms is not None:
        why += f", and is {sdpa_ms / best_ms:.3f}x sdpa"
    return tile, why


def autotune_vendor(seq_len: int, head_dim: int, heads: int, batch: int,
                    incumbent_tile: tuple[int, int, int], device="cuda",
                    reps: int = 2, collect: list | None = None) -> str:
    """Should this shape STEP ASIDE and run `sdpa+repack` instead? Reason, or RAISE.

    THE QUESTION THIS ANSWERS WAS PRE-REGISTERED AT GENERATION 23
    -------------------------------------------------------------
    `attn_single_tile`'s own source says of head_dim 64: *"the marginal case, sitting at
    exactly MIN_RESIDENT_BLOCKS... deliberately NOT implemented until a full sweep
    confirms the regression is real."* Finding 50 then measured, hot, `sdpa+repack` at
    9.987 us against the incumbent single-tile kernel's 11.189 on config 10 -- and
    deliberately did not act, because switching a shape to the vendor is a different
    change from adding a second Triton form and bundling them would have made v40's A/B
    unattributable.

    `attn_single_tile.pays()` is a residency argument, and a residency argument is a
    statement about whether the kernel CAN hide its latency -- not about whether the
    vendor is slower. Where the two disagree, only a timing can say. This routine is that
    timing, moved from a probe into the tuner, so the predicate stops being asserted on
    the nine shapes nobody had measured it on.

    TWO ARMS, ONE TRIAL BUDGET EACH -- WHICH IS HOW SDPA GETS AN EQUAL SWEEP
    ------------------------------------------------------------------------
    Finding 47 measured a 4.5% best-of-N-against-best-of-1 handicap and finding 48
    committed it. The handicap is a winner's curse: a `min` over many noisy arms of one
    parameterised family is biased low, and SDPA has no family to sweep -- it is one arm,
    and always will be.

    The equalisation is therefore not "sweep SDPA harder" (impossible) but **compare like
    for like**: this routine times exactly TWO arms -- the incumbent tile that
    `attn_single_tile.autotune_tile` already chose, and `sdpa+repack` -- with the same
    `_time`, the same `reps`, and one arm's budget each. Neither side takes a minimum the
    other does not. The looped form's sweep is a separate decision that has already run
    and already lost by the time this is called.

    That also means this routine never re-tunes the single-tile form. `incumbent_tile` is
    passed in, chosen by the parent's own routine, run unchanged -- the same structural
    guarantee `autotune_looped` has, and for the same reason: a shape this declines is
    byte-identical to the parent, so an A/B over it is attributable.

    `DECISIVE` IS THE INHERITED 10%, APPLIED IN THE INCUMBENT'S FAVOUR
    ------------------------------------------------------------------
    The vendor must beat what the model runs today by more than the margin below which
    the timer cannot separate two arms. Ties go to the incumbent, so this can only ever
    remove a kernel that is measurably losing -- never one that is merely level.
    """
    props = torch.cuda.get_device_properties(device)
    dm = heads * head_dim
    pb = probe_batch(batch, heads, props.multi_processor_count)

    probe_bytes = pb * seq_len * 3 * dm * 2
    budget = int(props.total_memory * PROBE_MEMORY_FRACTION)
    if probe_bytes > budget:
        raise ValueError(
            f"tuning probe would allocate {probe_bytes/2**20:.0f} MiB against a "
            f"{budget/2**20:.0f} MiB budget; not timing the vendor")

    qkv = torch.randn(pb, seq_len, 3 * dm, device=device, dtype=torch.float16)
    scale = head_dim ** -0.5
    ref = _reference(qkv, heads, head_dim)
    try:
        bm, w, st = incumbent_tile
        out, h = _single_with_handle(qkv, heads, head_dim, scale, bm, w, st)
        torch.cuda.synchronize()
        # Correctness before timing, even here: the incumbent is only the incumbent if
        # it is right, and an arm that does not match must never win or lose a sweep.
        if not torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL):
            raise ValueError(f"incumbent single_tile{incumbent_tile} did not match "
                             f"the reference; not ranking it")
        inc_ms = _time(lambda: _single_with_handle(
            qkv, heads, head_dim, scale, bm, w, st)[0], reps)
        sdpa_ms = _time(lambda: _reference(qkv, heads, head_dim), reps)
    finally:
        del qkv, ref

    if collect is not None:
        collect.append(("single_tile", tuple(incumbent_tile), inc_ms))
        collect.append(("sdpa", (), sdpa_ms))

    if sdpa_ms >= inc_ms * (1.0 - DECISIVE):
        raise ValueError(
            f"sdpa at {sdpa_ms*1e3:.3f} us did not clear single_tile{incumbent_tile} at "
            f"{inc_ms*1e3:.3f} us by {DECISIVE:.0%} at batch {pb}")
    return (f"sdpa+repack at {sdpa_ms*1e3:.3f} us beat single_tile{incumbent_tile} at "
            f"{inc_ms*1e3:.3f} us decisively ({inc_ms/sdpa_ms:.3f}x), two arms at "
            f"batch {pb}, one trial budget each")


def _single_with_handle(qkv, heads, head_dim, scale, bm, w, st):
    """`single_tile_attention` plus its `CompiledKernel`, so both forms report spills.

    Re-launches through the same JIT cache the launcher uses; the second launch is a
    cache hit and costs a dictionary lookup.
    """
    import triton
    from .attn_single_tile import _attn_single_tile, next_pow2, padded_head_dim
    b, s, _ = qkv.shape
    dm = heads * head_dim
    out = torch.empty((b, s, dm), device=qkv.device, dtype=qkv.dtype)
    h = _attn_single_tile[(triton.cdiv(s, bm), heads, b)](
        qkv, out, qkv.stride(0), qkv.stride(1), out.stride(0), out.stride(1), scale,
        S=s, DH=head_dim, DP=padded_head_dim(head_dim), DM=dm,
        BM=bm, BN=next_pow2(s), num_warps=w, num_stages=st)
    return out, h


def dispatch(form: str, tile, qkv, heads, head_dim, scale):
    """Run the chosen form. One call site, so the candidate's `_core` stays flat."""
    if form == "looped":
        bm, bn, w, st = tile
        return attn_looped.looped_attention(qkv, heads, head_dim, scale, bm, bn, w, st)
    bm, w, st = tile
    return attn_single_tile.single_tile_attention(qkv, heads, head_dim, scale, bm, w, st)
