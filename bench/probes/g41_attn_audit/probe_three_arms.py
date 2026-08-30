"""Which of THREE attention paths is fastest, per config, in the regime the model runs?

THE QUESTION, PRE-REGISTERED AT GENERATION 23 AND UNANSWERED SINCE
------------------------------------------------------------------
`bench/kernels/attn_single_tile.py` carries an OPEN QUESTION in its source:

    "The screen measured config 10 (head_dim 64) at -7.1% end to end -- the marginal
     case, sitting at exactly MIN_RESIDENT_BLOCKS, one pass, inside the +/-7% floor...
     It is deliberately NOT implemented until a full sweep confirms the regression is
     real."

Finding 50 answered it for config 10 and nowhere else: hot, `sdpa+repack` reads 9.987 us
against the incumbent single-tile kernel's 11.189. It declined to chase it because
switching a config to the vendor is a different change and bundling it would have made
v40's A/B unattributable.

So the predicate `attn_single_tile.applies()` is currently asserted, not measured, on
NINE more shapes. This probe measures all three paths on all thirteen runnable configs.

THE THREE ARMS
--------------
  single_tile   our loop-free kernel, swept over its complete legal grid
  looped        our g40 kernel, swept over its complete legal grid (where it applies)
  sdpa+repack   `F.scaled_dot_product_attention` plus the head-major repack -- the exact
                expression the frontier runs where both Triton forms decline, copied
                from `attn_choice._reference`

HOW THE ARMS ARE EQUALISED, SINCE SDPA HAS NO TILES TO SWEEP
--------------------------------------------------------------
Finding 47 measured a **4.5% best-of-N-against-best-of-1 handicap**: applying a
challenger's selection protocol to the incumbent made the incumbent 4.5% faster than
itself. That is a winner's curse -- a `min` over N noisy readings of near-identical arms
is biased low -- and it is not fixable by "sweeping SDPA harder", because SDPA has
exactly one arm. Three things are done instead, and all three numbers are printed:

  1. **Identical timer, identical repeat count, per arm.** Every arm (Triton or vendor)
     is timed by the same `_hot()` with the same `REPS`, so the per-arm reading is
     produced by the same instrument.

  2. **`sdpa best-of-1`** -- SDPA given exactly one arm's worth of budget, i.e. the same
     treatment as any single Triton tile. This is the number that is *fair per arm* and
     *unfair per form*, because the Triton forms then take a min over many arms.

  3. **`sdpa best-of-N`** -- SDPA re-timed `N` times, where `N` is the number of Triton
     arms that were admitted and timed for that shape, and the minimum kept. This grants
     the vendor exactly the same amount of minimum-taking the sweep grants our kernels.
     It is the number to rank on, and the gap between it and `best-of-1` IS the size of
     the winner's-curse term, measured rather than assumed.

A decision that flips between (2) and (3) is a decision made by the protocol and not by
the hardware, and is reported as such.

THE REGIME IS HOT, NOT FLUSHED
-------------------------------
[L53, finding 50]. The kernel runs L2-hot inside a replayed CUDA graph; `do_bench`
flushes L2 and pays a launch. Finding 50 measured the incumbent at 24.757 us flushed
against 11.04 us in the graph -- a 2.24x regime gap -- and finding 48's headline was wrong
by exactly that. The primary table here is `do_bench_cudagraph`. A flushed reading is
taken for the three *selected* arms only, as a cross-check against finding 31's table,
and the two are never divided into each other.

CORRECTNESS BEFORE TIMING, PER ARM. Locked tolerance 2e-3 / 2e-2, never widened. An arm
that does not match is dropped, not reported. Causal throughout (`is_causal=True`,
finding 32).

INDICATIVE ONLY [L41]: this probe proposes, it does not conclude. The conclusion is an
end-to-end A/B. Take the GPU lock.

    python3 bench/probes/g41_attn_audit/probe_three_arms.py [--ids 1 2 ...] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import triton.testing as tt

from bench.kernels import attn_looped, attn_single_tile
from bench.kernels.attn_choice import _reference, _single_with_handle
from bench.kernels.attn_looped import looped_attention
from bench.matrix import MATRIX

ATOL, RTOL = 2e-3, 2e-2          # the locked tolerance. Never widened.
REPS = 3                          # identical for every arm, Triton or vendor

# The probe tensor budget. Only config 6 (B=10000 -> 983 MiB of QKV) exceeds it; that
# shape is timed at a capped batch and then the SELECTED arms are re-checked at the real
# batch, so the cap is visible rather than assumed harmless.
PROBE_BYTES_BUDGET = 512 * 2 ** 20


def _hot(fn, reps: int = REPS) -> float:
    """L2 hot, launch amortized inside a graph -- the regime the model runs in."""
    return min(tt.do_bench_cudagraph(fn, rep=25, return_mode="min") for _ in range(reps))


def _flushed(fn, reps: int = REPS) -> float:
    """L2 flushed, launch paid -- finding 48's regime. Cross-check only."""
    return min(tt.do_bench(fn, warmup=10, rep=25, return_mode="min") for _ in range(reps))


def _make_qkv(b, s, dm, seed=11):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16,
                        generator=g) * 0.3)


def _arms(B, H, hd, S, props):
    """(form, tile, call) for every legal Triton arm on this device, both forms."""
    single = attn_single_tile.viable_tiles(
        S, hd, props.regs_per_multiprocessor,
        props.max_threads_per_multi_processor, props.warp_size)
    looped = (attn_looped.viable_tiles(B, H, S, hd, props)
              if attn_looped.applies(B, H, S, hd, props)[0] else [])
    return single, looped


def sweep(cfg, props, batch: int, full_grid: bool = True, restrict=None):
    """Time every admitted arm hot. Returns (rows, meta).

    `restrict`, if given, is {form: [tiles]} and limits the sweep -- used for the
    real-batch re-check of a shape whose sweep ran capped.
    """
    B, H, hd, S = batch, cfg.heads, cfg.head_dim, cfg.seq_len
    dm = H * hd
    qkv = _make_qkv(B, S, dm)
    scale = hd ** -0.5
    ref = _reference(qkv, H, hd)

    single, looped = _arms(B, H, hd, S, props)
    if restrict is not None:
        single = [t for t in single if t in restrict.get("single_tile", [])]
        looped = [t for t in looped if t in restrict.get("looped", [])]

    calls = []
    for bm, w, st in single:
        calls.append(("single_tile", (bm, w, st),
                      lambda bm=bm, w=w, st=st: _single_with_handle(
                          qkv, H, hd, scale, bm, w, st)))
    for bm, bn, w, st in looped:
        calls.append(("looped", (bm, bn, w, st),
                      lambda bm=bm, bn=bn, w=w, st=st: looped_attention(
                          qkv, H, hd, scale, bm, bn, w, st, _return_handle=True)))

    rows, n_admitted = [], 0
    for form, tile, call in calls:
        try:
            out, h = call()
            torch.cuda.synchronize()
        except Exception as exc:
            rows.append({"form": form, "tile": tile, "status": f"compile/run: {exc}"[:90]})
            continue
        if not torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL):
            rows.append({"form": form, "tile": tile, "status": "WRONG -- dropped"})
            continue
        if getattr(h, "n_spills", 0):
            rows.append({"form": form, "tile": tile, "status": "spills -- dropped",
                         "n_spills": int(h.n_spills)})
            continue
        try:
            ms = _hot(lambda: call()[0])
        except Exception as exc:
            rows.append({"form": form, "tile": tile, "status": f"timer: {exc}"[:90]})
            continue
        n_admitted += 1
        rows.append({"form": form, "tile": tile, "status": "ok", "hot_ms": ms,
                     "n_regs": int(h.n_regs), "n_spills": int(h.n_spills),
                     "smem": int(h.metadata.shared)})

    # ------------------------------------------------------------------ the vendor arm
    # best-of-1: one arm's worth of budget, exactly what each Triton tile got.
    sdpa_1 = _hot(lambda: _reference(qkv, H, hd))
    # best-of-N: the same amount of minimum-taking the SWEEP grants our kernels.
    sdpa_n = sdpa_1
    for _ in range(max(0, n_admitted - 1)):
        sdpa_n = min(sdpa_n, _hot(lambda: _reference(qkv, H, hd)))
    rows.append({"form": "sdpa", "tile": (), "status": "ok", "hot_ms": sdpa_n,
                 "hot_ms_best_of_1": sdpa_1, "n_trials": max(1, n_admitted),
                 "n_regs": 0, "n_spills": 0, "smem": 0})

    meta = {"batch_probed": B, "n_single_legal": len(single),
            "n_looped_legal": len(looped), "n_admitted": n_admitted,
            "sdpa_best_of_1_ms": sdpa_1, "sdpa_best_of_N_ms": sdpa_n}
    del qkv, ref
    torch.cuda.empty_cache()
    return rows, meta


def best_of(rows, form):
    ok = [r for r in rows if r["form"] == form and r["status"] == "ok"]
    return min(ok, key=lambda r: r["hot_ms"]) if ok else None


def main() -> int:
    from bench.gpu_lock import gpu_lock

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+",
                    default=[c.id for c in MATRIX if c.seq_len <= 4096])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    out = {"device": props.name, "sms": props.multi_processor_count,
           "reps": REPS, "configs": {}}

    with gpu_lock("g41 three-arm attention audit", timeout_s=21600):
        print(f"device {props.name}  SMs={props.multi_processor_count}  "
              f"L2={props.L2_cache_size/2**20:.0f} MB  REPS={REPS}\n")
        for cfg in MATRIX:
            if cfg.id not in a.ids:
                continue
            dm = cfg.d_model
            want = cfg.batch_size * cfg.seq_len * 3 * dm * 2
            batch = cfg.batch_size
            capped = False
            if want > PROBE_BYTES_BUDGET:
                batch = max(1, PROBE_BYTES_BUDGET // (cfg.seq_len * 3 * dm * 2))
                capped = True

            rows, meta = sweep(cfg, props, batch)
            meta["capped"] = capped
            meta["real_batch"] = cfg.batch_size

            # What the SHIPPED chooser decides today, on the real batch.
            try:
                v38_tile, v38_why = attn_single_tile.autotune_tile(
                    cfg.seq_len, cfg.head_dim, cfg.heads, cfg.batch_size)
                v38_plan = f"single_tile{v38_tile}"
            except Exception as exc:
                v38_tile, v38_plan, v38_why = None, "sdpa", str(exc)
            meta["v38_plan"], meta["v38_why"] = v38_plan, v38_why

            print("=" * 104)
            print(f"config {cfg.id:>2}   B={cfg.batch_size} H={cfg.heads} "
                  f"hd={cfg.head_dim} S={cfg.seq_len}"
                  + (f"   [swept at capped batch {batch}]" if capped else ""))
            print(f"  legal arms: {meta['n_single_legal']} single_tile, "
                  f"{meta['n_looped_legal']} looped, 1 sdpa   "
                  f"|  admitted+timed: {meta['n_admitted']} Triton arms")
            print(f"  shipped v38 plan: {v38_plan}")
            print("=" * 104)
            timed = [r for r in rows if r["status"] == "ok"]
            for r in sorted(timed, key=lambda r: r["hot_ms"]):
                extra = ""
                if r["form"] == "sdpa":
                    extra = (f"   best-of-1 {r['hot_ms_best_of_1']*1e3:.3f}"
                             f"  best-of-{r['n_trials']} {r['hot_ms']*1e3:.3f}")
                print(f"    {r['form'] + str(r['tile']):<30}"
                      f"{r['hot_ms']*1e3:>10.3f} us"
                      f"  regs={r['n_regs']:>4} spill={r['n_spills']} "
                      f"smem={r['smem']}{extra}")
            dropped = [r for r in rows if r["status"] != "ok"]
            if dropped:
                print(f"    ({len(dropped)} arms dropped: "
                      f"{sorted(set(r['status'].split(':')[0] for r in dropped))})")

            bs, bl = best_of(rows, "single_tile"), best_of(rows, "looped")
            sd = best_of(rows, "sdpa")
            meta["best_single_tile"] = bs and {"tile": bs["tile"], "ms": bs["hot_ms"]}
            meta["best_looped"] = bl and {"tile": bl["tile"], "ms": bl["hot_ms"]}

            # The incumbent is what the model runs TODAY: v38's derived/tuned single tile
            # where one exists, otherwise sdpa.
            inc = None
            if v38_tile is not None:
                inc = next((r for r in timed if r["form"] == "single_tile"
                            and r["tile"] == tuple(v38_tile)), None)
            if inc is None:
                inc = sd
            meta["incumbent_ms"] = inc["hot_ms"]
            meta["incumbent"] = inc["form"] + str(inc["tile"])

            print(f"\n  {'what the model runs today':<34}"
                  f"{inc['form'] + str(inc['tile']):<24}{inc['hot_ms']*1e3:>9.3f} us")
            for label, r in (("best single_tile", bs), ("best looped", bl),
                             ("sdpa+repack (best-of-N)", sd)):
                if r is None:
                    print(f"  {label:<34}{'--':<24}")
                    continue
                print(f"  {label:<34}{str(r['tile']):<24}{r['hot_ms']*1e3:>9.3f} us"
                      f"   {inc['hot_ms']/r['hot_ms']:>7.3f}x incumbent")
            if sd is not None:
                r1 = sd["hot_ms_best_of_1"]
                print(f"  {'sdpa+repack (best-of-1)':<34}{'':<24}{r1*1e3:>9.3f} us"
                      f"   {inc['hot_ms']/r1:>7.3f}x incumbent"
                      f"   [winner's-curse term "
                      f"{(r1/sd['hot_ms'] - 1)*100:.2f}%]")

            # Flushed cross-check on the selected arms only.
            fl = {}
            try:
                qkv = _make_qkv(batch, cfg.seq_len, dm)
                scale = cfg.head_dim ** -0.5
                if bs:
                    bm, w, st = bs["tile"]
                    fl["single_tile"] = _flushed(
                        lambda: _single_with_handle(qkv, cfg.heads, cfg.head_dim,
                                                    scale, bm, w, st)[0])
                if bl:
                    bm, bn, w, st = bl["tile"]
                    fl["looped"] = _flushed(
                        lambda: looped_attention(qkv, cfg.heads, cfg.head_dim, scale,
                                                 bm, bn, w, st))
                fl["sdpa"] = _flushed(lambda: _reference(qkv, cfg.heads, cfg.head_dim))
                del qkv
                torch.cuda.empty_cache()
            except Exception as exc:
                fl["error"] = str(exc)[:90]
            meta["flushed_ms"] = fl
            print("  flushed cross-check (never divided into a hot number): "
                  + "  ".join(f"{k}={v*1e3:.3f}" for k, v in fl.items()
                              if isinstance(v, float)))

            # Real-batch re-check where the sweep ran capped.
            if capped:
                keep = {"single_tile": [bs["tile"]] if bs else [],
                        "looped": [bl["tile"]] if bl else []}
                try:
                    rrows, rmeta = sweep(cfg, props, cfg.batch_size, restrict=keep)
                    meta["real_batch_recheck"] = {
                        r["form"] + str(r["tile"]): r.get("hot_ms")
                        for r in rrows if r["status"] == "ok"}
                    print("  REAL-BATCH re-check of the selected arms: "
                          + "  ".join(f"{k}={v*1e3:.3f}"
                                      for k, v in meta["real_batch_recheck"].items()))
                except Exception as exc:
                    meta["real_batch_recheck"] = {"error": str(exc)[:120]}
                    print(f"  REAL-BATCH re-check unavailable: {str(exc)[:100]}")
            print()
            out["configs"][str(cfg.id)] = {"meta": meta, "rows": rows}

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
