"""Does `attn_single_tile.autotune_tile`'s TIMER explain the tile it picks?

THE QUESTION
------------
Finding 51 measured `single_tile(16, 4, 1)` at 1.929 us against the shipped `(64, 4, 1)`'s
2.476 on config 2 -- 1.283x, replicated -- L2-hot inside a captured graph. The tile that
ships is chosen by `autotune_tile`, which sweeps at prime time on the real shapes. So
either the tuner never saw `(16, 4, 1)`, or it saw it and ranked it behind, and only one
of those is a bug in the tuner.

`autotune_tile` ranks with `do_bench(warmup=10, rep=25)`: L2 FLUSHED between reps, one
launch per call, no graph. `attn_choice._time` -- written eighteen generations later for
the same decision -- ranks with `do_bench_cudagraph`, and its docstring says why [L53]:
the kernel is replayed inside a captured CUDA graph on a 6.29 MB QKV buffer that a 48 MB
L2 has already seen. Finding 48's headline was wrong by 2.24x across exactly this gap.

So this probe runs BOTH timers over the SAME arms, in the same process, at the same shape,
with the same trial budget, and reports what `autotune_tile`'s own decision rule would
return under each. If the flushed timer is the reason, the two columns disagree on config
2 and agree everywhere else -- and that agreement IS the blast radius of changing it.

INDICATIVE ONLY [L41]. It proposes a fix; it concludes nothing about a candidate.
Take the GPU lock before running it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch

from bench.gpu_lock import gpu_lock, contention_report
from bench.matrix import MATRIX
from bench.kernels import attn_single_tile as ast
from bench.kernels.attn_choice import _reference, probe_batch, RTOL, ATOL


def _flushed(fn, reps=2):
    import triton.testing as tt
    return min(tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
               for _ in range(reps))


def _hot(fn, reps=2):
    import triton.testing as tt
    return min(tt.do_bench_cudagraph(fn, rep=25, return_mode="min")
               for _ in range(reps))


def _decide(timed: dict, derived):
    """`autotune_tile`'s own decision rule, verbatim, over an arbitrary timing map."""
    if not timed:
        return None
    best, best_ms = min(timed.items(), key=lambda kv: kv[1])
    base_ms = timed.get(derived)
    if base_ms is None or best_ms < base_ms * (1.0 - ast.DECISIVE):
        return best
    return derived


def sweep(cfg, device="cuda") -> dict:
    props = torch.cuda.get_device_properties(device)
    heads, hd, s = cfg.heads, cfg.head_dim, cfg.seq_len
    tiles = ast.viable_tiles(s, hd, props.regs_per_multiprocessor,
                             props.max_threads_per_multi_processor, props.warp_size)
    if not tiles:
        return {"config_id": cfg.id, "declined": "no viable tile"}
    derived = ast.choose_tile(s, hd, props.regs_per_multiprocessor,
                              props.max_threads_per_multi_processor, props.warp_size)
    pb = probe_batch(cfg.batch_size, heads, props.multi_processor_count)
    dm = heads * hd
    probe_bytes = pb * s * 3 * dm * 2
    budget = int(props.total_memory / 64.0)
    if probe_bytes > budget:
        return {"config_id": cfg.id, "declined":
                f"probe {probe_bytes / 2**20:.0f} MiB > {budget / 2**20:.0f} MiB budget"}

    qkv = torch.randn(pb, s, 3 * dm, device=device, dtype=torch.float16)
    scale = hd ** -0.5
    ref = _reference(qkv, heads, hd)

    flushed, hot, wrong = {}, {}, []
    for bm, w, st in tiles:
        fn = (lambda bm=bm, w=w, st=st:
              ast.single_tile_attention(qkv, heads, hd, scale, bm, w, st))
        try:
            out = fn()
            torch.cuda.synchronize()
        except Exception:
            continue
        # CORRECTNESS BEFORE TIMING. `autotune_tile` does NOT do this today; its arms are
        # gated only by `fits`/`pays`. Reported so the gap is a measured number and not an
        # assertion either way.
        if not torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL):
            wrong.append((bm, w, st))
            continue
        try:
            flushed[(bm, w, st)] = _flushed(fn)
            hot[(bm, w, st)] = _hot(fn)
        except Exception:
            continue
    del qkv, ref
    torch.cuda.empty_cache()

    return {
        "config_id": cfg.id, "batch": cfg.batch_size, "probe_batch": pb,
        "heads": heads, "head_dim": hd, "seq_len": s,
        "derived": list(derived) if derived else None,
        "n_tiles": len(tiles),
        "incorrect_arms": [list(t) for t in wrong],
        "flushed_us": {str(k): v * 1e3 for k, v in flushed.items()},
        "hot_us": {str(k): v * 1e3 for k, v in hot.items()},
        "picks_flushed": list(_decide(flushed, derived) or ()),
        "picks_hot": list(_decide(hot, derived) or ()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-contended", action="store_true")
    args = ap.parse_args()

    why = contention_report()
    if why and not args.allow_contended:
        print(f"REFUSING: {why}")
        return 2

    rows = []
    with gpu_lock("g42 tile-timer regime probe", timeout_s=1800):
        torch.set_float32_matmul_precision("high")
        props = torch.cuda.get_device_properties("cuda")
        for cfg in MATRIX:
            ok, _ = ast.applies(cfg.seq_len, cfg.head_dim, props)
            if not ok:
                print(f"cfg {cfg.id:>2}  single-tile declines this shape", flush=True)
                continue
            r = sweep(cfg)
            rows.append(r)
            if "declined" in r:
                print(f"cfg {cfg.id:>2}  {r['declined']}", flush=True)
                continue
            pf, ph = tuple(r["picks_flushed"]), tuple(r["picks_hot"])
            fus = r["flushed_us"].get(str(pf))
            hus = r["hot_us"].get(str(ph))
            dh = r["hot_us"].get(str(tuple(r["derived"])))
            mark = "SAME" if pf == ph else "*** DIFFERS ***"
            gain = (dh / hus) if (dh and hus) else float("nan")
            print(f"cfg {cfg.id:>2}  B={r['batch']:>5} pb={r['probe_batch']:>3} "
                  f"hd={r['head_dim']:>3} S={r['seq_len']:>4}  "
                  f"derived={tuple(r['derived'])}  "
                  f"flushed->{pf} ({fus:.3f}us)  hot->{ph} ({hus:.3f}us)  "
                  f"hot gain over derived {gain:.3f}x  {mark}"
                  + (f"  WRONG ARMS: {r['incorrect_arms']}" if r["incorrect_arms"] else ""),
                  flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
