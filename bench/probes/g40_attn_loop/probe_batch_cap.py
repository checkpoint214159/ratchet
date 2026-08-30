"""Does the tuner's PROBE BATCH pick the same winner the REAL batch would?

THE HAZARD, WHICH IS INHERITED AND NOT NEW
-------------------------------------------
`attn_single_tile.autotune_tile` caps the batch it tunes on at `4 * SM_count / heads`,
because per-program work stops depending on batch once the grid fills the machine and
timing config 6's 10000-row batch would allocate 983 MB to learn what 66 rows already say.
`attn_choice.autotune` reuses that cap verbatim so both forms are probed at one shape.

**But this candidate's predicate is a statement about the GRID, and the cap shrinks the
grid.** Config 6 runs at batch 10000 and is tuned at batch 66. If the looped form wins at
66 rows and loses at 10000, the tuner ships a regression into 83% of the matrix's wall
time -- on a config that is past the 3.0 cap and cannot pay for it.

Config 6 is also the one config that CANNOT be checked by `bench/abba.py`: the memory gate
declines co-residency there (finding 05's 410% spill, finding 49's caveat 1), which is why
the controller's sweeps run ids 1..5,7..13 and skip it. So this probe is the only
instrument that can see the question, and it answers it op-level, one model, under the
lock.

    for each shape: sweep both forms at the PROBE batch, then re-time the two winners at
    the REAL batch, and report whether the ranking survives.

INDICATIVE ONLY [L41]. Take the GPU lock.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import triton.testing as tt

from bench.kernels import attn_looped, attn_single_tile
from bench.kernels.attn_choice import _reference, _single_with_handle, probe_batch
from bench.kernels.attn_looped import looped_attention

ATOL, RTOL = 2e-3, 2e-2

SHAPES = (
    #  label                                 B      H  hd    S
    ("cfg  6  B=10000 H=4 hd=32 S=128",   10000,    4, 32, 128),
    ("cfg  5  B=128   H=4 hd=32 S=128",     128,    4, 32, 128),
    ("cfg  1  B=64    H=4 hd=32 S=128",      64,    4, 32, 128),
    ("cfg 11  B=64    H=16 hd=8 S=128",      64,   16,  8, 128),
    ("cfg  7  B=64    H=4 hd=8  S=128",      64,    4,  8, 128),
    ("cfg 13  B=64    H=4 hd=32 S=1024",     64,    4, 32, 1024),
)


def _hot(fn, reps=5):
    """L2 hot, launch amortized -- the regime the model runs the kernel in."""
    return min(tt.do_bench_cudagraph(fn, rep=50, return_mode="min") for _ in range(reps))


def best_at(B, H, hd, S, props, label):
    """Sweep both forms at batch B; return (best_single, best_looped) as (ms, tile)."""
    dm = H * hd
    g = torch.Generator(device="cuda").manual_seed(11)
    qkv = (torch.randn(B, S, 3 * dm, device="cuda", dtype=torch.float16,
                       generator=g) * 0.3)
    scale = hd ** -0.5
    ref = _reference(qkv, H, hd)

    single = attn_single_tile.viable_tiles(
        S, hd, props.regs_per_multiprocessor,
        props.max_threads_per_multi_processor, props.warp_size)
    looped = (attn_looped.viable_tiles(B, H, S, hd, props)
              if attn_looped.applies(B, H, S, hd, props)[0] else [])

    out = {}
    for form, tiles, call in (
            ("single_tile", single,
             lambda t: _single_with_handle(qkv, H, hd, scale, *t)),
            ("looped", looped,
             lambda t: looped_attention(qkv, H, hd, scale, *t, _return_handle=True))):
        best = None
        for t in tiles:
            try:
                o, h = call(t)
                torch.cuda.synchronize()
            except Exception:
                continue
            if not torch.allclose(o.float(), ref.float(), atol=ATOL, rtol=RTOL):
                continue
            if h.n_spills:
                continue
            ms = _hot(lambda t=t: call(t)[0])
            if best is None or ms < best[0]:
                best = (ms, t, h.n_regs, h.n_spills)
        out[form] = best
    out["sdpa"] = (_hot(lambda: _reference(qkv, H, hd)), (), 0, 0)
    del qkv, ref
    torch.cuda.empty_cache()
    return out


def time_tile(B, H, hd, S, form, tile):
    dm = H * hd
    g = torch.Generator(device="cuda").manual_seed(11)
    qkv = (torch.randn(B, S, 3 * dm, device="cuda", dtype=torch.float16,
                       generator=g) * 0.3)
    scale = hd ** -0.5
    if form == "sdpa":
        ms = _hot(lambda: _reference(qkv, H, hd))
    elif form == "single_tile":
        ms = _hot(lambda: _single_with_handle(qkv, H, hd, scale, *tile)[0])
    else:
        ms = _hot(lambda: looped_attention(qkv, H, hd, scale, *tile,
                                           _return_handle=True)[0])
    del qkv
    torch.cuda.empty_cache()
    return ms


def main() -> int:
    from bench.gpu_lock import gpu_lock

    props = torch.cuda.get_device_properties(0)
    with gpu_lock("g40 probe-batch-cap", timeout_s=14400):
        for label, B, H, hd, S in SHAPES:
            pb = probe_batch(B, H, props.multi_processor_count)
            print("=" * 96)
            print(f"{label}    real batch {B}, tuner probes at {pb}"
                  f"{'   (no cap)' if pb == B else ''}")
            print("=" * 96)
            at_probe = best_at(pb, H, hd, S, props, label)
            for form, row in at_probe.items():
                if row is None:
                    print(f"  probe batch {pb:<6} {form:<14} --")
                else:
                    print(f"  probe batch {pb:<6} {form:<14} {row[0]*1e3:9.3f} us  "
                          f"tile={row[1]}")

            winner = min((f for f, r in at_probe.items() if r is not None),
                         key=lambda f: at_probe[f][0])
            print(f"  -> the tuner would pick: {winner} {at_probe[winner][1]}")

            if pb == B:
                print("  (uncapped: the tuner tunes on the real batch)\n")
                continue

            print(f"  re-timed at the REAL batch {B}:")
            real = {}
            for form, row in at_probe.items():
                if row is None:
                    continue
                real[form] = time_tile(B, H, hd, S, form, row[1])
                print(f"  real  batch {B:<6} {form:<14} {real[form]*1e3:9.3f} us  "
                      f"tile={row[1]}")
            real_winner = min(real, key=lambda f: real[f])
            agree = "AGREES" if real_winner == winner else "DISAGREES -- the cap misleads"
            pen = real[winner] / real[real_winner]
            print(f"  -> at the real batch the best is: {real_winner}   {agree}"
                  f"   (cost of the tuner's choice: {pen:.3f}x)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
