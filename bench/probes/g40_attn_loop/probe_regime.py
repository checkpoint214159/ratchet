"""Does the looped form's 1.200x SURVIVE the cache regime the model actually runs in?

THE QUESTION THE CENSUS RAISED
-------------------------------
`probe_census.py` measures config 10's attention at **44.14 us/fwd across 4 calls = 11.04
us per call** inside the replayed CUDA graph. Finding 48 measured the same shape's
incumbent at **24.757 us per call** under `do_bench`. That is a **2.24x regime gap**, and
it is not noise -- it is the difference between the two timers:

    do_bench            flushes L2 between reps and pays a kernel launch per call.
                        Q/K/V for this shape is 6.29 MB and comes from HBM every time.
    the model           runs the kernel L2-hot (this card has 48 MB of L2) inside a
                        captured graph, so the launch is amortized and the operands are
                        already resident.

**Finding 48 priced its proposal by multiplying an in-graph time by an L2-flushed ratio.**
[L33]: a mechanism measured in isolation measures the isolation. The looped form's claimed
advantage is *pipelining* -- overlapping memory latency with compute across loop trips --
and memory latency is exactly what changes between those two regimes. If the operands are
already in L2 there may be far less latency left to hide, in which case the loop is paying
for online-softmax bookkeeping it cannot earn back.

So this probe times both forms in BOTH regimes, symmetrically:

    do_bench             L2 flushed, launch paid       -- finding 48's regime
    do_bench_cudagraph   L2 hot, launch amortized      -- the model's regime

CLAUDE.md forbids comparing a `do_bench` number to a `do_bench_cudagraph` number, and this
probe never does. It compares ARMS WITHIN a regime and reports the two ratios side by side.

SYMMETRY, BECAUSE THE PROJECT HAS PAID FOR ITS ABSENCE TWICE
-------------------------------------------------------------
Finding 47 measured a 4.5% best-of-N-against-best-of-1 handicap; finding 48 then committed
it. Here **both forms are swept over their complete legal grid on this device**, with the
same timer, the same repeat count and the same correctness gate, and `sdpa+repack` is
timed as a third arm. Arm counts are printed so the asymmetry, if any, is visible.

INDICATIVE ONLY [L41]. Take the GPU lock.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F
import triton
import triton.testing as tt

from bench.kernels import attn_looped, attn_single_tile
from bench.kernels.attn_choice import _reference, _single_with_handle
from bench.kernels.attn_looped import looped_attention

ATOL, RTOL = 2e-3, 2e-2

SHAPES = (
    #  label                      B   H   hd    S
    ("cfg 10  H=2 hd=64  S=128", 64,  2,  64, 128),
    ("cfg  9  H=1 hd=128 S=128", 64,  1, 128, 128),
    ("cfg  1  H=4 hd=32  S=128", 64,  4,  32, 128),
)


def _flushed(fn, reps=5):
    """L2 flushed, launch paid -- finding 48's regime."""
    return min(tt.do_bench(fn, warmup=25, rep=50, return_mode="min") for _ in range(reps))


def _hot(fn, reps=5):
    """L2 hot, launch amortized inside a graph -- the model's regime."""
    return min(tt.do_bench_cudagraph(fn, rep=50, return_mode="min") for _ in range(reps))


def sweep(B, H, hd, S, props):
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
    derived = attn_single_tile.choose_tile(
        S, hd, props.regs_per_multiprocessor,
        props.max_threads_per_multi_processor, props.warp_size)

    arms = []       # (form, tile, call)
    for bm, w, st in single:
        arms.append(("single_tile", (bm, w, st),
                     lambda bm=bm, w=w, st=st: _single_with_handle(
                         qkv, H, hd, scale, bm, w, st)))
    for bm, bn, w, st in looped:
        arms.append(("looped", (bm, bn, w, st),
                     lambda bm=bm, bn=bn, w=w, st=st: looped_attention(
                         qkv, H, hd, scale, bm, bn, w, st, _return_handle=True)))

    rows = []
    for form, tile, call in arms:
        try:
            out, h = call()
            torch.cuda.synchronize()
        except Exception:
            continue
        if not torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL):
            continue                              # correctness before timing
        f = _flushed(lambda: call()[0])
        ho = _hot(lambda: call()[0])
        rows.append((form, tile, f, ho, h.n_regs, h.n_spills, h.metadata.shared))

    s_f = _flushed(lambda: _reference(qkv, H, hd))
    s_h = _hot(lambda: _reference(qkv, H, hd))
    rows.append(("sdpa+repack", (), s_f, s_h, 0, 0, 0))
    del qkv, ref
    return rows, derived, len(single), len(looped)


def main() -> int:
    from bench.gpu_lock import gpu_lock

    props = torch.cuda.get_device_properties(0)
    with gpu_lock("g40 regime probe", timeout_s=14400):
        print(f"device: {props.name}  SMs={props.multi_processor_count}  "
              f"L2={props.L2_cache_size/2**20:.0f} MB\n")
        for label, B, H, hd, S in SHAPES:
            rows, derived, n_s, n_l = sweep(B, H, hd, S, props)
            print("=" * 100)
            print(f"{label}   B={B}  grid at BM=128 -> "
                  f"{attn_looped.grid_ctas(B, H, S, 128)} CTAs on "
                  f"{props.multi_processor_count} SMs")
            print(f"  arms swept: {n_s} single-tile, {n_l} looped, 1 sdpa   "
                  f"(derived single tile: {derived})")
            print("=" * 100)

            best = {}
            for form, tile, f, h, *_ in rows:
                for regime, t in (("flushed", f), ("hot", h)):
                    k = (form, regime)
                    if k not in best or t < best[k][0]:
                        best[k] = (t, tile)

            print(f"  {'arm':<34}{'flushed us':>12}{'hot us':>10}"
                  f"{'regs':>7}{'spill':>7}{'smem':>8}")
            for regime in ("flushed", "hot"):
                pass
            for form, tile, f, h, nr, ns, sm in sorted(rows, key=lambda r: r[3]):
                mark = ""
                if best.get((form, "hot"), (None,))[1] == tile:
                    mark = "  <- best hot for its form"
                print(f"    {form + str(tile):<32}{f*1e3:>12.3f}{h*1e3:>10.3f}"
                      f"{nr:>7}{ns:>7}{sm:>8}{mark}")

            # The incumbent is what the model runs: the derived single tile where one
            # exists, otherwise sdpa+repack.
            inc_form = "single_tile" if derived else "sdpa+repack"
            print(f"\n  {'regime':<12}{'incumbent':>14}{'best looped':>14}"
                  f"{'ratio':>10}   verdict")
            for regime in ("flushed", "hot"):
                if derived:
                    inc = next((r[2] if regime == "flushed" else r[3]) for r in rows
                               if r[0] == "single_tile" and r[1] == derived)
                else:
                    inc = next((r[2] if regime == "flushed" else r[3]) for r in rows
                               if r[0] == "sdpa+repack")
                lp = best.get(("looped", regime))
                if lp is None:
                    print(f"  {regime:<12}{inc*1e3:>14.3f}{'--':>14}")
                    continue
                ratio = inc / lp[0]
                v = ("clears DECISIVE" if ratio >= 1.0 + attn_single_tile.DECISIVE
                     else "inside DECISIVE -- no win")
                print(f"  {regime:<12}{inc*1e3:>14.3f}{lp[0]*1e3:>14.3f}"
                      f"{ratio:>9.3f}x   {v}   (incumbent = {inc_form})")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
