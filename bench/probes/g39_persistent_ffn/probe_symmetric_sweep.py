"""Sweep BOTH forms of the fused FFN block over the SAME arm grid, and separate the arms
that are genuinely persistent from the ones that are not.

F-03's sweep is asymmetric in two ways that both favour the candidate:

  * the FRONTIER is timed at one arm (its derived BM, 8 warps, min of 5 do_bench) while
    the PERSISTENT form is the minimum over 96 arms of min-of-3. Under unlockable clocks
    the minimum of 96 draws is systematically below a single draw, so part of the reported
    0.902-0.919x is the size of the search, not the kernel.

  * the winning persistent arms have `min(grid, ntiles) == ntiles`, i.e. ONE trip per
    program, so the grid-stride loop that is the entire mechanism never iterates.

This probe fixes both. It sweeps `BM x warps x stages` for the frontier and
`BM x warps x stages x grid` for the persistent form, reports best-of-N against
best-of-N at EQUAL N, and splits the persistent arms into `trips == 1` (no mechanism)
and `trips > 1` (the mechanism actually engaged). If the proposal is right, the best
`trips > 1` arm should beat the best frontier arm. If the best persistent arm is a
one-trip arm, the proposal has measured its own null.

INDICATIVE ONLY [L41].
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import triton

from bench.kernels.ffn_fused import _ffn_block_normed, fused_ffn_normed, launch_tile
from bench.probes.g39_persistent_ffn.probe_one_trip import (operands, persistent_ffn,
                                                            _bench)

PEAK = 88.2e12
RTOL, ATOL = 0.02, 0.002


def frontier_ffn(x, attn, n2w, n2b, w1, b1, w2, b2, nnw, nnb, eps,
                 block_m, num_warps, num_stages, store_next=True):
    """`fused_ffn_normed`, but with `num_stages` exposed.

    The frontier launcher does not pass `num_stages`, so it takes Triton's default. Three
    of F-03's four winning persistent arms differ from the frontier in num_stages AS WELL
    AS in the loop -- so if num_stages is what moves the number, the change is one keyword
    on the EXISTING launcher, not a persistent rewrite. This arm separates the two.
    """
    m, d = x.shape
    f = w1.shape[1]
    y = torch.empty((m, d), device=x.device, dtype=torch.float32)
    yn = torch.empty((m, d), device=x.device, dtype=torch.float16)
    h = _ffn_block_normed[(triton.cdiv(m, block_m),)](
        x, attn, n2w, n2b, w1, b1, w2, b2, nnw, nnb, y, yn, m, eps,
        D=d, F=f, BM=block_m, num_warps=num_warps, num_stages=num_stages,
        STORE_NEXT=store_next,
    )
    return y, yn, h


def main():
    from bench.gpu_lock import gpu_lock

    props = torch.cuda.get_device_properties(0)
    SM = props.multi_processor_count
    D = F = 128

    with gpu_lock("g39 symmetric sweep", timeout_s=7200):
        for tokens, cfgs in ((128, "cfg 2"), (2048, "cfg 4, 12"),
                             (8192, "cfg 1, 9, 10 -- declined by one_wave"),
                             (16384, "cfg 5 -- declined by one_wave")):
            x, attn, n2w, n2b, w1, b1, w2, b2 = operands(tokens, D, F)
            a = (x, attn, n2w, n2b, w1, b1, w2, b2, n2w, n2b, 1e-5)
            derived = launch_tile(tokens, SM)
            flops = 2 * 2 * tokens * D * F
            y_ref, _ = fused_ffn_normed(*a, block_m=derived, num_warps=8,
                                        store_next=True)
            torch.cuda.synchronize()

            print(f"\n{'='*94}\n{tokens} tokens  ({cfgs})   derived tile BM={derived}, "
                  f"grid={triton.cdiv(tokens, derived)}\n{'='*94}")

            # ------------------------------------------------ frontier, swept the same way
            # The SAME BM x warps x stages grid the persistent form gets, so best-of-N is
            # compared against best-of-N rather than best-of-96 against best-of-1.
            front = []
            for bm in (16, 32, 64, 128):
                if bm > tokens:
                    continue
                for w in (4, 8):
                    for st in (1, 2, 3, 4):
                        try:
                            fn = lambda bm=bm, w=w, st=st: frontier_ffn(
                                *a, block_m=bm, num_warps=w, num_stages=st,
                                store_next=True)[:2]
                            y, _ = fn()
                        except Exception:
                            continue
                        if not torch.allclose(y, y_ref, atol=ATOL, rtol=RTOL):
                            continue
                        torch.cuda.synchronize()
                        front.append((_bench(fn, reps=3), bm, w, st))
            front.sort()
            # And the arm the frontier ACTUALLY ships: derived BM, 8 warps, default stages.
            shipped = _bench(lambda: fused_ffn_normed(*a, block_m=derived, num_warps=8,
                                                      store_next=True), reps=3)

            # ------------------------------------------------ persistent, split by trips
            one_trip, multi = [], []
            for bm in (16, 32, 64, 128):
                if bm > tokens:
                    continue
                ntiles = triton.cdiv(tokens, bm)
                for w in (4, 8):
                    for st in (1, 2, 3, 4):
                        for gmul in (1, 2, 4):
                            grid = SM * gmul
                            progs = min(grid, ntiles)
                            trips = ntiles / progs
                            try:
                                fn = lambda bm=bm, w=w, st=st, g=grid: persistent_ffn(
                                    *a, block_m=bm, num_warps=w, grid=g, num_stages=st,
                                    store_next=True)[:2]
                                y, _ = fn()
                            except Exception:
                                continue
                            if not torch.allclose(y, y_ref, atol=ATOL, rtol=RTOL):
                                continue
                            torch.cuda.synchronize()
                            t = _bench(fn, reps=3)
                            (one_trip if trips <= 1.0 else multi).append(
                                (t, bm, w, st, grid, progs, trips))
            one_trip.sort()
            multi.sort()

            def show(label, row, extra=""):
                if not row:
                    print(f"  {label:<38} (no arm)")
                    return
                t = row[0]
                print(f"  {label:<38} {t*1e3:8.3f} us  "
                      f"{100*flops/(t*1e-3)/PEAK:5.1f}% peak   {extra}")

            print(f"  {'frontier AS SHIPPED (BM=%d w=8, default stages)' % derived:<38} "
                  f"{shipped*1e3:8.3f} us  "
                  f"{100*flops/(shipped*1e-3)/PEAK:5.1f}% peak")
            bf = front[0] if front else None
            show(f"frontier best of {len(front)} arms", bf,
                 f"BM={bf[1]} warps={bf[2]} stages={bf[3]}" if bf else "")
            ot = one_trip[0] if one_trip else None
            show(f"persistent best, ONE trip ({len(one_trip)})", ot,
                 f"BM={ot[1]} w={ot[2]} st={ot[3]} grid={ot[4]} "
                 f"progs={ot[5]}  <- NO MECHANISM" if ot else "")
            mt = multi[0] if multi else None
            show(f"persistent best, >1 trip ({len(multi)})", mt,
                 f"BM={mt[1]} w={mt[2]} st={mt[3]} grid={mt[4]} progs={mt[5]} "
                 f"trips={mt[6]:.2f}  <- mechanism engaged" if mt else "")

            if bf and mt:
                print(f"\n  MECHANISM RATIO (persistent>1trip / frontier-best) = "
                      f"{mt[0]/bf[0]:.4f}x  "
                      f"{'FASTER' if mt[0] < bf[0] else 'SLOWER'}")
            if bf and ot:
                print(f"  one-trip ratio (same geometry, loop runs once)     = "
                      f"{ot[0]/bf[0]:.4f}x")

            # equal-N control: how much does searching N arms buy on the frontier alone?
            if len(front) >= 2:
                print(f"  frontier spread across its {len(front)} arms: "
                      f"{front[0][0]*1e3:.3f} .. {front[-1][0]*1e3:.3f} us")


if __name__ == "__main__":
    main()
