"""Does F-03's persistent FFN block have a mechanism at the token counts it claims?

F-03 reports the persistent grid-stride form of `_ffn_block_normed` at 0.977x / 0.918x /
0.919x / 0.902x of the frontier at 128 / 2048 / 8192 / 16384 tokens, and prices the win at
+0.0138 of weighted_score on configs 2, 4 and 12.

TWO THINGS ARE WRONG WITH THAT NUMBER, AND THIS PROBE MEASURES BOTH.

(1) THE MECHANISM IS ABSENT AT THREE OF THE FOUR TOKEN COUNTS.
    The proposal's mechanism is "load w1/w2 ONCE per program instead of once per row
    tile". That saving is proportional to (trips per program - 1). The probe launches
    `min(grid, ntiles)` programs, and at its own winning arms:

        tokens   frontier BM/grid   persistent best        programs   trips/program
           128        16 /   8      BM=16 grid=66                 8           1.00
          2048        32 /  64      BM=32 grid=66                64           1.00
          8192        64 / 128      BM=64 grid=132              128           1.00
         16384        64 / 256      BM=16 grid=66                66          15.52

    At 128, 2048 and 8192 tokens the winning "persistent" arm is the frontier's own launch
    geometry with a loop that runs exactly once. Same BM, same program count, same weight
    load per program. There is nothing hoisted, because there is no second trip to hoist
    it out of. The only genuinely persistent winner is 16384 tokens -- which is config 5,
    where `one_wave` declines and the fused block is never dispatched.

(2) THE 8-10% IS A SELECTION ARTIFACT.
    The probe times the frontier ONCE (min of 5 do_bench) and the persistent form as the
    MINIMUM OVER 96 ARMS (4 BM x 2 warps x 4 stages x 3 grid multipliers), each min of 3.
    Under unlockable clocks the minimum of 96 noisy draws sits systematically below a
    single draw. Arm C below applies the probe's own candidate-side protocol to the
    FRONTIER kernel -- 96 repeats of the identical call, min of 3 each, take the min --
    and reports how far that falls below the frontier's own single min-of-5. If it falls
    by roughly the reported 8-10%, the effect is the search, not the kernel.

PROTOCOL. Arms are interleaved ABBA within each round, the cold round is discarded, and
the minimum over the remaining rounds is kept -- bench/abba.py's discipline applied at op
level. Outputs are checked against the frontier kernel's own output at the locked
tolerance before any timing runs. n_regs / n_spills / smem are read off the compiled
kernel for every arm.

INDICATIVE ONLY [L41]. This probe proposes; it does not conclude. Nothing here reaches the
ledger.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import triton
import triton.language as tl

from bench.kernels.ffn_fused import fused_ffn_normed, launch_tile, smem_bytes

PEAK = 88.2e12
RTOL, ATOL = 0.02, 0.002


@triton.jit
def _ffn_block_persistent(
    X, ATTN, N2W, N2B, W1, B1, W2, B2, NNW, NNB, Y, YN,
    M, EPS, NTILES,
    D: tl.constexpr, F: tl.constexpr, BM: tl.constexpr,
    STORE_NEXT: tl.constexpr,
):
    """F-03's kernel, transcribed verbatim from probes/ffn_persistent.py.

    Identical arithmetic to `_ffn_block_normed`; the row tile is a grid-stride loop and
    w1/w2/biases/norm parameters are hoisted above it.
    """
    rd = tl.arange(0, D)
    rf = tl.arange(0, F)
    w1 = tl.load(W1 + rd[:, None] * F + rf[None, :])
    w2 = tl.load(W2 + rf[:, None] * D + rd[None, :])
    b1 = tl.load(B1 + rf)[None, :].to(tl.float32)
    b2 = tl.load(B2 + rd)[None, :].to(tl.float32)
    n2w = tl.load(N2W + rd)[None, :].to(tl.float32)
    n2b = tl.load(N2B + rd)[None, :].to(tl.float32)
    nnw = tl.load(NNW + rd)[None, :].to(tl.float32)
    nnb = tl.load(NNB + rd)[None, :].to(tl.float32)

    for t in range(tl.program_id(0), NTILES, tl.num_programs(0)):
        rm = t * BM + tl.arange(0, BM)
        keep = rm < M
        off = rm[:, None] * D + rd[None, :]
        res = (tl.load(X + off, mask=keep[:, None], other=0.0).to(tl.float32)
               + tl.load(ATTN + off, mask=keep[:, None], other=0.0).to(tl.float32))
        mu = tl.sum(res, axis=1) / D
        d0 = res - mu[:, None]
        var = tl.sum(d0 * d0, axis=1) / D
        xn = d0 * tl.rsqrt(var[:, None] + EPS) * n2w + n2b
        h = tl.dot(xn.to(w1.dtype), w1, out_dtype=tl.float32) + b1
        h = h * 0.5 * (1.0 + tl.erf(h * 0.70710678118654752440))
        y = res + tl.dot(h.to(w2.dtype), w2, out_dtype=tl.float32) + b2
        tl.store(Y + off, y, mask=keep[:, None])
        if STORE_NEXT:
            mu2 = tl.sum(y, axis=1) / D
            d2 = y - mu2[:, None]
            var2 = tl.sum(d2 * d2, axis=1) / D
            yn = d2 * tl.rsqrt(var2[:, None] + EPS) * nnw + nnb
            tl.store(YN + off, yn.to(YN.dtype.element_ty), mask=keep[:, None])


def persistent_ffn(x, attn, n2w, n2b, w1, b1, w2, b2, nnw, nnb, eps,
                   block_m, num_warps, grid, num_stages=1, store_next=True):
    m, d = x.shape
    f = w1.shape[1]
    y = torch.empty((m, d), device=x.device, dtype=torch.float32)
    yn = torch.empty((m, d), device=x.device, dtype=torch.float16)
    ntiles = triton.cdiv(m, block_m)
    h = _ffn_block_persistent[(min(grid, ntiles),)](
        x, attn, n2w, n2b, w1, b1, w2, b2, nnw, nnb, y, yn, m, eps, ntiles,
        D=d, F=f, BM=block_m, num_warps=num_warps, num_stages=num_stages,
        STORE_NEXT=store_next,
    )
    return y, yn, h


def operands(tokens, D=128, F=128):
    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(7)
    return (
        torch.randn(tokens, D, device=dev, dtype=torch.float32, generator=g),      # x
        torch.randn(tokens, D, device=dev, dtype=torch.float16, generator=g),      # attn
        torch.randn(D, device=dev, dtype=torch.float32, generator=g),              # n2w
        torch.randn(D, device=dev, dtype=torch.float32, generator=g),              # n2b
        torch.randn(D, F, device=dev, dtype=torch.float16, generator=g) * 0.05,    # w1
        torch.randn(F, device=dev, dtype=torch.float16, generator=g),              # b1
        torch.randn(F, D, device=dev, dtype=torch.float16, generator=g) * 0.05,    # w2
        torch.randn(D, device=dev, dtype=torch.float16, generator=g),              # b2
    )


def _bench(fn, reps=3):
    return min(triton.testing.do_bench(fn, warmup=25, rep=50) for _ in range(reps))


def main():
    from bench.gpu_lock import gpu_lock

    props = torch.cuda.get_device_properties(0)
    SM = props.multi_processor_count
    D = F = 128

    with gpu_lock("g39 one-trip probe", timeout_s=7200):
        print(f"device: {props.name}  SMs={SM}  "
              f"smem/SM={props.shared_memory_per_multiprocessor}  "
              f"smem/block optin={props.shared_memory_per_block_optin}")
        print(f"fused-block smem at BM=32: {smem_bytes(D, F, 2, 32)} B  -> "
              f"{props.shared_memory_per_multiprocessor // smem_bytes(D, F, 2, 32)} "
              f"block(s)/SM\n")

        for tokens, cfgs in ((128, "2"), (2048, "4, 12"), (8192, "1, 9, 10 (declined)"),
                             (16384, "5 (declined)")):
            x, attn, n2w, n2b, w1, b1, w2, b2 = operands(tokens, D, F)
            a = (x, attn, n2w, n2b, w1, b1, w2, b2, n2w, n2b, 1e-5)
            bm = launch_tile(tokens, SM)
            ntiles = triton.cdiv(tokens, bm)
            flops = 2 * 2 * tokens * D * F

            print(f"\n{'='*88}\n{tokens} tokens  (configs {cfgs})   "
                  f"frontier: BM={bm} warps=8 grid={ntiles}\n{'='*88}")

            frontier = lambda: fused_ffn_normed(*a, block_m=bm, num_warps=8,
                                                store_next=True)
            y_ref, yn_ref = frontier()
            torch.cuda.synchronize()

            # ---- arm B: the geometry-MATCHED persistent form (same BM, same programs)
            matched = lambda: persistent_ffn(*a, block_m=bm, num_warps=8, grid=SM * 4,
                                             num_stages=1, store_next=True)[:2]
            y_m, yn_m, hm = persistent_ffn(*a, block_m=bm, num_warps=8, grid=SM * 4,
                                           num_stages=1, store_next=True)
            ok_m = torch.allclose(y_m, y_ref, atol=ATOL, rtol=RTOL)
            progs = min(SM * 4, ntiles)
            print(f"  persistent at matched BM={bm}: {progs} programs, "
                  f"{ntiles/progs:.2f} trips/program, matches={ok_m}")
            print(f"    n_regs={hm.n_regs} n_spills={hm.n_spills} smem={hm.metadata.shared}")

            # ---- ABBA-interleaved, cold round discarded, min of the rest -------------
            rounds, per = 6, {"frontier": [], "persistent": []}
            for r in range(rounds):
                order = (("frontier", frontier), ("persistent", matched))
                if r % 2:
                    order = tuple(reversed(order))
                for name, fn in order:
                    per[name].append(_bench(fn, reps=3))
            keep = {k: v[1:] for k, v in per.items()}      # discard the cold round
            tf, tp = min(keep["frontier"]), min(keep["persistent"])
            print(f"\n  ABBA, {rounds-1} kept rounds, min of min:")
            print(f"    frontier    {tf*1e3:8.3f} us   "
                  f"{100*flops/(tf*1e-3)/PEAK:5.1f}% of peak   "
                  f"rounds={[round(v*1e3,2) for v in keep['frontier']]}")
            print(f"    persistent  {tp*1e3:8.3f} us   "
                  f"{100*flops/(tp*1e-3)/PEAK:5.1f}% of peak   "
                  f"rounds={[round(v*1e3,2) for v in keep['persistent']]}")
            print(f"    ratio persistent/frontier = {tp/tf:.4f}x"
                  f"   <- F-03 claims 0.918-0.977x here")

            # ---- arm C: the SAME selection applied to the frontier itself ------------
            # F-03 timed the frontier once (min of 5) and the persistent form as the
            # minimum over 96 arms of min-of-3. Do the candidate-side protocol to the
            # frontier and see how much of the gap that alone explains.
            single = _bench(frontier, reps=5)
            draws = [_bench(frontier, reps=3) for _ in range(96)]
            best96 = min(draws)
            srt = sorted(draws)
            print(f"\n  SELECTION CONTROL -- frontier kernel, timed both ways:")
            print(f"    single min-of-5 draw (F-03's frontier protocol)      "
                  f"{single*1e3:8.3f} us")
            print(f"    min of 96 min-of-3 draws (F-03's candidate protocol) "
                  f"{best96*1e3:8.3f} us")
            print(f"    median of the 96 draws                               "
                  f"{srt[48]*1e3:8.3f} us")
            print(f"    apparent 'speedup' of the frontier over ITSELF       "
                  f"{best96/single:.4f}x"
                  f"   <- pure selection, no kernel change")


if __name__ == "__main__":
    main()
