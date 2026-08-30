"""A-06 premise probe: how much of config 6's HBM traffic is WEIGHTS?

THE QUESTION, AND WHY THE OBVIOUS TOOL IS UNAVAILABLE
-----------------------------------------------------
A-06 proposes pinning the fp16 weight arena in L2 with a persisting-access window. The
whole proposal is worth zero unless the weights are currently MISSING L2. The direct
measurement is `ncu --metrics dram__bytes_read.sum`, which on this box returns
ERR_NVGPUCTRPERM: WSL2 denies GPU performance counters and the fix (a modprobe option on
the host driver) is not reachable from inside the guest.

So the premise is measured differently: by a CONTRAST that isolates weight DRAM traffic
as the only difference between two otherwise-identical kernels.

THE CONTRAST
------------
`_ffn_block_off` is `bench/kernels/ffn_fused._ffn_block` plus one extra argument: a
per-program element offset into the weight arena, loaded from an int32 array. Identical
instruction stream, identical arithmetic, identical activation traffic. The only thing
that changes between arms is HOW MANY DISTINCT weight copies the grid touches:

    NCOPY = 1     all 20000 programs read the same 64 KiB pair  (what the real kernel does)
    NCOPY = 16    a 1 MiB arena -- still far inside a 48 MiB L2
    NCOPY = 512   a 32 MiB arena, reuse distance ~74 MB > L2   -- weight reads must miss

If the real kernel (NCOPY=1) is already getting its weights from L2, then going to
NCOPY=512 must ADD the full worst-case weight traffic, and the measured delta will match
the analytic prediction. If instead NCOPY=1 were already missing, the delta would be
small -- there would be nothing left to lose.

    weight bytes per program : 2 * D * F * 2 = 64 KiB at D = F = 128
    programs                 : ceil(M / BM) = 1_280_000 / 64 = 20000
    worst-case weight traffic: 20000 * 64 KiB = 1.31 GB = 2.14 ms at 613.7 GB/s

L38 COMPLIANCE -- THE PROBE MUST BE ABLE TO FIRE
------------------------------------------------
Two positive controls, because a null from an instrument that cannot detect anything is
not evidence:

  1. NCOPY=512 is the control for the TRAFFIC measurement. If it does not slow down, the
     contrast cannot see weight traffic and no arm's result means anything.
  2. NCOPY=512 + a persisting window over the whole 32 MiB arena (inside this card's
     33 MiB max set-aside) is the control for the PERSISTENCE API. If persistence cannot
     speed THAT up -- a working set deliberately built to thrash -- then a null on
     NCOPY=1 says nothing about the API either.

Predictions, written before the run (L37 / the forward-prediction discipline of finding 25):

    hot   (NCOPY=1)    ~ activation floor; weight traffic ~ 0
    warm  (NCOPY=16)   ~ hot; 1 MiB is still nothing against 48 MiB
    cold  (NCOPY=512)  ~ hot + 2.1 ms
    hot  + persist     == hot   (nothing to gain)
    cold + persist     <  cold  (this is what the feature is FOR)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))   # repo root

import torch
import triton
import triton.language as tl

import l2_persist as L
from bench.matrix import BY_ID


@triton.jit
def _ffn_block_off(
    XN, RES, W1, B1, W2, B2, Y, WOFF,
    M,
    D: tl.constexpr, F: tl.constexpr, BM: tl.constexpr,
):
    """bench/kernels/ffn_fused._ffn_block, plus a per-program weight-arena offset.

    The offset is a 4-byte load per program (80 KB over the whole grid, 0.005% of the
    activation traffic) and is present in EVERY arm, so it cancels in the contrast.
    """
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    rf = tl.arange(0, F)
    keep = rm < M

    off = tl.load(WOFF + pid)

    xn = tl.load(XN + rm[:, None] * D + rd[None, :], mask=keep[:, None], other=0.0)
    w1 = tl.load(W1 + off + rd[:, None] * F + rf[None, :])
    w2 = tl.load(W2 + off + rf[:, None] * D + rd[None, :])
    b1 = tl.load(B1 + rf)
    b2 = tl.load(B2 + rd)

    h = tl.dot(xn, w1, out_dtype=tl.float32) + b1[None, :].to(tl.float32)
    h = h * 0.5 * (1.0 + tl.erf(h * 0.70710678118654752440))
    y = tl.dot(h.to(w2.dtype), w2, out_dtype=tl.float32) + b2[None, :].to(tl.float32)

    res = tl.load(RES + rm[:, None] * D + rd[None, :], mask=keep[:, None], other=0.0)
    tl.store(Y + rm[:, None] * D + rd[None, :], res + y, mask=keep[:, None])


BM = 64
WARPS = 4
BW = 613.7e9          # measured device bandwidth, ledger/device.json


def timeit(fn, reps=15, warmup=5):
    """Minimum of N, per CLAUDE.md: clocks are not lockable under WSL2."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a = torch.cuda.Event(True)
        b = torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return min(ts)


def main():
    cfg = BY_ID[6]
    m = cfg.tokens
    d, f = cfg.d_model, cfg.ffn_dim
    grid = triton.cdiv(m, BM)
    pair_elems = d * f + f * d          # W1 and W2 laid out contiguously in one arena
    pair_bytes = pair_elems * 2

    print(f"config 6: M={m:,} D={d} F={f}  BM={BM} -> {grid:,} programs")
    print(f"weight pair: {pair_bytes/1024:.0f} KiB   "
          f"worst-case weight traffic {grid*pair_bytes/1e9:.3f} GB "
          f"= {grid*pair_bytes/BW*1e3:.3f} ms at {BW/1e9:.1f} GB/s")
    act_bytes = m * (d * 2 + d * 4 + d * 4)
    print(f"activation traffic {act_bytes/1e9:.3f} GB = {act_bytes/BW*1e3:.3f} ms")
    print(f"L2 {torch.cuda.get_device_properties(0).L2_cache_size/2**20:.0f} MiB   "
          f"maxPersistingL2 {L.max_persisting_l2_bytes()/2**20:.0f} MiB")
    print()

    dev = "cuda"
    xn = torch.randn(m, d, device=dev, dtype=torch.float16)
    res = torch.randn(m, d, device=dev, dtype=torch.float32)
    y = torch.empty(m, d, device=dev, dtype=torch.float32)
    b1 = torch.randn(f, device=dev, dtype=torch.float16)
    b2 = torch.randn(d, device=dev, dtype=torch.float16)

    NCOPIES = [1, 16, 512]
    arenas, w1s, w2s, woffs = {}, {}, {}, {}
    for n in NCOPIES:
        arena = torch.randn(n * pair_elems, device=dev, dtype=torch.float16) * 0.05
        arenas[n] = arena
        w1s[n] = arena[: d * f].view(d, f)          # copy 0; offsets index the rest
        w2s[n] = arena[d * f: 2 * d * f].view(f, d)
        pid = torch.arange(grid, device=dev, dtype=torch.int32)
        woffs[n] = ((pid % n) * pair_elems).to(torch.int32)
        print(f"NCOPY={n:<4} arena {arena.numel()*2/2**20:8.2f} MiB  "
              f"reuse distance ~{n*(pair_bytes + BM*d*10)/1e6:7.1f} MB")
    print()

    # --- the activation-only floor: exactly the FFN kernel's activation traffic --------
    # fp32 read (res) + fp16 read (xn) + fp32 write (y) = 1280 B/token, the same three
    # streams the FFN megakernel moves. One kernel, no temporary: torch type-promotes.
    def act_only():
        torch.add(res, xn, out=y)

    # --- the arms ---------------------------------------------------------------------
    def run(n):
        _ffn_block_off[(grid,)](xn, res, w1s[n], b1, w2s[n], b2, y, woffs[n], m,
                                D=d, F=f, BM=BM, num_warps=WARPS)

    stream = torch.cuda.current_stream().cuda_stream

    def with_persist(n, fn):
        """Install the window over arena n, run, take it down."""
        arena = arenas[n]
        nbytes = arena.numel() * 2
        cap = L.max_persisting_l2_bytes()
        L.set_persisting_set_aside(min(nbytes, cap))
        # NVIDIA's guidance: hitRatio = set_aside / window when the window is larger.
        ratio = 1.0 if nbytes <= cap else cap / nbytes
        L.set_window(stream, arena.data_ptr(), min(nbytes, L.max_window_bytes()), ratio)
        try:
            return fn()
        finally:
            L.clear_window(stream)
            L.set_persisting_set_aside(0)

    results = {}
    # Interleave: three passes round-robin over every arm, keep the min, so a clock
    # excursion cannot land on one arm only.
    for _pass in range(3):
        for n in NCOPIES:
            t = timeit(lambda n=n: run(n), reps=7, warmup=3)
            key = f"ffn NCOPY={n}"
            results[key] = min(results.get(key, 1e9), t)
        for n in NCOPIES:
            t = with_persist(n, lambda n=n: timeit(lambda: run(n), reps=7, warmup=3))
            key = f"ffn NCOPY={n} +persist"
            results[key] = min(results.get(key, 1e9), t)
        t = timeit(act_only, reps=7, warmup=3)
        results["activation-floor"] = min(results.get("activation-floor", 1e9), t)

    print(f"{'arm':<28} {'ms':>9} {'eff GB/s (act only)':>21}")
    for k in ["activation-floor"] + [f"ffn NCOPY={n}" for n in NCOPIES] + \
             [f"ffn NCOPY={n} +persist" for n in NCOPIES]:
        t = results[k]
        print(f"{k:<28} {t:9.3f} {act_bytes/(t*1e-3)/1e9:21.1f}")
    print()

    hot = results["ffn NCOPY=1"]
    cold = results["ffn NCOPY=512"]
    predicted = grid * pair_bytes / BW * 1e3
    print(f"cold - hot            = {cold - hot:7.3f} ms")
    print(f"predicted full weight-miss cost = {predicted:7.3f} ms")
    print(f"fraction of the worst case that NCOPY=1 is ALREADY saving: "
          f"{(cold - hot)/predicted*100:.1f}%")
    print()
    for n in NCOPIES:
        a, b = results[f"ffn NCOPY={n}"], results[f"ffn NCOPY={n} +persist"]
        print(f"persist effect NCOPY={n:<4}: {a:7.3f} -> {b:7.3f} ms  "
              f"({(a-b)/a*100:+.2f}%)")

    # --- the cuBLAS side: QKV projection, the other big weight consumer ---------------
    print()
    qkv_w = torch.randn(3 * d, d, device=dev, dtype=torch.float16)
    qkv_b = torch.randn(3 * d, device=dev, dtype=torch.float16)
    tq = timeit(lambda: torch.nn.functional.linear(xn, qkv_w, qkv_b), reps=11, warmup=5)
    qkv_act = m * (d * 2 + 3 * d * 2)
    print(f"F.linear QKV [{m},{d}]x[{d},{3*d}] : {tq:.3f} ms  "
          f"activation floor {qkv_act/BW*1e3:.3f} ms "
          f"({qkv_act/(tq*1e-3)/1e9:.1f} GB/s counting activations only)")
    print(f"  worst-case cuBLAS weight re-read (128-row tiles, 96 KiB each): "
          f"{(m//128)*3*d*d*2/1e9:.3f} GB = {(m//128)*3*d*d*2/BW*1e3:.3f} ms")


if __name__ == "__main__":
    main()
