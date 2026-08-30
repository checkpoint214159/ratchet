"""A-06, the other half: what does the SET-ASIDE cost when there is nothing to gain?

`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, N)` carves N bytes out of the 48 MiB
L2 and reserves them for persisting accesses. This card allows up to 33 MiB -- 69% of the
whole cache. Config 6 streams 1.6 GB per FFN kernel through that cache, so if the reserve
is not fully used by the pinned window, the streaming path loses capacity for nothing.

Arms, all on the real NCOPY=1 access pattern (one 64 KiB weight pair, what the frontier
kernel actually does):

    set-aside 0        no persistence at all              (the frontier)
    set-aside 1 MiB    window over the 64 KiB weight pair (a sized set-aside)
    set-aside 33 MiB   window over the 64 KiB weight pair (the maximum reserve)

Prediction before the run: all three within the noise floor. Config 6's activation stream
has ZERO reuse -- every byte is read once -- so shrinking its share of L2 costs nothing
either, and pinning 64 KiB that is already resident gains nothing. The interesting
outcome would be a LOSS on the 33 MiB arm, which would make A-06 not merely inert but a
hazard.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))   # repo root

import torch
import triton

import l2_persist as L
from bench.matrix import BY_ID
from probe_weight_traffic import _ffn_block_off, BM, WARPS, timeit


def main():
    cfg = BY_ID[6]
    m, d, f = cfg.tokens, cfg.d_model, cfg.ffn_dim
    grid = triton.cdiv(m, BM)
    dev = "cuda"

    xn = torch.randn(m, d, device=dev, dtype=torch.float16)
    res = torch.randn(m, d, device=dev, dtype=torch.float32)
    y = torch.empty(m, d, device=dev, dtype=torch.float32)
    # One contiguous weight arena, exactly as A-06 specifies, so the window covers both.
    arena = torch.randn(2 * d * f, device=dev, dtype=torch.float16) * 0.05
    w1 = arena[: d * f].view(d, f)
    w2 = arena[d * f:].view(f, d)
    b1 = torch.randn(f, device=dev, dtype=torch.float16)
    b2 = torch.randn(d, device=dev, dtype=torch.float16)
    woff = torch.zeros(grid, device=dev, dtype=torch.int32)

    def run():
        _ffn_block_off[(grid,)](xn, res, w1, b1, w2, b2, y, woff, m,
                                D=d, F=f, BM=BM, num_warps=WARPS)

    stream = torch.cuda.current_stream().cuda_stream
    arms = {"set-aside 0 (frontier)": 0,
            "set-aside 1 MiB": 1 << 20,
            "set-aside 33 MiB (max)": L.max_persisting_l2_bytes()}

    results = {k: 1e9 for k in arms}
    for _pass in range(3):
        for name, aside in arms.items():
            if aside:
                got = L.set_persisting_set_aside(aside)
                L.set_window(stream, arena.data_ptr(), arena.numel() * 2, 1.0)
            try:
                t = timeit(run, reps=7, warmup=3)
            finally:
                if aside:
                    L.clear_window(stream)
                    L.set_persisting_set_aside(0)
            results[name] = min(results[name], t)

    base = results["set-aside 0 (frontier)"]
    print(f"{'arm':<26} {'ms':>8} {'vs frontier':>12}")
    for k in arms:
        print(f"{k:<26} {results[k]:8.3f} {(base - results[k]) / base * 100:+11.2f}%")


if __name__ == "__main__":
    main()
