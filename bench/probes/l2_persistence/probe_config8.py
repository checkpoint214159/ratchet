"""A-06 at config 8 -- the one announced row where the arithmetic does NOT close it.

Config 6 was settled by contrast (see finding 33): its whole four-layer fp16 weight set is
768 KiB, 1.6% of a 48 MiB L2, and cannot be evicted. Config 8 is different, and the number
is a coincidence worth stating exactly:

    d_model = ffn_dim = 1024, 4 layers
    per layer  (4 D^2 + 2 D F) * 2 B = 12.00 MiB
    all layers                        = 48.00 MiB   ==  this card's L2, to the byte

So the weights of layer 1 CAN be evicted by the weights of layers 2-4 plus the activation
stream, which is precisely the situation A-06 describes and config 6 does not exhibit.
This probe tests it directly rather than arguing about it.

WHAT IS TIMED
-------------
The four weight-bearing GEMMs of each layer, in order, over one contiguous 48 MiB arena
laid out exactly as A-06 specifies -- so a single access-policy window covers the whole
model's weights. Norms and attention are omitted: they carry no weights, and omitting them
REMOVES activation traffic, which biases the experiment TOWARD finding the weights
resident. The bias runs against the null, which is the direction an honest null needs.

    arm A   no persistence
    arm B   window over the whole 48 MiB arena, set-aside 33 MiB (this card's max),
            hitRatio = 33/48 per NVIDIA's guidance for a window larger than the reserve

Prediction before the run: A-06's ceiling here is the weight traffic actually RE-fetched.
The compulsory floor is 48 MiB per forward = 0.082 ms at 613.7 GB/s, 1.25% of config 8's
measured 6.549 ms. Persistence cannot touch that floor; it can only recover re-fetches, and
it can protect at most 33 of the 48 MiB.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))   # repo root

import torch
import torch.nn.functional as F

import l2_persist as L
from bench.matrix import BY_ID
from probe_weight_traffic import timeit

BW = 613.7e9


def main():
    cfg = BY_ID[8]
    m, d, f, layers = cfg.tokens, cfg.d_model, cfg.ffn_dim, cfg.layers
    dev = "cuda"

    # One contiguous fp16 arena holding every layer's weights, as A-06 specifies.
    per_layer = 4 * d * d + 2 * d * f
    arena = torch.empty(layers * per_layer, device=dev, dtype=torch.float16).normal_(0, 0.02)
    print(f"config 8: {m:,} tokens, d_model {d}, {layers} layers")
    print(f"weight arena {arena.numel() * 2 / 2**20:.2f} MiB   "
          f"L2 {torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB   "
          f"max persisting {L.max_persisting_l2_bytes() / 2**20:.0f} MiB")

    ws = []
    for i in range(layers):
        base = i * per_layer
        qkv = arena[base: base + 3 * d * d].view(3 * d, d)
        out = arena[base + 3 * d * d: base + 4 * d * d].view(d, d)
        w1 = arena[base + 4 * d * d: base + 4 * d * d + d * f].view(f, d)
        w2 = arena[base + 4 * d * d + d * f: base + per_layer].view(d, f)
        ws.append((qkv, out, w1, w2))

    x = torch.empty(m, d, device=dev, dtype=torch.float16).normal_(0, 1)

    def forward():
        h = x
        for qkv, out, w1, w2 in ws:
            t = F.linear(h, qkv)            # [m, 3d]
            h = F.linear(t[:, :d], out)     # [m, d]
            h = F.linear(h, w1)             # [m, f]
            h = F.linear(h, w2)             # [m, d]
        return h

    stream = torch.cuda.current_stream().cuda_stream
    nbytes = arena.numel() * 2
    cap = L.max_persisting_l2_bytes()

    results = {"no persistence": 1e9, "persisting window": 1e9}
    for _pass in range(3):
        results["no persistence"] = min(results["no persistence"],
                                        timeit(forward, reps=9, warmup=3))
        L.set_persisting_set_aside(min(nbytes, cap))
        L.set_window(stream, arena.data_ptr(), nbytes, min(1.0, cap / nbytes))
        try:
            results["persisting window"] = min(results["persisting window"],
                                               timeit(forward, reps=9, warmup=3))
        finally:
            L.clear_window(stream)
            L.set_persisting_set_aside(0)

    a, b = results["no persistence"], results["persisting window"]
    print(f"\n{'arm':<22} {'ms':>8}")
    print(f"{'no persistence':<22} {a:8.3f}")
    print(f"{'persisting window':<22} {b:8.3f}   {(a - b) / a * 100:+.2f}%")
    print(f"\ncompulsory weight floor: {nbytes / BW * 1e3:.3f} ms "
          f"({nbytes / BW * 1e3 / 6.549 * 100:.2f}% of config 8's measured 6.549 ms)")

    # Per-GEMM roofline, the same check finding 33 ran on config 6's QKV.
    print()
    for name, w, nout in (("qkv  [m,d]x[d,3d]", ws[0][0], 3 * d),
                          ("outp [m,d]x[d,d] ", ws[0][1], d),
                          ("ffn1 [m,d]x[d,f] ", ws[0][2], f)):
        t = timeit(lambda w=w: F.linear(x, w), reps=15, warmup=5)
        act = m * (d * 2 + nout * 2)
        wb = w.numel() * 2
        print(f"{name}: {t * 1e3:7.1f} us   activation floor {act / BW * 1e6:7.1f} us "
              f"({t / (act / BW * 1e3) * 100:6.1f}% of floor)   weights {wb / 2**20:.2f} MiB "
              f"= {wb / (act + wb) * 100:.1f}% of compulsory traffic")


if __name__ == "__main__":
    main()
