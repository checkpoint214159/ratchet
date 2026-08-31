"""Do v16's FFN megakernel constants survive a timer that can resolve them?

THE QUESTION, AND WHY IT IS OPEN
---------------------------------
`bench/kernels/ffn_fused.py` has no autotuner. `_ffn_block`'s tile is `BLOCK_M = 64,
NUM_WARPS = 8`, two class attributes set on `v16_ffn_megakernel` and justified in the
source as *"measured best at every shape that fits"* -- measured at generation 16, with
`do_bench`, which finding 53 then showed is blind at these sizes. Every tile decision in
the attention package has since been re-derived; this one never was.

WHERE IT ACTUALLY FIRES, TRACED RATHER THAN ASSUMED
-----------------------------------------------------
Two different routines call into this file and only one of them is unswept:

  * `_ffn_block` via `fused_ffn(..., self.BLOCK_M, self.NUM_WARPS)` -- reached on the
    v23 plain branch when `fused_ffn_used and _nomask`, i.e. where `amortizes` is true.
    At d_model = ffn_dim = 128 that needs >= 25600 tokens, so on the announced matrix it
    is configs 6 and 13; at d_model = 32 the crossover falls to 6400 tokens, which adds
    config 7. **These three are the constants under test.**

  * `_ffn_block_normed` via `fused_ffn_normed(..., self.launch_bm, self.launch_warps)` --
    reached on v34's launch-bound branch, where `one_wave` holds and `amortizes` does
    not: configs 2, 3, 4 and 12. Its `block_m` is DERIVED per shape by
    `ffn_fused.launch_tile` from the measured SM count, and its warp count is swept at
    prime time by `v34._pick_warps`. Not a constant, and not this probe's subject --
    though `_pick_warps` does rank with the flushed `do_bench`, which is finding 53's
    defect one file over, and is reported separately.

WHAT IS SWEPT
-------------
Every `(block_m, num_warps)` that `ffn_fused.fits` accepts on the measured device, at the
REAL shapes, under BOTH timers, with correctness checked against the same arithmetic the
model's un-fused path performs, at the locked tolerance, before any arm is timed. Two
independent passes, because a single ranking of a grid is not a ranking (L56, and finding
53's whole subject).

THE REGIME, STATED RATHER THAN ASSUMED
----------------------------------------
The call site is one node of a replayed CUDA graph with the model resident. This probe
allocates the same tensors at the same shapes and times the same kernel inside
`torch.inference_mode()` -- which matters, because outside it `do_bench_cudagraph` raises
after any model has run and `hot_time` silently degrades to `do_bench`.

Where this probe's regime IS the call site's: configs 6 and 13 stream 655 MB and 34 MB of
activations past a 48 MB L2, so the kernel is HBM-bound and L2-cold either way and there
is nothing for residency to change. Where it is WEAKER: config 7 is 8192 tokens at
d_model 32, small enough that co-residency and launch state matter, exactly as they do
for attention. So a config-7 result here proposes and does not conclude; configs 6 and 13
are the ones this instrument can answer.

    python3 bench/probes/g43_stable_tiles/ffn_tile_sweep.py --ids 6 13 7 --passes 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

RTOL, ATOL = 0.02, 0.002          # the locked tolerance. Never widened.
QUANTUM_US = 1.024

# Every warp count Triton accepts up to the point where a 128-row tile has more warps
# than rows to give them. Nothing is fitted: arms that do not compile or do not fit are
# dropped by the sweep itself.
WARP_CANDIDATES = (1, 2, 4, 8, 16)
BLOCK_CANDIDATES = (16, 32, 64, 128)


def main() -> int:
    import torch
    import torch.nn.functional as F
    from bench.kernels import ffn_fused as ff
    from bench.kernels.attn_single_tile import flushed_time, hot_time
    from bench.matrix import BY_ID

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    smem_optin = props.shared_memory_per_block_optin

    for cid in a.ids:
        c = BY_ID[cid]
        d, f, tokens = c.d_model, c.ffn_dim, c.tokens
        fires = ff.fits(d, f, 2, 64, smem_optin) and ff.amortizes(tokens, d, f, 2)
        print(f"\n{'='*78}\ncfg {cid}: tokens={tokens} d_model={d} ffn_dim={f}   "
              f"`fused_ffn` fires here: {fires}")
        if not fires:
            print("  (amortizes/fits declines -- `_ffn_block` is not the kernel that "
                  "runs on this row; skipping)")
            continue

        torch.manual_seed(1234)
        xn = torch.randn(tokens, d, device=dev, dtype=torch.float16)
        res = torch.randn(tokens, d, device=dev, dtype=torch.float32)
        w1 = torch.randn(d, f, device=dev, dtype=torch.float16) * (d ** -0.5)
        w2 = torch.randn(f, d, device=dev, dtype=torch.float16) * (f ** -0.5)
        b1 = torch.randn(f, device=dev, dtype=torch.float16)
        b2 = torch.randn(d, device=dev, dtype=torch.float16)

        # WHAT THE KERNEL MUST REPRODUCE: the un-fused path's own arithmetic, in fp32,
        # which is what the model computes where the megakernel declines.
        with torch.inference_mode():
            h = F.linear(xn.float(), w1.float().t(), b1.float())
            ref = res + F.linear(F.gelu(h, approximate="none"),
                                 w2.float().t(), b2.float())

        arms = [(bm, w) for bm in BLOCK_CANDIDATES for w in WARP_CANDIDATES
                if ff.fits(d, f, 2, bm, smem_optin)]
        rows = {}
        with torch.inference_mode():
            for bm, w in arms:
                try:
                    out = ff.fused_ffn(xn, res, w1, b1, w2, b2, bm, w)
                    torch.cuda.synchronize()
                except Exception as exc:
                    rows[(bm, w)] = ("did not launch", str(exc)[:60])
                    continue
                if not torch.allclose(out.float(), ref, atol=ATOL, rtol=RTOL):
                    rows[(bm, w)] = ("FAILED TOLERANCE",
                                     f"max_abs {(out.float()-ref).abs().max():.3e}")
                    continue
                fn = (lambda bm=bm, w=w:
                      ff.fused_ffn(xn, res, w1, b1, w2, b2, bm, w))
                flu = [flushed_time(fn, 2) * 1e3 for _ in range(a.passes)]
                hot = [hot_time(fn, 2) * 1e3 for _ in range(a.passes)]
                rows[(bm, w)] = (flu, hot)
        del xn, res, w1, w2, b1, b2, ref
        torch.cuda.empty_cache()

        shipped = (64, 8)
        print(f"{'tile (bm,warps)':<18} {'flushed us':>26}  {'hot us':>26}")
        ok = {}
        for arm in arms:
            v = rows[arm]
            mark = "   <- SHIPPED (v16)" if arm == shipped else ""
            if isinstance(v[0], str):
                print(f"{str(arm):<18} {v[0]:>26}  {v[1]:>26}{mark}")
                continue
            flu, hot = v
            ok[arm] = (min(flu), min(hot))
            print(f"{str(arm):<18} "
                  f"{'  '.join(f'{x:10.3f}' for x in flu):>26}  "
                  f"{'  '.join(f'{x:10.3f}' for x in hot):>26}{mark}")
        if not ok:
            print("  no arm both matched and timed")
            continue

        def _q(us):
            return abs(us / QUANTUM_US - round(us / QUANTUM_US)) < 1e-6

        for label, idx in (("flushed", 0), ("hot", 1)):
            vals = {k: v[idx] for k, v in ok.items()}
            best = min(vals, key=vals.get)
            base = vals.get(shipped)
            nq = sum(1 for v in vals.values() if _q(v))
            print(f"  {label:<8} best {best} at {vals[best]:.3f} us; "
                  f"v16's {shipped} at {base:.3f} us -> "
                  f"{base/vals[best]:.3f}x;  "
                  f"{len(set(round(v,3) for v in vals.values()))} distinct of "
                  f"{len(vals)}, {nq} on the event quantum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
