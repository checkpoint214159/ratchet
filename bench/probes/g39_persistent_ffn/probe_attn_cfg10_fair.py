"""Config 10, measured symmetrically: the looped kernel against a PROPERLY TUNED incumbent.

`probe_looped_attn.py` swept the looped form over 180 arms (BM x BN x warps x stages) and
the incumbent `attn_single_tile` over four (BM only, at 4 warps), then reported the ratio.
That is exactly the best-of-N-against-best-of-1 handicap finding 47 measured at 4.5% -- and
finding 31 says config 10's best single tile is 32 rows at EIGHT warps, which that sweep
never tried. Any config-10 claim from it is unearned.

This probe fixes it. Both forms are swept over their full legal grid, the winners are then
run head to head ABBA-interleaved with the cold round discarded, and the incumbent's own
`autotune_tile` -- the routine the real candidate calls at prime time -- is timed as a
third arm so the comparison is against what the model ACTUALLY runs, not against the best
arm a sweep can find for it.

Outputs are checked against `sdpa + repack` at the locked tolerance before anything is
timed. n_regs / n_spills / smem are reported for both winners.

INDICATIVE ONLY [L41].
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import triton

from bench.kernels.attn_single_tile import autotune_tile, single_tile_attention
from bench.probes.g39_persistent_ffn.probe_looped_attn import (_bench, looped_attention,
                                                               sdpa_repack)

ATOL, RTOL = 0.002, 0.02

# config 10: B=64, heads=2, d_model=128 -> head_dim 64, seq 128.  Also config 9 (H=1,
# hd=128) as the shape F-02's kill bar is written against.
SHAPES = (("cfg 10", 64, 2, 64, 128), ("cfg 9", 64, 1, 128, 128))


def main():
    from bench.gpu_lock import gpu_lock

    props = torch.cuda.get_device_properties(0)
    dev = torch.device("cuda")

    with gpu_lock("g39 fair config-10 comparison", timeout_s=7200):
        for label, B, H, hd, S in SHAPES:
            dm = H * hd
            g = torch.Generator(device="cuda").manual_seed(11)
            qkv = torch.randn(B, S, 3 * dm, device=dev, dtype=torch.float16,
                              generator=g) * 0.3
            scale = hd ** -0.5
            ref = sdpa_repack(qkv, H, hd)
            torch.cuda.synchronize()

            print(f"\n{'='*90}\n{label}  B={B} H={H} hd={hd} S={S}\n{'='*90}")

            # ---- incumbent, swept over its FULL grid (BM x warps x stages) ------------
            single = []
            for bm in (16, 32, 64, 128):
                for w in (2, 4, 8):
                    for st in (1, 2, 3):
                        try:
                            o = single_tile_attention(qkv, H, hd, scale, bm, w, st)
                            torch.cuda.synchronize()
                        except Exception:
                            continue
                        if not torch.allclose(o.float(), ref.float(), atol=ATOL,
                                              rtol=RTOL):
                            continue
                        t = _bench(lambda bm=bm, w=w, st=st: single_tile_attention(
                            qkv, H, hd, scale, bm, w, st), reps=3)
                        single.append((t, bm, w, st))
            single.sort()

            # ---- looped, swept over the same shape of grid ---------------------------
            loop = []
            for bm in (16, 32, 64, 128):
                for bn in (16, 32, 64, 128):
                    for w in (2, 4, 8):
                        for st in (1, 2, 3, 4):
                            try:
                                o, h = looped_attention(qkv, H, hd, scale, bm, bn, w, st)
                                torch.cuda.synchronize()
                            except Exception:
                                continue
                            if not torch.allclose(o.float(), ref.float(), atol=ATOL,
                                                  rtol=RTOL):
                                continue
                            t = _bench(lambda bm=bm, bn=bn, w=w, st=st: looped_attention(
                                qkv, H, hd, scale, bm, bn, w, st)[0], reps=3)
                            loop.append((t, bm, bn, w, st, h.n_regs, h.n_spills,
                                         h.metadata.shared))
            loop.sort()

            print(f"  single_tile: {len(single)} legal arms swept"
                  + (f", best {single[0][0]*1e3:.3f} us at BM={single[0][1]} "
                     f"warps={single[0][2]} stages={single[0][3]}" if single
                     else " -- DECLINES this shape"))
            if not loop:
                print("  looped: no arm compiled and matched")
                continue
            lt, lbm, lbn, lw, lst, lr, lsp, lsm = loop[0]
            print(f"  looped     : {len(loop)} legal arms swept, best {lt*1e3:.3f} us at "
                  f"BM={lbm} BN={lbn} warps={lw} stages={lst}")
            print(f"               n_regs={lr} n_spills={lsp} smem={lsm}  "
                  f"grid={B*H*triton.cdiv(S,lbm)} CTAs on {props.multi_processor_count} SMs")

            # ---- what the MODEL actually runs, via the incumbent's own tuner ----------
            tuned = None
            try:
                tuned, why = autotune_tile(S, hd, H, B, dev)
                print(f"  autotune_tile (what the candidate calls at prime time) -> "
                      f"{tuned}  ({why})")
            except Exception as e:
                print(f"  autotune_tile unavailable: {e}")

            # ---- head to head, ABBA, cold round discarded ----------------------------
            arms = {"looped": lambda: looped_attention(qkv, H, hd, scale, lbm, lbn,
                                                       lw, lst)[0],
                    "sdpa+repack": lambda: sdpa_repack(qkv, H, hd)}
            if single:
                _, sbm, sw, sst = single[0]
                arms["single_tile"] = lambda: single_tile_attention(qkv, H, hd, scale,
                                                                    sbm, sw, sst)
            if tuned:
                tbm, tw, tst = tuned
                arms["single_tile(tuned)"] = lambda: single_tile_attention(
                    qkv, H, hd, scale, tbm, tw, tst)

            names = list(arms)
            per = {n: [] for n in names}
            for r in range(6):
                order = names if r % 2 == 0 else list(reversed(names))
                for n in order:
                    per[n].append(_bench(arms[n], reps=3))
            kept = {n: min(v[1:]) for n, v in per.items()}     # discard cold round

            print(f"\n  ABBA head-to-head, 5 kept rounds, min of min:")
            base = kept.get("single_tile(tuned)") or kept.get("single_tile") \
                or kept["sdpa+repack"]
            for n in names:
                print(f"    {n:<22} {kept[n]*1e3:8.3f} us   "
                      f"{base/kept[n]:6.3f}x vs the incumbent")


if __name__ == "__main__":
    main()
