"""Print the FULL grid of every sweep, one row per tile, so a decision can be audited.

[L62]'s cheap check, which nobody had run before generation 42: print the sweep and look
at how many distinct values it contains. Here it is asked of REPLICATED sweeps -- do the
two passes over the same grid, in the same process, back to back, produce the same
ordering?

`--after` builds and primes another candidate first, which is what `bench/abba.py` does
and what the fresh-process stability probe found to be the whole difference: a tuner
asked in a one-arm process behaves differently from the same tuner asked second.

    python3 bench/probes/g43_stable_tiles/sweep_grids.py --id 2 --sweeps 3 \
        --after v42_hot_tuned_tile
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def _reference(tag: str):
    spec = importlib.util.spec_from_file_location(
        tag, REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    import torch
    from bench.candidates import REGISTRY
    from bench.kernels import attn_single_tile as ast
    from bench.matrix import BY_ID

    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--after", default=None,
                    help="build and prime this candidate before sweeping")
    a = ap.parse_args()

    c = BY_ID[a.id]
    dev = torch.device("cuda")

    if a.after:
        ref = _reference("ref_grids")
        tcfg = ref.TransformerConfig(
            batch_size=c.batch_size, seq_len=c.seq_len, d_model=c.d_model,
            num_heads=c.heads, ffn_dim=c.ffn_dim, num_layers=c.layers, causal=c.causal)
        tcfg.validate()
        torch.manual_seed(1234)
        base = ref.BaselineTransformer(tcfg)
        mdl = REGISTRY[a.after].build(ref.BaselineTransformer)(tcfg)
        ref.copy_model_weights(base, mdl)
        mdl = mdl.to(device=dev, dtype=torch.float32).eval()
        del base
        x, m = ref.generate_random_case(tcfg, dev, torch.float32, seed=1234,
                                        padding_ratio=0.0, input_scale=1.0)
        with torch.inference_mode():
            mdl(x, m)
        print(f"# primed {a.after}: {getattr(mdl, 'attn_tile', None)}")

    props = torch.cuda.get_device_properties(dev)
    s, hd, h, b = c.seq_len, c.head_dim, c.heads, c.batch_size
    tiles = ast.viable_tiles(s, hd, props.regs_per_multiprocessor,
                             props.max_threads_per_multi_processor, props.warp_size)
    derived = ast.choose_tile(s, hd, props.regs_per_multiprocessor,
                              props.max_threads_per_multi_processor, props.warp_size)
    pb = max(1, min(b, 4 * props.multi_processor_count // max(1, h)))
    qkv = torch.randn(pb, s, 3 * h * hd, device=dev, dtype=torch.float16)
    scale = hd ** -0.5

    print(f"# cfg {a.id}: S={s} hd={hd} H={h} B={b} probe_batch={pb} derived={derived}")
    # INSIDE `inference_mode`, because that is where `_decide_attn` runs -- `abba.py`
    # primes every arm inside one and so does the graded harness. Measured (see
    # `timer_fallback.py`): OUTSIDE it, once any model has been run under inference mode,
    # `do_bench_cudagraph` raises `Inplace update to inference tensor outside
    # InferenceMode` and `hot_time`'s bare `except` silently returns `do_bench`'s number
    # instead. A grid dumped outside inference mode is therefore a grid of the FLUSHED
    # timer wearing the hot timer's name, every value an exact multiple of 1.024 us --
    # which is exactly what the first draft of this probe printed and nearly concluded
    # from. The regime of the probe is the regime of the answer, again.
    grids = []
    with torch.inference_mode():
        for _ in range(a.sweeps):
            row = {}
            for bm, w, st in tiles:
                fn = (lambda bm=bm, w=w, st=st:
                      ast.single_tile_attention(qkv, h, hd, scale, bm, w, st))
                fn()
                torch.cuda.synchronize()
                row[(bm, w, st)] = ast.hot_time(fn, 2) * 1e3
            grids.append(row)

    hdr = "  ".join(f"sweep{i+1:>2}" for i in range(a.sweeps))
    print(f"{'tile':<14} {hdr}     winner-of-each")
    for t in tiles:
        vals = "  ".join(f"{g[t]:7.3f}" for g in grids)
        mark = "  <- derived" if t == derived else ""
        print(f"{str(t):<14} {vals}{mark}")
    wins = [min(g, key=g.get) for g in grids]
    print(f"\nwinners : {wins}")
    for i, g in enumerate(grids):
        best = min(g, key=g.get)
        print(f"sweep {i+1}: best {best} at {g[best]:.3f} us, derived "
              f"{derived} at {g[derived]:.3f} us -> {g[derived]/g[best]:.3f}x, "
              f"{len(set(round(v, 3) for v in g.values()))} distinct of {len(g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
