"""Does `hot_time` actually use the hot timer once a model is resident and captured?

`attn_single_tile.hot_time` wraps `do_bench_cudagraph` in a bare `except Exception` and
falls back to `do_bench` -- and its own docstring says so: *"A device or context that
refuses capture falls back to `do_bench` and the caller is none the wiser."* That was
written as a robustness note. This asks whether it is a robustness note or a silent
regression to the exact instrument generation 42 was built to remove.

The tell is arithmetic and needs no instrumentation to read: the CUDA event quantum on
this card is 1.024 us, so a reading produced by `do_bench` is an exact integer multiple
of it and a reading produced by `do_bench_cudagraph` is not.

    python3 bench/probes/g43_stable_tiles/timer_fallback.py --id 2 --after v42_hot_tuned_tile
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

QUANTUM_US = 1.024


def _reference(tag: str):
    spec = importlib.util.spec_from_file_location(
        tag, REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def _quantized(us: float) -> str:
    if us != us:
        return "RAISED"
    q = us / QUANTUM_US
    return "QUANTIZED" if abs(q - round(q)) < 1e-6 else "sub-quantum"


def main() -> int:
    import torch
    import triton.testing as tt
    from bench.candidates import REGISTRY
    from bench.kernels import attn_single_tile as ast
    from bench.matrix import BY_ID

    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--after", default=None)
    # ONE CONTEXT PER PROCESS, and this is not fastidiousness. A `do_bench_cudagraph`
    # that raises mid-capture leaves the CUDA RNG offset in a state where the NEXT
    # `torch.randn` dies with "Offset increment outside graph capture encountered
    # unexpectedly" -- so a probe that tries both contexts in one process measures the
    # first one and then crashes, which is how this was nearly missed.
    ap.add_argument("--ctx", choices=("grad", "inference"), default="inference")
    a = ap.parse_args()

    c = BY_ID[a.id]
    dev = torch.device("cuda")

    def report(stage: str, inference: bool):
        # THE CALL SITE RUNS INSIDE `torch.inference_mode()`. `bench/abba.py` primes
        # every arm inside one, and so does the graded harness; `_decide_attn` therefore
        # runs there too. Whether that matters is the whole question, so it is a
        # parameter rather than an assumption.
        import contextlib
        ctx = torch.inference_mode() if inference else contextlib.nullcontext()
        with ctx:
            _report(stage + (" [inference_mode]" if inference else " [no grad ctx]"))

    def _report(stage: str):
        props = torch.cuda.get_device_properties(dev)
        s, hd, h, b = c.seq_len, c.head_dim, c.heads, c.batch_size
        pb = max(1, min(b, 4 * props.multi_processor_count // max(1, h)))
        qkv = torch.randn(pb, s, 3 * h * hd, device=dev, dtype=torch.float16)
        scale = hd ** -0.5
        tile = ast.choose_tile(s, hd, props.regs_per_multiprocessor,
                               props.max_threads_per_multi_processor, props.warp_size)
        fn = lambda: ast.single_tile_attention(qkv, h, hd, scale, *tile)   # noqa: E731
        fn()
        torch.cuda.synchronize()

        try:
            graph_us = min(tt.do_bench_cudagraph(fn, rep=25, return_mode="min")
                           for _ in range(2)) * 1e3
            graph_err = None
        except Exception as exc:
            graph_us, graph_err = float("nan"), f"{type(exc).__name__}: {exc}"
        flush_us = min(tt.do_bench(fn, warmup=10, rep=25, return_mode="min")
                       for _ in range(2)) * 1e3
        hot_us = ast.hot_time(fn, 2) * 1e3

        print(f"\n--- {stage} --- tile {tile}")
        print(f"  do_bench_cudagraph : {graph_us:9.3f} us  {_quantized(graph_us)}"
              f"{'  ' + graph_err.splitlines()[0] if graph_err else ''}")
        print(f"  do_bench (flushed) : {flush_us:9.3f} us  {_quantized(flush_us)}")
        print(f"  hot_time           : {hot_us:9.3f} us  {_quantized(hot_us)}"
              f"{'   <-- IT FELL BACK' if _quantized(hot_us) == 'QUANTIZED' else ''}")
        del qkv

    report("before any model is built", a.ctx == "inference")

    if a.after:
        ref = _reference("ref_fallback")
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
        print(f"\n# primed {a.after}: tile {getattr(mdl, 'attn_tile', None)}")
        report(f"after {a.after} is built, primed and captured",
               a.ctx == "inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
