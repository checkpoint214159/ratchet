"""THE CENSUS finding 48 asked for, on the candidate that actually ships.

Finding 48 priced its config-10 proposal at +0.0048 of `weighted_score` by taking
attention's share of config 10 from "F-00 section 1" -- 17.6%, 42.69 us/forward -- and
multiplying by an op-level ratio of 1.200x. Two things make that number unsafe to build
on, and this probe replaces both:

  1. **The share was not measured on `v38_stream_fallback`.** The only device censuses
     in the record (finding 43) are of `v34_launch_bound` on configs 9 and 1. v38
     descends from v36, whose Triton projection GEMMs are 1.05-1.60x faster than the
     cuBLAS calls they replace and whose GELU epilogue deletes a whole kernel. Making
     everything *except* attention faster RAISES attention's share, so v34's census
     understates the lever -- but by an unknown amount, and a lever priced off the wrong
     denominator is not priced.

  2. **An op-level ratio measured under `do_bench` does not transfer into a replayed
     CUDA graph** [L33, L53]. `do_bench` flushes L2 and pays a launch; the attention
     kernel in the model runs L2-hot with its launch amortized inside a captured graph.
     Finding 48 measured 24.757 us for the incumbent op-level against an in-model
     10.67 us/call implied by its own table -- a 2.3x regime gap it then ignored when it
     applied the ratio in full. This probe measures the in-graph time directly, so the
     ceiling is priced on the number the model actually pays.

WHAT IT REPORTS
    per-kernel device time inside the replayed graph, one forward's worth, classified;
    attention's share; and the DILUTED CEILING in units of `weighted_score` under the
    deliberately generous assumption that the op-level ratio transfers in full.

INDICATIVE ONLY [L41]. A probe may propose; it may never conclude. Take the GPU lock.
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
from torch.profiler import ProfilerActivity, profile

from bench.matrix import BY_ID

# The number of forwards profiled. Divided out of every figure reported.
FORWARDS = 20
# Settling: finding 42's addendum measured ~130 calls after graph capture before the
# numbers mean anything. 200 is the same warmup abba.py uses.
WARMUP = 200


def _reference(tag: str):
    spec = importlib.util.spec_from_file_location(
        tag, REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def classify(name: str) -> str:
    """Bucket a kernel name into a stage of the forward.

    Deliberately explicit rather than clever: an unrecognised kernel lands in `other`
    and is printed by name, so a miscount is visible instead of silently absorbed.
    """
    n = name.lower()
    if "attn" in n or "flash" in n or "attention" in n or "fmha" in n:
        return "attention"
    if "gemm" in n or "cutlass" in n or "_proj_gemm" in n or "sgemm" in n:
        return "projection GEMM"
    if "layer_norm" in n or "layernorm" in n or "norm" in n:
        return "layernorm"
    if "gelu" in n:
        return "gelu"
    if "ffn" in n:
        return "ffn"
    if "memcpy" in n or "copy" in n:
        return "copy"
    if "elementwise" in n or "vectorized" in n or "add" in n:
        return "elementwise"
    return "other"


def census(config_id: int, arm: str) -> dict:
    from bench.candidates import REGISTRY

    cfg = BY_ID[config_id]
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    ref = _reference(f"ref_census_{config_id}")
    tcfg = ref.TransformerConfig(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, d_model=cfg.d_model,
        num_heads=cfg.heads, ffn_dim=cfg.ffn_dim, num_layers=cfg.layers,
        causal=cfg.causal)
    tcfg.validate()

    torch.manual_seed(1234)
    base = ref.BaselineTransformer(tcfg)
    mdl = REGISTRY[arm].build(ref.BaselineTransformer)(tcfg)
    ref.copy_model_weights(base, mdl)
    mdl = mdl.to(device=dev, dtype=torch.float32).eval()
    base = base.to(device=dev, dtype=torch.float32).eval()
    x, m = ref.generate_random_case(tcfg, dev, torch.float32, seed=1234,
                                    padding_ratio=0.0, input_scale=1.0)

    # CORRECTNESS BEFORE TIMING, and before profiling: a census of a wrong model is a
    # census of nothing.
    with torch.inference_mode():
        want = base(x, m)
        res = ref.compare_outputs(want, mdl(x, m), rtol=0.02, atol=0.002)
    del base

    # L36: assert the mechanism engaged. If `attn_used` is False the whole proposal is
    # aimed at a kernel that is not running.
    engaged = {
        "attn_used": getattr(mdl, "attn_used", None),
        "attn_tile": getattr(mdl, "attn_tile", None),
        "attn_reason": getattr(mdl, "attn_reason", None),
        "gemm_sites": list(getattr(mdl, "gemm_sites", ()) or ()),
        "launch_reason": getattr(mdl, "launch_reason", None),
        "stream_path": getattr(mdl, "stream_path", None),
    }

    with torch.inference_mode():
        for _ in range(WARMUP):
            mdl(x, m)
        torch.cuda.synchronize()

        # Wall time of the settled steady state, measured the way the grader does.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        torch.cuda.synchronize()
        for i in range(100):
            starts[i].record()
            mdl(x, m)
            ends[i].record()
        torch.cuda.synchronize()
        wall = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))[50]

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(FORWARDS):
                mdl(x, m)
            torch.cuda.synchronize()

    per_kernel = defaultdict(lambda: [0.0, 0])
    for ev in prof.key_averages():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = ev.self_cuda_time_total
        if t <= 0:
            continue
        per_kernel[ev.key][0] += t / FORWARDS
        per_kernel[ev.key][1] += ev.count / FORWARDS

    return {"config_id": config_id, "arm": arm, "wall_ms": wall,
            "passed": bool(res.passed), "max_abs": float(res.max_abs_error),
            "engaged": engaged, "per_kernel": dict(per_kernel)}


def report(c: dict) -> None:
    total = sum(v[0] for v in c["per_kernel"].values())
    print(f"\n{'='*90}")
    print(f"CENSUS  config {c['config_id']}  arm {c['arm']}")
    print(f"{'='*90}")
    print(f"  correctness   passed={c['passed']}  max_abs={c['max_abs']:.3e}")
    for k, v in c["engaged"].items():
        print(f"  {k:<16} {v}")
    print(f"  wall (median of 100, settled)   {c['wall_ms']*1e3:9.2f} us")
    print(f"  device time in the graph        {total:9.2f} us "
          f"({total/(c['wall_ms']*1e3)*100:.1f}% of wall)")
    print()
    buckets = defaultdict(lambda: [0.0, 0])
    rows = sorted(c["per_kernel"].items(), key=lambda kv: -kv[1][0])
    print(f"  {'kernel':<62}{'calls':>7}{'us/fwd':>9}{'%dev':>7}")
    for name, (t, n) in rows:
        b = classify(name)
        buckets[b][0] += t
        buckets[b][1] += n
        print(f"    {name[:60]:<60}{n:>7.1f}{t:>9.2f}{t/total*100:>7.1f}")
    print(f"\n  {'bucket':<62}{'calls':>7}{'us/fwd':>9}{'%dev':>7}{'%wall':>8}")
    for b, (t, n) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
        print(f"    {b:<60}{n:>7.1f}{t:>9.2f}{t/total*100:>7.1f}"
              f"{t/(c['wall_ms']*1e3)*100:>8.1f}")
    return buckets, total


def ceiling(c: dict, buckets, ratio: float, current_speedup: float,
            n_configs: int = 14) -> None:
    """The DILUTED figure, stated before any kernel is written [L33]."""
    attn = buckets["attention"][0]
    wall_us = c["wall_ms"] * 1e3
    saved = attn * (1.0 - 1.0 / ratio)
    frac = saved / wall_us
    new_speedup = current_speedup / (1.0 - frac)
    capped_old = min(current_speedup, 3.0)
    capped_new = min(new_speedup, 3.0)
    print(f"\n{'-'*90}\nDILUTED CEILING  (op-level ratio {ratio:.3f}x assumed to "
          f"transfer IN FULL -- it will not)\n{'-'*90}")
    print(f"  attention in the graph        {attn:9.2f} us/fwd "
          f"({attn/wall_us*100:.1f}% of wall)")
    print(f"  saved at {ratio:.3f}x            {saved:9.2f} us/fwd  "
          f"({frac*100:.2f}% of wall)")
    print(f"  config speedup    {current_speedup:.4f} -> {new_speedup:.4f}   "
          f"(capped {capped_old:.4f} -> {capped_new:.4f})")
    print(f"  Delta weighted_score          {(capped_new-capped_old)/n_configs:+.5f}")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, default=10)
    ap.add_argument("--arm", default="v38_stream_fallback")
    ap.add_argument("--ratio", type=float, default=1.200,
                    help="op-level ratio to price the ceiling at (finding 48: 1.200x)")
    ap.add_argument("--speedup", type=float, default=2.33,
                    help="the config's current speedup on the shipping candidate")
    a = ap.parse_args()

    from bench.gpu_lock import gpu_lock
    with gpu_lock(f"g40 census cfg{a.id}", timeout_s=14400):
        c = census(a.id, a.arm)
        buckets, _ = report(c)
        ceiling(c, buckets, a.ratio, a.speedup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
