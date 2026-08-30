"""Profile BOTH arms in ONE process, so the device census is a comparison and not two.

WHY THIS EXISTS
---------------
`probe_census.py` run separately on v38 and then on v40 produced a contradiction:

    arm                     wall      device    attention (x4)
    v38_stream_fallback   251.90 us  222.50 us   44.14  (_attn_single_tile)
    v40_looped_attn       237.57     223.64      45.98  (_attn_looped)

v40 is 14.3 us faster at the wall and 1.1 us SLOWER on the device, with its attention
kernel apparently 1.8 us/fwd worse -- while `bench/abba.py` says v40 is 1.0366x on this
config, replicated to 0.02% across two runs.

Those two censuses are **different processes**, which is exactly the comparison finding 42
showed is unsafe: the wall of a sub-millisecond config moves 5-39% between runs of
byte-identical code, and a cross-process device total inherits whatever the host was
doing. So the contradiction may be entirely an artefact of comparing two runs.

This probe removes that variable and nothing else: both models are built and settled in
one process, then profiled back to back, interleaved so neither gets the cold slot. If
the attention row still fails to move, the end-to-end win is real but is NOT the looped
kernel running faster, and this candidate's stated mechanism is wrong even though its
number is right.

INDICATIVE ONLY [L41]. Take the GPU lock.
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
from bench.probes.g40_attn_loop.probe_census import classify

ARMS = ("v38_stream_fallback", "v40_looped_attn")
FORWARDS = 20
WARMUP = 200
ROUNDS = 3


def _reference(tag: str):
    spec = importlib.util.spec_from_file_location(
        tag, REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def _wall(model, x, m, n=100):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    with torch.inference_mode():
        torch.cuda.synchronize()
        for i in range(n):
            starts[i].record()
            model(x, m)
            ends[i].record()
        torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in zip(starts, ends))[n // 2]


def _profile(model, x, m):
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(FORWARDS):
            model(x, m)
        torch.cuda.synchronize()
    per = defaultdict(lambda: [0.0, 0])
    for ev in prof.key_averages():
        if ev.device_type != torch.autograd.DeviceType.CUDA:
            continue
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = ev.self_cuda_time_total
        if t > 0:
            per[ev.key][0] += t / FORWARDS
            per[ev.key][1] += ev.count / FORWARDS
    return dict(per)


def main() -> int:
    from bench.candidates import REGISTRY
    from bench.gpu_lock import gpu_lock

    config_id = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    c = BY_ID[config_id]
    ref = _reference("ref_pair")
    cfg = ref.TransformerConfig(
        batch_size=c.batch_size, seq_len=c.seq_len, d_model=c.d_model,
        num_heads=c.heads, ffn_dim=c.ffn_dim, num_layers=c.layers, causal=c.causal)
    cfg.validate()
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    with gpu_lock("g40 paired census", timeout_s=7200):
        torch.manual_seed(1234)
        base = ref.BaselineTransformer(cfg)
        models = {}
        for arm in ARMS:
            mdl = REGISTRY[arm].build(ref.BaselineTransformer)(cfg)
            ref.copy_model_weights(base, mdl)
            models[arm] = mdl.to(device=dev, dtype=torch.float32).eval()
        base = base.to(device=dev, dtype=torch.float32).eval()
        x, m = ref.generate_random_case(cfg, dev, torch.float32, seed=1234,
                                        padding_ratio=0.0, input_scale=1.0)

        with torch.inference_mode():
            want = base(x, m)
            for arm in ARMS:
                r = ref.compare_outputs(want, models[arm](x, m), rtol=0.02, atol=0.002)
                assert r.passed, f"{arm} failed correctness: {r.max_abs_error:.3e}"
        del base

        for arm in ARMS:                       # settle BOTH before timing either
            with torch.inference_mode():
                for _ in range(WARMUP):
                    models[arm](x, m)
        torch.cuda.synchronize()

        walls = {a: [] for a in ARMS}
        cens = {}
        for r in range(ROUNDS):
            order = ARMS if r % 2 == 0 else tuple(reversed(ARMS))
            for arm in order:
                walls[arm].append(_wall(models[arm], x, m))
                cens[arm] = _profile(models[arm], x, m)

        print(f"\nconfig {config_id}: paired census, {ROUNDS} ABBA rounds, "
              f"{WARMUP} warmup, {FORWARDS} profiled forwards\n")
        for arm in ARMS:
            mdl = models[arm]
            print(f"  {arm:<22} form={getattr(mdl,'attn_form','single_tile'):<12}"
                  f"tile={getattr(mdl,'attn_tile',None)}")
        print(f"\n  {'arm':<22}{'wall us (min of rounds)':>24}{'device us':>12}"
              f"{'gap us':>9}")
        tot = {}
        for arm in ARMS:
            tot[arm] = sum(v[0] for v in cens[arm].values())
            w = min(walls[arm])
            print(f"  {arm:<22}{w:>24.2f}{tot[arm]:>12.2f}{w - tot[arm]:>9.2f}")
        print(f"  {'ratio v38/v40':<22}"
              f"{min(walls[ARMS[0]])/min(walls[ARMS[1]]):>24.4f}"
              f"{tot[ARMS[0]]/tot[ARMS[1]]:>12.4f}")

        print(f"\n  {'bucket':<20}{'v38 us/fwd':>12}{'v40 us/fwd':>12}{'delta':>10}")
        buckets = {}
        for arm in ARMS:
            b = defaultdict(float)
            for name, (t, _) in cens[arm].items():
                b[classify(name)] += t
            buckets[arm] = b
        for k in sorted(set(buckets[ARMS[0]]) | set(buckets[ARMS[1]])):
            a, bb = buckets[ARMS[0]][k], buckets[ARMS[1]][k]
            print(f"  {k:<20}{a:>12.2f}{bb:>12.2f}{bb - a:>+10.2f}")

        print(f"\n  attention kernels, per call:")
        for arm in ARMS:
            for name, (t, n) in sorted(cens[arm].items(), key=lambda kv: -kv[1][0]):
                if classify(name) == "attention":
                    print(f"    {arm:<22}{name[:40]:<42}{n:>5.1f} calls "
                          f"{t:>8.2f} us/fwd  {t/max(n,1):>7.3f} us/call")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
