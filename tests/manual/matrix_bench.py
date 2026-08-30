"""Measure the graphed fp16 kernel on the real competition matrix vs torch.compile + eager.

Ports the ben-branch comparison: geomean speedup over torch.compile, correctness vs the
fp32 eager baseline (abs<=2e-3 OR rel<=2e-2), on the announced causal / ffn_dim==d_model
configs. Every heavy op is a kernel from this repo (flash_attention + linear_tf32).
"""
import importlib.util
import sys
from pathlib import Path

import torch
import triton

BENCH = Path("benchmarks/reference/torch_transformer_benchmark.py").resolve()
spec = importlib.util.spec_from_file_location("tt_bench", BENCH)
mod = importlib.util.module_from_spec(spec)
sys.modules["tt_bench"] = mod
spec.loader.exec_module(mod)

from ratchet.kernels.graphed import graphed_forward

# (id, batch, d_model, heads, seq, layers, ffn_dim) -- subset of the 14, all causal.
CONFIGS = [
    (1, 64, 128, 4, 128, 4, 128),
    (2, 1, 128, 4, 128, 4, 128),      # launch-bound: CUDA graph should shine
    (13, 64, 128, 4, 1024, 4, 128),   # attention-heavy
    (8, 64, 1024, 4, 128, 4, 1024),   # head_dim 256
]
ONLY = set(int(a) for a in sys.argv[1:] if a.isdigit())


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def run(cid, B, d, H, S, L, F):
    cfg = mod.TransformerConfig(batch_size=B, seq_len=S, d_model=d, num_heads=H,
                                ffn_dim=F, num_layers=L, causal=True)
    dev = torch.device("cuda")
    base = mod.BaselineTransformer(cfg).to(dev).eval()
    x = torch.randn(B, S, d, device=dev)

    with torch.no_grad():
        ref = base(x, None).float()

    # torch.compile baseline (ben's comparison point)
    compiled = torch.compile(base, mode="max-autotune-no-cudagraphs")
    with torch.no_grad():
        for _ in range(3):
            compiled(x, None)

    # our candidate: graphed fp16 kernels
    cand = mod.UserOptimizedTransformer(cfg).to(dev).eval()
    mod.copy_model_weights(base, cand)
    cand.forward = graphed_forward.__get__(cand, type(cand))
    with torch.no_grad():
        got = cand(x, None).float()

    abs_err = (got - ref).abs()
    ok = bool(((abs_err <= 2e-3) | (abs_err <= 2e-2 * ref.abs())).all())

    t_eager = bench(lambda: base(x, None))
    t_comp = bench(lambda: compiled(x, None))
    t_ours = bench(lambda: cand(x, None))
    print(f"cfg{cid:<2} B{B} d{d} H{H} S{S} F{F}: correct={ok} | "
          f"eager={t_eager:.3f} compile={t_comp:.3f} ours={t_ours:.3f} ms | "
          f"ours vs compile={t_comp/t_ours:.2f}x  vs eager={t_eager/t_ours:.2f}x")
    return t_comp / t_ours


ratios = []
for c in CONFIGS:
    if ONLY and c[0] not in ONLY:
        continue
    try:
        ratios.append(run(*c))
    except Exception as e:
        print(f"cfg{c[0]}: FAILED {type(e).__name__}: {str(e)[:150]}")
if ratios:
    geo = torch.tensor(ratios).log().mean().exp().item()
    print(f"\ngeomean vs torch.compile = {geo:.3f}x  ({len(ratios)} configs)")
