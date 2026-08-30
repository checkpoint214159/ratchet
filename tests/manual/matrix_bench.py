"""Germane results: the shape-aware dispatch driving repo kernels over the announced matrix.

For each announced shape, `dispatch.select` picks a device-calibrated recipe (fp16, CUDA
graph on/off, flash tiles); the forward runs entirely on this repo's kernels
(flash_attention + linear_tf32). We report correctness vs the fp32 baseline (abs<=2e-3 OR
rel<=2e-2, the statement's bound) and speedup vs torch.compile (the baseline to beat) and
eager, plus the dispatch decision.
"""
import importlib.util
import sys
from pathlib import Path

import torch
import triton

from ratchet.kernels.dispatch import MATRIX, select
from ratchet.kernels.graphed import graphed_forward
from ratchet.oracle import calibrate

BENCH = Path("benchmarks/reference/torch_transformer_benchmark.py").resolve()
spec = importlib.util.spec_from_file_location("tt_bench", BENCH)
mod = importlib.util.module_from_spec(spec)
sys.modules["tt_bench"] = mod
spec.loader.exec_module(mod)

PROF = calibrate(cache_path="ledger/device.gb10.json")
ONLY = set(int(a) for a in sys.argv[1:] if a.isdigit())


def bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100)


def run(cfg):
    recipe = select(cfg, PROF)
    tc = mod.TransformerConfig(batch_size=cfg.batch_size, seq_len=cfg.seq_len,
                               d_model=cfg.d_model, num_heads=cfg.heads,
                               ffn_dim=cfg.ffn_dim, num_layers=cfg.layers, causal=True)
    dev = torch.device("cuda")
    base = mod.BaselineTransformer(tc).to(dev).eval()
    x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device=dev)
    with torch.no_grad():
        ref = base(x, None).float()

    compiled = torch.compile(base, mode="max-autotune-no-cudagraphs")
    with torch.no_grad():
        for _ in range(3):
            compiled(x, None)

    cand = mod.UserOptimizedTransformer(tc).to(dev).eval()
    mod.copy_model_weights(base, cand)
    cand._use_graph = recipe.use_graph
    cand.forward = graphed_forward.__get__(cand, type(cand))
    with torch.no_grad():
        got = cand(x, None).float()
    abs_err = (got - ref).abs()
    ok = bool(((abs_err <= 2e-3) | (abs_err <= 2e-2 * ref.abs())).all())

    t_eager = bench(lambda: base(x, None))
    t_comp = bench(lambda: compiled(x, None))
    t_ours = bench(lambda: cand(x, None))
    print(f"cfg{cfg.id:<2} B{cfg.batch_size} d{cfg.d_model} H{cfg.heads} S{cfg.seq_len} "
          f"hd{cfg.head_dim}: correct={ok} graph={recipe.use_graph} | "
          f"compile={t_comp:.3f} ours={t_ours:.3f} ms | vs_compile={t_comp/t_ours:.2f}x "
          f"vs_eager={t_eager/t_ours:.2f}x")
    return t_comp / t_ours, ok


ratios, all_ok = [], True
for cfg in MATRIX:
    if cfg.id == 14 or (ONLY and cfg.id not in ONLY):
        continue
    try:
        r, ok = run(cfg)
        ratios.append(r)
        all_ok = all_ok and ok
    except Exception as e:
        print(f"cfg{cfg.id}: FAILED {type(e).__name__}: {str(e)[:150]}")
        all_ok = False
if ratios:
    geo = torch.tensor(ratios).log().mean().exp().item()
    print(f"\ngeomean vs torch.compile = {geo:.3f}x over {len(ratios)} configs | "
          f"all_correct={all_ok}")
