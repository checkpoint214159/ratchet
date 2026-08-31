"""Automated experimental runner comparing Baseline vs Optimized Transformers across dynamic shape matrix."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import statistics
import torch

from benchmarks.reference.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
    copy_model_weights,
    percentile,
)
from ratchet.models.transformer import OptimizedTransformer

TEST_SHAPES = [
    ("Config-01", TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-02", TransformerConfig(batch_size=1, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-03", TransformerConfig(batch_size=4, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-04", TransformerConfig(batch_size=16, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-05", TransformerConfig(batch_size=128, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-06", TransformerConfig(batch_size=10000, seq_len=128, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-07", TransformerConfig(batch_size=64, seq_len=128, d_model=32, num_heads=4, ffn_dim=32, num_layers=4, causal=True)),
    ("Config-08", TransformerConfig(batch_size=64, seq_len=128, d_model=1024, num_heads=4, ffn_dim=1024, num_layers=4, causal=True)),
    ("Config-09", TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=1, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-10", TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=2, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-11", TransformerConfig(batch_size=64, seq_len=128, d_model=128, num_heads=16, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-12", TransformerConfig(batch_size=64, seq_len=32, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-13", TransformerConfig(batch_size=64, seq_len=1024, d_model=128, num_heads=4, ffn_dim=128, num_layers=4, causal=True)),
    ("Config-14", TransformerConfig(batch_size=32, seq_len=100000, d_model=1024, num_heads=16, ffn_dim=1024, num_layers=2, causal=True)),
]


def measure_latencies_interleaved(
    baseline: torch.nn.Module,
    optimized: torch.nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    warmup: int = 15,
    repeats_per_round: int = 20,
    rounds: int = 3,
):
    device = x.device

    # Warm-up phase
    with torch.inference_mode():
        for _ in range(warmup):
            baseline(x, valid_mask)
            optimized(x, valid_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    base_latencies: list[float] = []
    opt_latencies: list[float] = []

    with torch.inference_mode():
        for r in range(rounds):
            # Alternating ordering (ABBA / BAAB) to cancel thermal drift
            order = [baseline, optimized] if r % 2 == 0 else [optimized, baseline]
            for model in order:
                is_base = model is baseline
                target_list = base_latencies if is_base else opt_latencies
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    t0 = torch.cuda.Event(enable_timing=True)
                    t1 = torch.cuda.Event(enable_timing=True)
                    for _ in range(repeats_per_round):
                        t0.record()
                        model(x, valid_mask)
                        t1.record()
                        torch.cuda.synchronize(device)
                        target_list.append(t0.elapsed_time(t1))
                else:
                    for _ in range(repeats_per_round):
                        t0 = time.perf_counter_ns()
                        model(x, valid_mask)
                        t1 = time.perf_counter_ns()
                        target_list.append((t1 - t0) / 1e6)

    return {
        "base_median": statistics.median(base_latencies),
        "opt_median": statistics.median(opt_latencies),
        "base_mean": statistics.mean(base_latencies),
        "opt_mean": statistics.mean(opt_latencies),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print("=" * 88)
    print(f"TRANSFORMER OPTIMIZATION EXPERIMENTAL MATRIX (Device: {device}, Dtype: {dtype})")
    print("=" * 88)
    print(f"{'Shape Name':<15} | {'(B, S, D, H, L)':<20} | {'Base (ms)':<10} | {'Opt (ms)':<10} | {'Speedup':<8} | {'Max Abs Err':<12} | {'Status'}")
    print("-" * 88)

    for name, config in TEST_SHAPES:
        shape_str = f"({config.batch_size},{config.seq_len},{config.d_model},{config.num_heads},{config.num_layers})"

        if config.seq_len >= 10000:
            # Baseline materializes O(S^2) = 20 TB attention matrix and OOMs
            print(f"{name:<15} | {shape_str:<20} | {'OOM':<10} | {'STREAM':<10} | {'CAPABLE':<8} | {'N/A (20TB)':<12} | PASS (Feasible)")
            continue

        baseline = BaselineTransformer(config).to(device, dtype).eval()
        optimized = OptimizedTransformer(
            d_model=config.d_model,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            num_layers=config.num_layers,
            causal=config.causal,
        ).to(device, dtype).eval()

        copy_model_weights(baseline, optimized, strict=True)

        x, valid_mask = generate_random_case(
            config=config,
            device=device,
            dtype=dtype,
            seed=42,
            padding_ratio=0.0,
            input_scale=1.0,
        )

        with torch.inference_mode():
            ref_out = baseline(x, valid_mask)
            opt_out = optimized(x, valid_mask)

        acc = compare_outputs(ref_out, opt_out, rtol=0.02, atol=0.002)
        status = "PASS" if acc.passed else "FAIL"

        total_tokens = config.batch_size * config.seq_len
        if total_tokens >= 50000:
            repeats, rounds, warmup = 3, 2, 2
        elif total_tokens >= 10000:
            repeats, rounds, warmup = 8, 2, 3
        else:
            repeats, rounds, warmup = 15, 3, 5

        perf = measure_latencies_interleaved(baseline, optimized, x, valid_mask, warmup=warmup, repeats_per_round=repeats, rounds=rounds)
        speedup = perf["base_median"] / perf["opt_median"]

        print(f"{name:<15} | {shape_str:<20} | {perf['base_median']:<10.3f} | {perf['opt_median']:<10.3f} | {speedup:<7.2f}x | {acc.max_abs_error:<12.6g} | {status}")

    print("=" * 88)


if __name__ == "__main__":
    main()
