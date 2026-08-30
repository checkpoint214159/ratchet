"""Standardized benchmark harness for comparable synthetic experiments.

This script provides a common stress profile for all explored methods so their
results can be compared under the same conditions across:
- plain PageRank
- failure-weighted PageRank
- tuning-pressure
- cross-hardware transfer scenarios
- FP8 / reduced-precision scenarios

The benchmark is intentionally synthetic and CPU-only so that multiple methods can
be compared on the same basis without requiring full kernel execution.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ratchet.experiments.pagerank_difficulty_forecasting import (
    failure_weighted_pagerank,
    generate_larger_config_space,
    make_graph,
    pagerank,
    rank_quality,
    true_difficulty,
    tuning_pressure_scores,
)


DEFAULT_HARDWARE_PROFILES = ["cuda", "hip", "cpu"]
DEFAULT_QUANTIZATION_MODES = ["fp32", "fp16", "fp8"]


def hardware_profile_bias(
    hardware_profile: str,
    cfg: tuple[int, int, int, int],
) -> float:
    """Synthetic hardware-specific stress modifiers."""
    block_m, block_n, num_warps, num_stages = cfg
    base = 0.0
    profile_map = {
        "cpu": {"shared_memory": 0.05, "registers": 0.06, "warp": 0.04, "stages": 0.03},
        "cuda": {"shared_memory": 0.10, "registers": 0.12, "warp": 0.08, "stages": 0.07},
        "hip": {"shared_memory": 0.12, "registers": 0.09, "warp": 0.10, "stages": 0.09},
        "xpu": {"shared_memory": 0.08, "registers": 0.10, "warp": 0.07, "stages": 0.06},
    }
    profile = profile_map.get(hardware_profile, profile_map["cpu"])

    base += profile["shared_memory"] * max(0.0, (block_m * block_n) / 4096.0 - 1.0)
    base += profile["registers"] * (num_warps / 8.0)
    base += profile["warp"] * max(0.0, (num_warps - 8) / 8.0)
    base += profile["stages"] * max(0.0, (num_stages - 3) / 2.0)
    return base


def quantization_penalty(
    quantization_mode: str,
    cfg: tuple[int, int, int, int],
) -> float:
    """Synthetic reduced-precision stress modifier."""
    block_m, block_n, num_warps, num_stages = cfg
    mode_map = {
        "fp32": 0.0,
        "fp16": 0.08,
        "fp8": 0.18,
        "bf16": 0.10,
    }
    penalty = mode_map.get(quantization_mode, 0.0)
    penalty += 0.05 * max(0.0, (block_m * block_n) / 4096.0 - 1.0)
    penalty += 0.04 * (num_warps / 16.0)
    penalty += 0.03 * max(0.0, num_stages - 3)
    return penalty


def synthetic_difficulty(
    cfg: tuple[int, int, int, int],
    hardware_profile: str = "cuda",
    quantization_mode: str = "fp32",
) -> float:
    """Ground-truth difficulty under a common stress model."""
    difficulty = true_difficulty(cfg)
    difficulty += hardware_profile_bias(hardware_profile, cfg)
    difficulty += quantization_penalty(quantization_mode, cfg)
    return max(0.0, difficulty)


def summarize_method(
    configs: list[tuple[int, int, int, int]],
    scores: dict[tuple[int, int, int, int], float],
    true_scores: dict[tuple[int, int, int, int], float],
) -> dict[str, float]:
    k = min(10, max(1, len(configs) // 20))
    precision_at_k, spearman = rank_quality(scores, true_scores, k=k)
    cutoff = max(1, len(configs) // 5)
    sorted_true = sorted(true_scores, key=true_scores.get, reverse=True)
    hard_region = set(sorted_true[:cutoff])
    predicted_top = set(sorted(scores, key=scores.get, reverse=True)[:cutoff])
    overlap = len(predicted_top & hard_region)
    random_overlap = (cutoff * len(hard_region)) / max(1, len(configs))
    gain_vs_random = (overlap / max(1, cutoff)) - (random_overlap / max(1, cutoff))
    return {
        "precision_at_k": float(precision_at_k),
        "spearman": float(spearman),
        "gain_vs_random": float(gain_vs_random),
        "overlap": float(overlap),
        "cutoff": float(cutoff),
    }


def evaluate_methods_under_stress(
    configs: list[tuple[int, int, int, int]],
    seed: int,
    hardware_profile: str = "cuda",
    quantization_mode: str = "fp32",
    noise_scale: float = 0.12,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    true_scores = {
        cfg: max(
            0.0,
            synthetic_difficulty(cfg, hardware_profile, quantization_mode)
            + rng.uniform(-noise_scale, noise_scale),
        )
        for cfg in configs
    }
    graph = make_graph(configs)

    plain_scores = pagerank(graph)
    weighted_scores = failure_weighted_pagerank(graph, true_scores)
    pressure_scores = tuning_pressure_scores(configs, true_scores)
    random_scores = {cfg: rng.random() for cfg in configs}

    return {
        "plain_pagerank": summarize_method(configs, plain_scores, true_scores),
        "failure_weighted_pagerank": summarize_method(configs, weighted_scores, true_scores),
        "tuning_pressure": summarize_method(configs, pressure_scores, true_scores),
        "random_baseline": summarize_method(configs, random_scores, true_scores),
    }


def aggregate_trial_results(
    trials: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    methods: dict[str, dict[str, list[float]]] = {}
    for trial in trials:
        for method_name, metrics in trial.items():
            methods.setdefault(method_name, {})
            for metric_name, value in metrics.items():
                methods[method_name].setdefault(metric_name, []).append(float(value))

    summary: dict[str, dict[str, float]] = {}
    for method_name, metric_values in methods.items():
        summary[method_name] = {
            metric_name: statistics.median(values)
            for metric_name, values in metric_values.items()
        }
    return summary


def benchmark_scenarios(
    hardware_profiles: list[str] | None = None,
    quantization_modes: list[str] | None = None,
    seeds: int = 20,
    max_runtime_seconds: int = 1800,
    block_m_count: int = 4,
    block_n_count: int = 5,
    warp_count: int = 4,
    stage_count: int = 4,
    noise_scale: float = 0.12,
) -> dict[str, object]:
    hardware_profiles = hardware_profiles or DEFAULT_HARDWARE_PROFILES
    quantization_modes = quantization_modes or DEFAULT_QUANTIZATION_MODES

    scenario_entries: list[dict[str, object]] = []
    deadline = time.monotonic() + max_runtime_seconds

    for hardware_profile in hardware_profiles:
        for quantization_mode in quantization_modes:
            scenario_trials: list[dict[str, dict[str, float]]] = []
            scenario_name = f"{hardware_profile}/{quantization_mode}"
            for seed in range(seeds):
                if time.monotonic() >= deadline:
                    break
                configs = generate_larger_config_space(
                    block_m_count,
                    block_n_count,
                    warp_count,
                    stage_count,
                )
                trial = evaluate_methods_under_stress(
                    configs,
                    seed,
                    hardware_profile=hardware_profile,
                    quantization_mode=quantization_mode,
                    noise_scale=noise_scale,
                )
                scenario_trials.append(trial)
            summary = aggregate_trial_results(scenario_trials)
            scenario_entries.append(
                {
                    "scenario_name": scenario_name,
                    "hardware_profile": hardware_profile,
                    "quantization_mode": quantization_mode,
                    "trial_count": len(scenario_trials),
                    "summary": summary,
                }
            )

    return {
        "hardware_profiles": hardware_profiles,
        "quantization_modes": quantization_modes,
        "seeds": seeds,
        "max_runtime_seconds": max_runtime_seconds,
        "scenarios": scenario_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardized synthetic benchmark across hardware and precision stress scenarios."
    )
    parser.add_argument("--hardware-profiles", nargs="*", default=DEFAULT_HARDWARE_PROFILES)
    parser.add_argument("--quantization-modes", nargs="*", default=DEFAULT_QUANTIZATION_MODES)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--max-runtime-seconds", type=int, default=1800)
    parser.add_argument("--block-m-count", type=int, default=4)
    parser.add_argument("--block-n-count", type=int, default=5)
    parser.add_argument("--warp-count", type=int, default=4)
    parser.add_argument("--stage-count", type=int, default=4)
    parser.add_argument("--noise-scale", type=float, default=0.12)
    parser.add_argument(
        "--output",
        type=str,
        default="research/experiment_summaries/standardized_method_benchmark.json",
    )
    parser.add_argument("--print-every", type=int, default=5)
    return parser.parse_args()


def run_standardized_benchmark(args: argparse.Namespace) -> dict[str, object]:
    payload = benchmark_scenarios(
        hardware_profiles=args.hardware_profiles,
        quantization_modes=args.quantization_modes,
        seeds=args.seeds,
        max_runtime_seconds=args.max_runtime_seconds,
        block_m_count=args.block_m_count,
        block_n_count=args.block_n_count,
        warp_count=args.warp_count,
        stage_count=args.stage_count,
        noise_scale=args.noise_scale,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("=" * 88)
    print("STANDARDIZED DIFFICULTY-BENCHMARK SUMMARY")
    print("=" * 88)
    for entry in payload["scenarios"]:
        scenario = entry["scenario_name"]
        summary = entry["summary"]
        print(
            f"{scenario}: "
            f"plain={summary['plain_pagerank']['spearman']:.3f}, "
            f"weighted={summary['failure_weighted_pagerank']['spearman']:.3f}, "
            f"pressure={summary['tuning_pressure']['spearman']:.3f}, "
            f"random={summary['random_baseline']['spearman']:.3f}"
        )
    print(f"Results written to: {output_path}")
    print("=" * 88)
    return payload


if __name__ == "__main__":
    run_standardized_benchmark(parse_args())
