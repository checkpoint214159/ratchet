"""Run a multi-seed synthetic benchmark over the explored difficulty methods.

This script is designed for a few-hour screening run. It keeps the search space
large enough to expose weak ranking signals but still stays entirely synthetic and
CPU-only, which makes it appropriate for comparing:
- plain PageRank
- failure-weighted PageRank
- tuning-pressure / neighborhood risk propagation
- a random baseline

The script stops gracefully when a configured time budget is reached and writes a
JSON summary to disk.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

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


def evaluate_methods(
    configs: list[tuple[int, int, int, int]],
    seed: int,
    noise_scale: float = 0.12,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    true_scores = {
        cfg: max(
            0.0,
            true_difficulty(cfg) + rng.uniform(-noise_scale, noise_scale),
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
        "failure_weighted_pagerank": summarize_method(
            configs, weighted_scores, true_scores
        ),
        "tuning_pressure": summarize_method(configs, pressure_scores, true_scores),
        "random_baseline": summarize_method(configs, random_scores, true_scores),
    }


def aggregate_trial_results(trials: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 2–4 hour synthetic benchmark of the explored difficulty methods.")
    parser.add_argument("--max-runtime-seconds", type=int, default=10800, help="Hard time budget in seconds. Default is 3 hours.")
    parser.add_argument("--seeds", type=int, default=30, help="Target number of random seeds to evaluate before stopping.")
    parser.add_argument("--block-m-count", type=int, default=4, help="Number of block_M values in the synthetic search grid.")
    parser.add_argument("--block-n-count", type=int, default=5, help="Number of block_N values in the synthetic search grid.")
    parser.add_argument("--warp-count", type=int, default=4, help="Number of warp values in the synthetic search grid.")
    parser.add_argument("--stage-count", type=int, default=4, help="Number of stage values in the synthetic search grid.")
    parser.add_argument("--noise-scale", type=float, default=0.12, help="Noise scale for the synthetic ground-truth difficulty objective.")
    parser.add_argument("--output", type=str, default="research/experiment_summaries/method_benchmark.json", help="Where to write the final JSON summary.")
    parser.add_argument("--print-every", type=int, default=5, help="Print progress every N seeds.")
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    deadline = time.monotonic() + args.max_runtime_seconds
    trials: list[dict[str, dict[str, float]]] = []
    completed = 0
    for seed in range(args.seeds):
        if time.monotonic() >= deadline:
            break

        configs = generate_larger_config_space(
            args.block_m_count,
            args.block_n_count,
            args.warp_count,
            args.stage_count,
        )
        trial = evaluate_methods(configs, seed, noise_scale=args.noise_scale)
        trials.append(trial)
        completed += 1

        if args.print_every and completed % args.print_every == 0:
            summary = aggregate_trial_results(trials)
            print(
                f"seed={seed} completed={completed} "
                f"plain={summary['plain_pagerank']['spearman']:.3f} "
                f"weighted={summary['failure_weighted_pagerank']['spearman']:.3f} "
                f"pressure={summary['tuning_pressure']['spearman']:.3f}"
            )

    if not trials:
        raise RuntimeError("No trials were evaluated before the runtime deadline expired.")

    summary = aggregate_trial_results(trials)
    payload = {
        "completed_trials": completed,
        "max_runtime_seconds": args.max_runtime_seconds,
        "block_m_count": args.block_m_count,
        "block_n_count": args.block_n_count,
        "warp_count": args.warp_count,
        "stage_count": args.stage_count,
        "noise_scale": args.noise_scale,
        "summary": summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print("=" * 88)
    print("LARGE-SCALE DIFFICULTY MODEL BENCHMARK")
    print("=" * 88)
    for method_name, metrics in summary.items():
        print(
            f"{method_name}: "
            f"spearman={metrics['spearman']:.3f}, "
            f"precision_at_k={metrics['precision_at_k']:.3f}, "
            f"gain_vs_random={metrics['gain_vs_random']:.3f}, "
            f"overlap={metrics['overlap']:.0f}/{metrics['cutoff']:.0f}"
        )
    print(f"Results written to: {output_path}")
    print("=" * 88)
    return payload


if __name__ == "__main__":
    run_benchmark(parse_args())
