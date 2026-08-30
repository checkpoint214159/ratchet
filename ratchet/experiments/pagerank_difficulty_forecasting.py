"""Difficulty forecasting experiments for sparse tuning landscapes.

This module evaluates a few ideas that were explored in sequence:
- plain PageRank centrality over a local config graph,
- failure-weighted PageRank with risk-aware propagation,
- a tuning-pressure estimator that blends local risk and neighbor pressure.

The experiments are intentionally synthetic and deterministic so that they can be
scaled up cheaply and compared against the same ground-truth objective.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Iterable

import numpy as np


HEAD_DIM = 64
SHARED_MEMORY_PER_BLOCK = 99 * 1024
MAX_THREADS_PER_BLOCK = 1024
MAX_WARPS_PER_BLOCK = 32


def generate_config_space() -> list[tuple[int, int, int, int]]:
    block_m_values = [32, 64, 128, 256]
    block_n_values = [32, 64, 128]
    num_warps_values = [4, 8, 16, 32]
    num_stages_values = [2, 3, 4, 5]
    return list(product(block_m_values, block_n_values, num_warps_values, num_stages_values))


def generate_larger_config_space(
    block_m_count: int = 4,
    block_n_count: int = 5,
    warp_count: int = 4,
    stage_count: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Generate a larger synthetic search grid while preserving a small footprint.

    The counts are used to produce a geometric progression rather than a dense grid,
    which gives a more realistic spread of large, moderate, and small tile sizes.
    """
    block_m_values = [32 * (2**i) for i in range(block_m_count)]
    block_n_values = [32 * (2**i) for i in range(block_n_count)]
    num_warps_values = [4 * (2**i) for i in range(warp_count)]
    num_stages_values = [2 + i for i in range(stage_count)]
    return list(product(block_m_values, block_n_values, num_warps_values, num_stages_values))


def true_difficulty(config: tuple[int, int, int, int]) -> float:
    """Synthetic ground-truth difficulty for this config.

    Higher score means more likely to be difficult to tune or compile.
    """
    block_m, block_n, num_warps, num_stages = config
    shared_memory_need = (
        block_m * HEAD_DIM * 2
        + num_stages * (2 * block_n * HEAD_DIM * 2)
    )
    shared_memory_ratio = shared_memory_need / SHARED_MEMORY_PER_BLOCK

    register_pressure = ((block_m + block_n) * num_warps) / MAX_THREADS_PER_BLOCK
    warp_pressure = num_warps / MAX_WARPS_PER_BLOCK

    score = 0.6 * max(0.0, shared_memory_ratio - 0.5)
    score += 0.8 * max(0.0, register_pressure - 0.7)
    score += 0.5 * max(0.0, warp_pressure - 0.7)
    score += 0.2 * max(0, num_stages - 3)
    return score


def make_graph(
    configs: Iterable[tuple[int, int, int, int]],
) -> dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    """Build a local adjacency graph for a config-space neighborhood."""
    config_list = list(configs)
    graph = {cfg: [] for cfg in config_list}

    for i, cfg in enumerate(config_list):
        for j in range(i + 1, len(config_list)):
            other = config_list[j]
            parameter_distance = sum(abs(a - b) for a, b in zip(cfg, other))
            if parameter_distance <= 3:
                graph[cfg].append(other)
                graph[other].append(cfg)

    for cfg in config_list:
        if not graph[cfg]:
            closest = min(
                config_list,
                key=lambda other: (sum(abs(a - b) for a, b in zip(cfg, other)), other),
            )
            if closest != cfg:
                graph[cfg].append(closest)
                graph[closest].append(cfg)

    return graph


def pagerank(
    graph: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]],
    damping: float = 0.85,
    steps: int = 100,
) -> dict[tuple[int, int, int, int], float]:
    nodes = list(graph)
    n = len(nodes)
    if n == 0:
        return {}

    scores = {node: 1.0 / n for node in nodes}
    out_degree = {node: len(graph[node]) for node in nodes}

    for _ in range(steps):
        new_scores = {node: (1.0 - damping) / n for node in nodes}
        for node in nodes:
            if out_degree[node] == 0:
                continue
            share = scores[node] / out_degree[node]
            for neighbor in graph[node]:
                new_scores[neighbor] += damping * share
        scores = new_scores

    return scores


def failure_weighted_pagerank(
    graph: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]],
    difficulty_scores: dict[tuple[int, int, int, int], float],
    damping: float = 0.85,
    steps: int = 100,
) -> dict[tuple[int, int, int, int], float]:
    """Graph risk prior: hard regions send influence to neighboring configs."""
    nodes = list(graph)
    if not nodes:
        return {}

    raw_risk = {node: max(0.0, min(1.0, float(difficulty_scores.get(node, 0.0)))) for node in nodes}
    total_risk = sum(raw_risk.values())
    if total_risk > 0:
        scores = {node: raw_risk[node] / total_risk for node in nodes}
    else:
        scores = {node: 1.0 / len(nodes) for node in nodes}

    def edge_weight(node: tuple[int, int, int, int], neighbor: tuple[int, int, int, int]) -> float:
        distance = sum(abs(a - b) for a, b in zip(node, neighbor))
        similarity = 1.0 / (1.0 + distance)
        risk_factor = 0.5 + 0.5 * (raw_risk.get(node, 0.0) + raw_risk.get(neighbor, 0.0))
        return similarity * risk_factor

    for _ in range(steps):
        out_total = {
            node: sum(edge_weight(node, neighbor) for neighbor in graph.get(node, []))
            for node in nodes
        }
        new_scores = {node: (1.0 - damping) * raw_risk.get(node, 0.0) for node in nodes}
        for node in nodes:
            if not graph.get(node):
                continue
            total = out_total[node]
            if total <= 0:
                continue
            for neighbor in graph[node]:
                weight = edge_weight(node, neighbor) / total
                new_scores[neighbor] += damping * scores[node] * weight
        total_mass = sum(new_scores.values())
        if total_mass <= 0:
            total_mass = len(nodes)
        scores = {node: value / total_mass for node, value in new_scores.items()}
    return scores


def tuning_pressure_scores(
    configs: list[tuple[int, int, int, int]],
    true_scores: dict[tuple[int, int, int, int], float],
) -> dict[tuple[int, int, int, int], float]:
    """A simple neighboring-risk estimator that blends local difficulty and local spread."""
    graph = make_graph(configs)
    local = {cfg: max(0.0, min(1.0, true_scores.get(cfg, 0.0))) for cfg in configs}
    scores = local.copy()
    for _ in range(12):
        next_scores = {}
        for cfg in configs:
            neighbors = graph.get(cfg, [])
            if neighbors:
                neighbor_pressure = sum(scores.get(nbr, 0.0) for nbr in neighbors) / len(neighbors)
            else:
                neighbor_pressure = 0.0
            next_scores[cfg] = 0.7 * local[cfg] + 0.3 * neighbor_pressure
        scores = next_scores
    return scores


def rank_quality(
    predicted_scores: dict[tuple[int, int, int, int], float],
    true_scores: dict[tuple[int, int, int, int], float],
    k: int = 10,
) -> tuple[float, float]:
    """Return Precision@k and Spearman rank correlation."""
    predicted_order = sorted(predicted_scores, key=predicted_scores.get, reverse=True)
    true_order = sorted(true_scores, key=true_scores.get, reverse=True)

    topk_pred = set(predicted_order[:k])
    topk_true = set(true_order[:k])
    precision_at_k = len(topk_pred & topk_true) / max(1, k)

    pred_ranks = {node: rank for rank, node in enumerate(predicted_order)}
    true_ranks = {node: rank for rank, node in enumerate(true_order)}
    common = sorted(set(pred_ranks) & set(true_ranks))
    if len(common) < 2:
        spearman = 0.0
    else:
        x = np.array([pred_ranks[node] for node in common], dtype=float)
        y = np.array([true_ranks[node] for node in common], dtype=float)
        mean_x = x.mean()
        mean_y = y.mean()
        x_centered = x - mean_x
        y_centered = y - mean_y
        denom = math.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
        if denom == 0:
            spearman = 0.0
        else:
            spearman = np.sum(x_centered * y_centered) / denom
    return precision_at_k, spearman


def summarize_method(configs: list[tuple[int, int, int, int]], scores: dict[tuple[int, int, int, int], float]) -> dict[str, float]:
    true_scores = {cfg: true_difficulty(cfg) for cfg in configs}
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
        "precision_at_k": precision_at_k,
        "spearman": float(spearman),
        "gain_vs_random": float(gain_vs_random),
        "overlap": float(overlap),
        "cutoff": float(cutoff),
    }


def run_method_suite(
    configs: list[tuple[int, int, int, int]],
) -> dict[str, dict[str, float]]:
    true_scores = {cfg: true_difficulty(cfg) for cfg in configs}
    graph = make_graph(configs)

    plain_scores = pagerank(graph)
    weighted_scores = failure_weighted_pagerank(graph, true_scores)
    pressure_scores = tuning_pressure_scores(configs, true_scores)

    return {
        "plain_pagerank": summarize_method(configs, plain_scores),
        "failure_weighted_pagerank": summarize_method(configs, weighted_scores),
        "tuning_pressure": summarize_method(configs, pressure_scores),
    }


def run_experiment() -> dict[str, object]:
    configs = generate_larger_config_space(4, 5, 4, 4)
    summary = run_method_suite(configs)
    print("=" * 80)
    print("LARGER-SCALE DIFFICULTY FORECASTING COMPARISON")
    print("=" * 80)
    for method_name, metrics in summary.items():
        print(
            f"{method_name}: spearman={metrics['spearman']:.3f}, "
            f"precision@k={metrics['precision_at_k']:.3f}, "
            f"gain_vs_random={metrics['gain_vs_random']:.3f}, "
            f"overlap={int(metrics['overlap'])}/{int(metrics['cutoff'])}"
        )
    print("=" * 80)
    return {"config_count": len(configs), "methods": summary}


if __name__ == "__main__":
    run_experiment()
