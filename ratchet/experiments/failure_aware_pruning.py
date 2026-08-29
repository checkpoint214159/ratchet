"""
Failure-aware search space pruning experiment.

Idea: Instead of treating compile failures as atomic events, extract patterns from them
(e.g. "BLOCK_M > sequence_length fails") and use those to prune the search space before
the optimizer even tries infeasible configs.

This reduces wasted evaluations and improves effective sampling density.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import dual_annealing


@dataclass
class KernelConfig:
    """A point in the kernel configuration space."""

    BLOCK_M: int
    BLOCK_N: int
    num_warps: int
    num_stages: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.BLOCK_M, self.BLOCK_N, self.num_warps, self.num_stages)

    @staticmethod
    def from_tuple(t: tuple[int, int, int, int]) -> KernelConfig:
        return KernelConfig(*t)


@dataclass
class CompileFailure:
    """Record a config that failed to compile, with the reason."""

    config: KernelConfig
    reason: str
    device_shared_memory: int = 99 * 1024  # RTX 4060: 99 KB


class FailureAnalyzer:
    """Extract constraints from compile failures to prune the search space."""

    def __init__(self, head_dim: int = 64):
        self.head_dim = head_dim
        self.failures: list[CompileFailure] = []
        self.constraints: list[Callable[[KernelConfig], bool]] = []

    def record_failure(self, config: KernelConfig, reason: str) -> None:
        """Record a failure and extract constraints from it."""
        failure = CompileFailure(config, reason)
        self.failures.append(failure)

        # Heuristic constraint extraction:
        # If shared memory exceeded, record the formula that was violated
        if "shared_memory" in reason or "out of memory" in reason:
            # SMEM = BLOCK_M * d * 2  +  stages * (2 * BLOCK_N * d * 2)  bytes
            actual_smem = (
                config.BLOCK_M * self.head_dim * 2
                + config.num_stages * (2 * config.BLOCK_N * self.head_dim * 2)
            )
            if actual_smem > failure.device_shared_memory:
                # Create a constraint that excludes this region
                def smem_constraint(
                    cfg: KernelConfig,
                    max_smem: int = failure.device_shared_memory,
                ) -> bool:
                    required = (
                        cfg.BLOCK_M * self.head_dim * 2
                        + cfg.num_stages * (2 * cfg.BLOCK_N * self.head_dim * 2)
                    )
                    return required <= max_smem

                self.constraints.append(smem_constraint)

        # If register spill detected, penalize large warps with large tiles
        if "register" in reason:

            def register_constraint(cfg: KernelConfig) -> bool:
                # Rough heuristic: high register pressure happens when
                # (BLOCK_M + BLOCK_N) * num_warps is too large
                return (cfg.BLOCK_M + cfg.BLOCK_N) * cfg.num_warps < 1024

            self.constraints.append(register_constraint)

    def is_feasible(self, config: KernelConfig) -> bool:
        """Check if a config passes all learned constraints."""
        return all(constraint(config) for constraint in self.constraints)

    def get_valid_configs(
        self, all_configs: list[KernelConfig]
    ) -> list[KernelConfig]:
        """Filter configs to only feasible ones."""
        return [cfg for cfg in all_configs if self.is_feasible(cfg)]

    def summary(self) -> dict:
        """Return analysis summary."""
        return {
            "failure_count": len(self.failures),
            "constraint_count": len(self.constraints),
            "failure_reasons": [f.reason for f in self.failures],
        }


def generate_config_space(
    block_m_values: list[int] = None,
    block_n_values: list[int] = None,
    num_warps_values: list[int] = None,
    num_stages_values: list[int] = None,
) -> list[KernelConfig]:
    """Generate a configuration space."""
    if block_m_values is None:
        block_m_values = [64, 128, 256]
    if block_n_values is None:
        block_n_values = [32, 64, 128]
    if num_warps_values is None:
        num_warps_values = [4, 8, 16]
    if num_stages_values is None:
        num_stages_values = [2, 3, 4, 5]

    configs = []
    for bm in block_m_values:
        for bn in block_n_values:
            for nw in num_warps_values:
                for ns in num_stages_values:
                    configs.append(KernelConfig(bm, bn, nw, ns))
    return configs


def simulate_kernel_performance(
    config: KernelConfig,
    shape_n: int = 512,
    shape_d: int = 64,
    baseline_time_ns: float = 50000.0,
) -> tuple[str, float]:
    """
    Simulate kernel timing for a config.
    Returns: (status, time_ns)

    Status is one of: "ok", "compile_error"
    """
    # Simulate compile failures based on realistic constraints
    head_dim = 64

    # Constraint 1: Shared memory limit (RTX 4060 has 99 KB)
    smem_needed = (
        config.BLOCK_M * head_dim * 2
        + config.num_stages * (2 * config.BLOCK_N * head_dim * 2)
    )
    if smem_needed > 99 * 1024:
        return ("compile_error", float("inf"))

    # Constraint 2: BLOCK_M must be <= sequence length
    if config.BLOCK_M > shape_n:
        return ("compile_error", float("inf"))

    # Constraint 3: Register spill heuristic
    if (config.BLOCK_M + config.BLOCK_N) * config.num_warps > 1024:
        # ~40% chance to compile if this heuristic is violated
        if np.random.random() < 0.6:
            return ("compile_error", float("inf"))

    # Constraint 4: Too many stages with small tiles can cause issues
    if config.num_stages > 4 and config.BLOCK_N < 64:
        if np.random.random() < 0.3:
            return ("compile_error", float("inf"))

    # If we get here, the kernel compiles. Simulate realistic speedup.
    # Better configs (larger tiles, good stage count) are faster.
    speedup_factor = 1.0
    speedup_factor += (config.BLOCK_M / 128.0) * 0.2  # Larger tiles help
    speedup_factor += (config.BLOCK_N / 128.0) * 0.15
    speedup_factor -= (config.num_stages - 3) * 0.05  # Too many stages hurt
    speedup_factor += (config.num_warps / 8.0) * 0.1  # More warps can help

    # Add noise
    speedup_factor += np.random.normal(0, 0.08)
    speedup_factor = max(0.7, speedup_factor)

    simulated_time = baseline_time_ns / speedup_factor
    return ("ok", simulated_time)


@dataclass
class SearchResult:
    """Result of a tuning run."""

    method: str  # "baseline" or "pruned"
    budget: int
    configs_evaluated: int
    configs_compiled: int
    compile_failure_rate: float
    best_speedup: float
    best_config: KernelConfig
    total_time_ns: float
    evaluations_wasted_on_failures: int


def run_search(
    all_configs: list[KernelConfig],
    budget: int = 50,
    pruner: FailureAnalyzer | None = None,
    seed: int = 42,
) -> SearchResult:
    """
    Run dual annealing search with optional failure-aware pruning.

    Args:
        all_configs: All possible configurations
        budget: Max number of evaluations (dual annealing limit)
        pruner: Optional FailureAnalyzer to filter space
        seed: Random seed

    Returns:
        SearchResult with metrics
    """
    np.random.seed(seed)

    # Filter configs if pruner provided
    if pruner is not None:
        active_configs = pruner.get_valid_configs(all_configs)
    else:
        active_configs = all_configs

    if not active_configs:
        raise ValueError("No valid configs after pruning")

    # Track evaluations
    evaluations_count = 0
    compile_failures = 0
    best_successful_time = float("inf")
    best_successful_config = None

    def fitness(x_normalized):
        nonlocal evaluations_count, compile_failures, best_successful_time, best_successful_config

        # Map normalized (0, 1) coords to discrete configs
        block_m_vals = sorted(set(cfg.BLOCK_M for cfg in active_configs))
        block_n_vals = sorted(set(cfg.BLOCK_N for cfg in active_configs))
        num_w_vals = sorted(set(cfg.num_warps for cfg in active_configs))
        num_s_vals = sorted(set(cfg.num_stages for cfg in active_configs))

        # Quantize to nearest valid config
        idx_m = int(x_normalized[0] * (len(block_m_vals) - 1))
        idx_n = int(x_normalized[1] * (len(block_n_vals) - 1))
        idx_w = int(x_normalized[2] * (len(num_w_vals) - 1))
        idx_s = int(x_normalized[3] * (len(num_s_vals) - 1))

        cfg = KernelConfig(
            block_m_vals[idx_m],
            block_n_vals[idx_n],
            num_w_vals[idx_w],
            num_s_vals[idx_s],
        )

        evaluations_count += 1
        status, time_ns = simulate_kernel_performance(cfg)

        if status == "compile_error":
            compile_failures += 1
            return 1e10  # Large finite penalty for failures

        # Track best successful config
        if time_ns < best_successful_time:
            best_successful_time = time_ns
            best_successful_config = cfg

        return time_ns

    # Run dual annealing
    bounds = [(0, 1) for _ in range(4)]
    result = dual_annealing(
        fitness, bounds, seed=seed, maxfun=budget, accept=-10
    )

    # If no successful config was found, report best attempted
    if best_successful_config is None:
        best_config = active_configs[0]
        best_time_ns = float("inf")
        best_speedup = 0.0
    else:
        best_config = best_successful_config
        best_time_ns = best_successful_time
        baseline_time = 50000.0
        best_speedup = baseline_time / best_time_ns

    return SearchResult(
        method="pruned" if pruner else "baseline",
        budget=budget,
        configs_evaluated=evaluations_count,
        configs_compiled=evaluations_count - compile_failures,
        compile_failure_rate=compile_failures / max(1, evaluations_count),
        best_speedup=best_speedup,
        best_config=best_config,
        total_time_ns=best_time_ns,
        evaluations_wasted_on_failures=compile_failures,
    )


def run_experiment(output_dir: Path = None) -> dict:
    """
    Run the full failure-aware pruning experiment.

    Compares:
    1. Baseline: dual annealing on full space
    2. Pruned: dual annealing on space filtered by learned constraints
    """
    if output_dir is None:
        output_dir = Path("research/experiments/failure_aware_pruning")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FAILURE-AWARE SEARCH SPACE PRUNING EXPERIMENT")
    print("=" * 70)

    # Generate full config space
    all_configs = generate_config_space()
    print(f"\nGenerated {len(all_configs)} total configurations")

    # Run baseline (no pruning)
    print("\n" + "-" * 70)
    print("BASELINE: Dual annealing without pruning (200 evals)")
    print("-" * 70)
    baseline_result = run_search(all_configs, budget=200, pruner=None, seed=42)
    print(f"  Evaluations: {baseline_result.configs_evaluated}")
    print(f"  Compiled successfully: {baseline_result.configs_compiled}")
    print(f"  Compile failure rate: {baseline_result.compile_failure_rate:.1%}")
    print(f"  Evaluations wasted on failures: {baseline_result.evaluations_wasted_on_failures}")
    print(f"  Best speedup found: {baseline_result.best_speedup:.2f}x")
    print(f"  Best config: {baseline_result.best_config}")

    # Simulate learning from failures in baseline run
    # (In real scenario, these would be actual compile errors)
    print("\n" + "-" * 70)
    print("Learning from baseline failures...")
    print("-" * 70)

    pruner = FailureAnalyzer(head_dim=64)

    # Simulate some realistic failures
    test_failures = [
        (KernelConfig(256, 128, 16, 5), "shared_memory out of memory"),
        (KernelConfig(512, 128, 16, 4), "shared_memory out of memory"),
        (KernelConfig(128, 128, 16, 5), "register spill on shared memory"),
        (KernelConfig(256, 64, 16, 5), "shared_memory out of memory"),
    ]

    for cfg, reason in test_failures:
        pruner.record_failure(cfg, reason)

    summary = pruner.summary()
    print(f"  Recorded {summary['failure_count']} failure patterns")
    print(f"  Extracted {summary['constraint_count']} constraints")
    print(f"  Failure reasons: {set(summary['failure_reasons'])}")

    # Run pruned search
    print("\n" + "-" * 70)
    print("PRUNED: Dual annealing with failure-aware constraints (200 evals)")
    print("-" * 70)
    pruned_result = run_search(all_configs, budget=200, pruner=pruner, seed=42)
    print(f"  Evaluations: {pruned_result.configs_evaluated}")
    print(f"  Compiled successfully: {pruned_result.configs_compiled}")
    print(f"  Compile failure rate: {pruned_result.compile_failure_rate:.1%}")
    print(f"  Evaluations wasted on failures: {pruned_result.evaluations_wasted_on_failures}")
    print(f"  Best speedup found: {pruned_result.best_speedup:.2f}x")
    print(f"  Best config: {pruned_result.best_config}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    failure_reduction = 1.0 - (
        pruned_result.evaluations_wasted_on_failures
        / max(1, baseline_result.evaluations_wasted_on_failures)
    )
    speedup_change = pruned_result.best_speedup - baseline_result.best_speedup

    print(
        f"  Failure reduction: {failure_reduction:.1%} "
        f"({baseline_result.evaluations_wasted_on_failures} → {pruned_result.evaluations_wasted_on_failures})"
    )
    print(f"  Speedup quality change: {speedup_change:+.2f}x")
    print(
        f"  Efficiency gain (speedup per eval): "
        f"baseline={baseline_result.best_speedup / baseline_result.configs_evaluated:.4f}, "
        f"pruned={pruned_result.best_speedup / pruned_result.configs_evaluated:.4f}"
    )

    # Save results
    results = {
        "experiment": "failure_aware_search_space_pruning",
        "timestamp": str(Path(__file__).stat().st_mtime),
        "baseline": {
            "method": baseline_result.method,
            "budget": baseline_result.budget,
            "configs_evaluated": baseline_result.configs_evaluated,
            "configs_compiled": baseline_result.configs_compiled,
            "compile_failure_rate": baseline_result.compile_failure_rate,
            "best_speedup": baseline_result.best_speedup,
            "best_config": {
                "BLOCK_M": baseline_result.best_config.BLOCK_M,
                "BLOCK_N": baseline_result.best_config.BLOCK_N,
                "num_warps": baseline_result.best_config.num_warps,
                "num_stages": baseline_result.best_config.num_stages,
            },
            "evaluations_wasted": baseline_result.evaluations_wasted_on_failures,
        },
        "pruned": {
            "method": pruned_result.method,
            "budget": pruned_result.budget,
            "configs_evaluated": pruned_result.configs_evaluated,
            "configs_compiled": pruned_result.configs_compiled,
            "compile_failure_rate": pruned_result.compile_failure_rate,
            "best_speedup": pruned_result.best_speedup,
            "best_config": {
                "BLOCK_M": pruned_result.best_config.BLOCK_M,
                "BLOCK_N": pruned_result.best_config.BLOCK_N,
                "num_warps": pruned_result.best_config.num_warps,
                "num_stages": pruned_result.best_config.num_stages,
            },
            "evaluations_wasted": pruned_result.evaluations_wasted_on_failures,
        },
        "metrics": {
            "failure_reduction_pct": failure_reduction * 100,
            "speedup_quality_change": speedup_change,
            "efficiency_baseline": baseline_result.best_speedup
            / baseline_result.configs_evaluated,
            "efficiency_pruned": pruned_result.best_speedup / pruned_result.configs_evaluated,
        },
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

    return results


if __name__ == "__main__":
    run_experiment()
