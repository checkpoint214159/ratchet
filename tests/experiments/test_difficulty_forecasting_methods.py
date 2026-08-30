from __future__ import annotations

from ratchet.experiments.pagerank_difficulty_forecasting import (
    generate_larger_config_space,
    run_method_suite,
)


def test_method_suite_ranks_weighted_graph_methods_above_plain_pagerank():
    configs = generate_larger_config_space(4, 5, 4, 4)
    summary = run_method_suite(configs)

    assert set(summary) >= {"plain_pagerank", "failure_weighted_pagerank", "tuning_pressure"}
    assert summary["failure_weighted_pagerank"]["spearman"] > summary["plain_pagerank"]["spearman"]
    assert summary["tuning_pressure"]["spearman"] > summary["plain_pagerank"]["spearman"]
    assert summary["failure_weighted_pagerank"]["spearman"] > 0.0


def test_standardized_benchmark_supports_transfer_and_fp8_scenarios():
    from ratchet.experiments.standardized_method_benchmark import benchmark_scenarios

    result = benchmark_scenarios(
        hardware_profiles=["cuda", "hip"],
        quantization_modes=["fp32", "fp8"],
        seeds=2,
        max_runtime_seconds=1,
        block_m_count=3,
        block_n_count=3,
        warp_count=3,
        stage_count=3,
    )

    scenario_names = {entry["scenario_name"] for entry in result["scenarios"]}
    assert "cuda/fp32" in scenario_names
    assert "cuda/fp8" in scenario_names
    assert "hip/fp8" in scenario_names
    assert all(entry["summary"]["failure_weighted_pagerank"]["spearman"] >= -1.0 for entry in result["scenarios"])
