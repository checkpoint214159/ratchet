"""Integrity checks for the definition-only first Intel future protocol."""

from __future__ import annotations

import ast
import json
from hashlib import sha256
from pathlib import Path

from ratchet.benchmarks import baseline_from_configuration

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "research" / "protocols" / "PROTO-INTEL-0001.json"
BIBLIOGRAPHY = ROOT / "research" / "paper" / "bibliography.bib"
EVALUATOR = ROOT / "benchmarks" / "reference" / "torch_transformer_benchmark.py"

TOP_LEVEL_FIELDS = {
    "schema_version",
    "protocol_id",
    "hypothesis_id",
    "source_idea_id",
    "source_survey_id",
    "scope",
    "execution_status",
    "execution_permitted",
    "target",
    "hypothesis",
    "motivation",
    "literature_keys",
    "evaluator_contract",
    "arms",
    "evaluation_cases",
    "conditional_dtype_expansion",
    "correctness_protocol",
    "timing_protocol",
    "stop_criteria",
    "acceptance_criteria",
}
FORBIDDEN_FIELDS = {
    "event_id",
    "experiment_id",
    "environment_id",
    "environment_artifact_digest",
    "candidate",
    "candidate_state",
    "correctness_result",
    "timing_result",
    "memory_result",
    "baseline_comparison",
    "current_best_comparison",
    "artifact_digests",
    "decision",
    "decision_reason",
    "paper_inclusion",
}
FORBIDDEN_FIELD_FRAGMENTS = (
    "event",
    "candidate",
    "result",
    "comparison",
    "artifact",
    "decision",
    "environment",
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text())


def _keys_recursively(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(
            *(_keys_recursively(item) for item in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_keys_recursively(item) for item in value))
    return set()


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_protocol_has_the_exact_definition_only_schema_and_no_execution_permission():
    protocol = _protocol()

    assert set(protocol) == TOP_LEVEL_FIELDS
    assert {
        "schema_version": 1,
        "protocol_id": "PROTO-INTEL-0001",
        "hypothesis_id": "HYP-0001",
        "source_idea_id": "IDEA-0001",
        "source_survey_id": "LIT-SURVEY-0001",
        "scope": "definition_only_future_protocol",
        "execution_status": "not_run_hardware_unavailable",
        "execution_permitted": False,
    }.items() <= protocol.items()
    assert set(protocol["target"]) == {
        "vendor",
        "backend",
        "device_family",
        "qualification_gate",
    }
    assert "ratchet.intel.xpu.compiled.v1" in protocol["hypothesis"]
    assert "ratchet.intel.xpu.eager.v1" in protocol["hypothesis"]


def test_protocol_custody_and_tolerance_match_the_protected_evaluator_or_rule():
    protocol = _protocol()
    custody = protocol["evaluator_contract"]
    correctness = protocol["correctness_protocol"]
    assert isinstance(custody, dict)
    assert isinstance(correctness, dict)

    assert custody == {
        "relative_path": "benchmarks/reference/torch_transformer_benchmark.py",
        "sha256": sha256(EVALUATOR.read_bytes()).hexdigest(),
    }
    assert correctness == {
        "shared_weights": "strict",
        "identical_inputs": True,
        "trial_count": 5,
        "seed_offsets": [0, 1, 2, 3, 4],
        "executable_rule": "abs_error <= atol OR abs_error <= rtol * abs(reference)",
        "atol": 0.002,
        "rtol": 0.02,
    }

    comparison = _function(ast.parse(EVALUATOR.read_text()), "compare_outputs")
    assert any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitOr)
        and isinstance(node.left, ast.Name)
        and isinstance(node.right, ast.Name)
        and {node.left.id, node.right.id} == {"abs_ok", "rel_ok"}
        for node in ast.walk(comparison)
    )


def test_protocol_has_the_exact_float32_case_matrix_and_conditional_dtypes():
    protocol = _protocol()
    cases = protocol["evaluation_cases"]
    expansion = protocol["conditional_dtype_expansion"]
    assert isinstance(cases, list)
    assert isinstance(expansion, dict)

    assert cases == [
        {
            "case_id": "default",
            "batch_size": 8,
            "sequence_length": 128,
            "model_width": 512,
            "head_count": 8,
            "feed_forward_width": 2048,
            "layer_count": 6,
            "causal": False,
            "padding_ratio": 0.0,
            "dtype": "float32",
            "seed": 1234,
        },
        {
            "case_id": "causal",
            "batch_size": 2,
            "sequence_length": 257,
            "model_width": 512,
            "head_count": 8,
            "feed_forward_width": 2048,
            "layer_count": 2,
            "causal": True,
            "padding_ratio": 0.0,
            "dtype": "float32",
            "seed": 2234,
        },
        {
            "case_id": "padded",
            "batch_size": 4,
            "sequence_length": 127,
            "model_width": 256,
            "head_count": 8,
            "feed_forward_width": 1024,
            "layer_count": 2,
            "causal": False,
            "padding_ratio": 0.25,
            "dtype": "float32",
            "seed": 3234,
        },
        {
            "case_id": "long",
            "batch_size": 1,
            "sequence_length": 512,
            "model_width": 512,
            "head_count": 8,
            "feed_forward_width": 2048,
            "layer_count": 2,
            "causal": False,
            "padding_ratio": 0.0,
            "dtype": "float32",
            "seed": 4234,
        },
    ]
    assert expansion == {
        "condition": "qualified_backend_probe_supports_dtype",
        "dtypes": ["bfloat16", "float16"],
        "case_ids": ["default", "long"],
        "future_only": True,
    }


def test_protocol_arms_resolve_existing_unavailable_definition_only_baselines():
    protocol = _protocol()
    arms = protocol["arms"]
    assert isinstance(arms, list)
    assert len(arms) == 2
    assert all(isinstance(arm, dict) for arm in arms)

    definitions = []
    for arm in arms:
        assert set(arm) in (
            {"baseline_id", "configuration_path", "kind", "definition_only"},
            {
                "baseline_id",
                "configuration_path",
                "kind",
                "definition_only",
                "compilation",
            },
        )
        configuration = ROOT / arm["configuration_path"]
        definition = baseline_from_configuration(json.loads(configuration.read_text()))
        assert definition.baseline_id == arm["baseline_id"]
        assert definition.availability.value == "unavailable"
        assert definition.validation.value == "unvalidated"
        assert definition.definition_only is True
        definitions.append(definition)

    eager, compiled = definitions
    assert (eager.baseline_id, eager.kind.value) == (
        "ratchet.intel.xpu.eager.v1",
        "eager",
    )
    assert (compiled.baseline_id, compiled.kind.value) == (
        "ratchet.intel.xpu.compiled.v1",
        "compiled",
    )
    assert protocol["arms"][1]["compilation"] == {
        "backend": "inductor",
        "mode": "default",
        "fullgraph": False,
        "dynamic": False,
    }


def test_protocol_timing_equals_the_baseline_and_retains_required_measurement_method():
    protocol = _protocol()
    timing = protocol["timing_protocol"]
    assert isinstance(timing, dict)
    baseline_timing = timing["baseline_protocol"]
    assert isinstance(baseline_timing, dict)

    for name in ("eager.json", "compiled.json"):
        configuration = json.loads(
            (ROOT / "benchmarks" / "runners" / "configurations" / name).read_text()
        )
        assert baseline_timing == configuration["timing"]
    assert timing["timed_input_seed_offset"] == 100000
    assert timing["input_creation_excluded"] is True
    assert timing["retained_measurement_data"] == [
        "raw_samples",
        "median",
        "mean",
        "p90",
        "minimum",
        "standard_error",
        "paired_bootstrap_95_percent_interval",
        "peak_allocated_memory",
        "peak_reserved_memory",
        "clock_power_status",
        "method",
    ]


def test_protocol_citations_stops_and_acceptance_are_exact_and_ordered():
    protocol = _protocol()
    bibliography = BIBLIOGRAPHY.read_text()
    stops = protocol["stop_criteria"]
    acceptance = protocol["acceptance_criteria"]
    assert isinstance(stops, list)
    assert isinstance(acceptance, dict)

    assert protocol["literature_keys"] == [
        "ansel2024pytorch",
        "pytorch_xpu_2026",
        "schoonhoven2022autotuning",
    ]
    assert all(f"{{{key}," in bibliography for key in protocol["literature_keys"])
    assert [item["order"] for item in stops] == list(range(1, 8))
    assert stops[0] == {
        "order": 1,
        "trigger": "current_xpu_runtime_unavailable",
        "action": "stop before construction, compilation, correctness, profiling, timing, or kernel work",
    }
    assert [item["trigger"] for item in stops[1:]] == [
        "authoritative_evaluator_hash_or_contract_mismatch",
        "backend_not_qualified",
        "synchronization_or_device_events_missing",
        "compilation_failure",
        "timeout_or_crash",
        "any_correctness_failure",
    ]
    assert acceptance == {
        "all_correctness_cases_pass": True,
        "latency_intervals_disjoint": True,
        "paired_speedup_95_percent_lower_bound": {"operator": ">", "value": 1.02},
        "unexplained_peak_memory_increase_ratio": {"operator": "<=", "value": 0.05},
        "failure_outcome": "reject_or_inconclusive_never_promote",
    }


def test_protocol_recursively_prohibits_event_result_and_execution_surfaces():
    protocol = _protocol()
    keys = _keys_recursively(protocol)

    assert not FORBIDDEN_FIELDS & keys
    assert not {
        key
        for key in keys
        if any(fragment in key for fragment in FORBIDDEN_FIELD_FRAGMENTS)
    }
    assert not list((ROOT / "research" / "protocols").glob("*.py"))
