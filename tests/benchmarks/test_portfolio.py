"""CPU-only contracts for definition-only future baseline protocols."""

from __future__ import annotations

import ast
import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

from ratchet.benchmarks import (
    REFERENCE_BENCHMARK_PATH,
    REFERENCE_BENCHMARK_SHA256,
    BaselineAvailability,
    BaselineBackend,
    BaselineKind,
    BaselineValidation,
    BaselineVendor,
    DispatchVerificationState,
    baseline_from_configuration,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGURATIONS = ROOT / "benchmarks" / "runners" / "configurations"
NAMES = ("eager.json", "compiled.json", "sdpa.json", "vendor_library.json")


def _configuration(name: str) -> dict[str, object]:
    return json.loads((CONFIGURATIONS / name).read_text())


def _definitions():
    return tuple(baseline_from_configuration(_configuration(name)) for name in NAMES)


def test_exact_portfolio_preserves_custody_full_workload_and_future_no_run_state():
    definitions = _definitions()

    assert {definition.kind for definition in definitions} == set(BaselineKind)
    assert {definition.baseline_id for definition in definitions} == {
        "ratchet.intel.xpu.eager.v1",
        "ratchet.intel.xpu.compiled.v1",
        "ratchet.intel.xpu.sdpa.v1",
        "ratchet.intel.xpu.vendor_library.v1",
    }
    assert all(definition.vendor is BaselineVendor.INTEL for definition in definitions)
    assert all(
        definition.target_backend is BaselineBackend.XPU for definition in definitions
    )
    assert all(
        definition.availability is BaselineAvailability.UNAVAILABLE
        and definition.validation is BaselineValidation.UNVALIDATED
        and definition.definition_only
        and definition.workload == "reference_transformer"
        for definition in definitions
    )
    assert all(
        definition.custody.relative_path == REFERENCE_BENCHMARK_PATH
        and definition.custody.sha256 == REFERENCE_BENCHMARK_SHA256
        for definition in definitions
    )
    assert sha256((ROOT / REFERENCE_BENCHMARK_PATH).read_bytes()).hexdigest() == (
        REFERENCE_BENCHMARK_SHA256
    )


def test_protocols_separate_compilation_first_run_and_steady_state():
    definitions = _definitions()

    assert [definition.compilation.policy for definition in definitions] == [
        "eager",
        "torch.compile",
        "eager",
        "vendor-library",
    ]
    compiled = definitions[1].compilation
    assert (
        compiled.enabled,
        compiled.backend,
        compiled.mode,
        compiled.fullgraph,
        compiled.dynamic,
    ) == (True, "inductor", "default", False, False)
    for definition in definitions:
        assert definition.timing.compilation_recorded_separately
        assert definition.timing.first_run_recorded_separately
        assert definition.timing.steady_state_recorded_separately
        assert (
            definition.timing.synchronization
            == "backend synchronize before and after measured region"
        )
        assert (
            definition.timing.timing_method
            == "backend device events with host-timer cross-check"
        )
        assert definition.timing.warmup_completed_calls == 20
        assert definition.timing.ordering_blocks == 10
        assert definition.timing.completed_calls_per_model_per_block == 30
        assert definition.timing.ordering_patterns == ("ABBA", "BAAB")


def test_attention_substitutions_are_constrained_to_the_authoritative_seam():
    eager, compiled, sdpa, vendor_library = _definitions()

    assert eager.attention_substitution is None
    assert compiled.attention_substitution is None
    assert sdpa.attention_substitution is not None
    assert vendor_library.attention_substitution is not None
    for definition in (sdpa, vendor_library):
        substitution = definition.attention_substitution
        assert substitution is not None
        assert substitution.scope == "attention_core_only"
        assert substitution.weight_copy_compatible
        assert substitution.valid_mask_semantics == "preserved"
        assert substitution.causal_semantics == "preserved"
        assert substitution.output_contract == "unchanged"
    assert (
        sdpa.attention_substitution.primitive
        == "torch.nn.functional.scaled_dot_product_attention"
    )
    assert vendor_library.attention_substitution.primitive == "oneDNN Graph SDPA"
    assert (
        vendor_library.attention_substitution.dispatch_verification.state
        is DispatchVerificationState.REQUIRED_UNVERIFIED
    )
    assert (
        vendor_library.attention_substitution.dispatch_verification.method
        == "oneDNN Graph partition inspection"
    )


@pytest.mark.parametrize(
    ("backend", "vendor", "baseline_id"),
    [
        (
            BaselineBackend.CUDA.value,
            BaselineVendor.NVIDIA.value,
            "ratchet.nvidia.cuda.eager.v1",
        ),
        (
            BaselineBackend.HIP.value,
            BaselineVendor.AMD.value,
            "ratchet.amd.hip.eager.v1",
        ),
    ],
)
def test_schema_requires_a_correctly_renamed_vendor_and_backend_namespace(
    backend: str, vendor: str, baseline_id: str
):
    configuration = copy.deepcopy(_configuration("eager.json"))
    configuration["target_backend"] = backend
    configuration["vendor"] = vendor
    configuration["baseline_id"] = baseline_id

    definition = baseline_from_configuration(configuration)

    assert definition.baseline_id == baseline_id
    assert definition.target_backend.value == backend
    assert definition.vendor.value == vendor


def test_schema_rejects_relabeling_an_intel_baseline_id_for_cuda():
    configuration = _configuration("eager.json")
    configuration["target_backend"] = BaselineBackend.CUDA.value
    configuration["vendor"] = BaselineVendor.NVIDIA.value

    with pytest.raises(ValueError, match="namespace"):
        baseline_from_configuration(configuration)


@pytest.mark.parametrize(
    ("name", "path", "value", "message"),
    [
        (
            "eager.json",
            ("timing", "first_run_recorded_separately"),
            False,
            "first run",
        ),
        (
            "eager.json",
            ("reference_benchmark", "sha256"),
            "0" * 64,
            "authoritative",
        ),
        ("eager.json", ("definition_only",), False, "cannot enable execution"),
        (
            "sdpa.json",
            ("attention_substitution", "valid_mask_semantics"),
            "changed",
            "valid-mask",
        ),
        (
            "vendor_library.json",
            ("attention_substitution", "dispatch_verification", "method"),
            "arbitrary dispatch check",
            "attention substitution",
        ),
        (
            "vendor_library.json",
            (
                "attention_substitution",
                "dispatch_verification",
                "expected_provider",
            ),
            "arbitrary provider",
            "attention substitution",
        ),
    ],
)
def test_configuration_rejects_unprovenanced_or_semantically_changed_variants(
    name: str, path: tuple[str, ...], value: object, message: str
):
    configuration = _configuration(name)
    target: dict[str, object] = configuration
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        baseline_from_configuration(configuration)


def test_public_baseline_package_has_no_framework_import_or_execution_entry_point():
    source = (ROOT / "ratchet" / "benchmarks" / "__init__.py").read_text()
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]

    assert not any(name == "torch" or name.startswith("torch.") for name in imports)
    assert not {"run", "execute", "measure", "benchmark"} & {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
