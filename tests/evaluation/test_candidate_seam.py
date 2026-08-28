"""Source-only characterization of the protected future candidate seam."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from ratchet.evaluation import (
    AUTHORITATIVE_CANDIDATE_INTEGRATION_CONTRACT,
    REFERENCE_BENCHMARK_PATH,
    REFERENCE_BENCHMARK_SHA256,
    CandidateIntegrationContract,
    CandidateSeamContract,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / REFERENCE_BENCHMARK_PATH
EVALUATION = ROOT / "ratchet" / "evaluation" / "__init__.py"


def _source() -> str:
    return REFERENCE.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(REFERENCE))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_protected_reference_hash_and_public_contract_are_exact():
    contract = AUTHORITATIVE_CANDIDATE_INTEGRATION_CONTRACT

    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == (
        REFERENCE_BENCHMARK_SHA256
    )
    assert contract.seam.evaluator_path == REFERENCE_BENCHMARK_PATH
    assert contract.seam.candidate_class == "UserOptimizedTransformer"
    assert contract.seam.observed_base_class == "BaselineTransformer"
    assert contract.seam.forward_parameters == ("x", "valid_token_mask")
    assert (
        contract.weight_copy.parameter_name_mismatch_handling
        == "customize copy_model_weights"
    )
    assert contract.implementation_state == "structural_only"


def test_ast_observes_current_candidate_base_and_characterizes_weight_copy_path():
    tree = _tree()
    optimized = _class(tree, "UserOptimizedTransformer")
    forward = next(
        node
        for node in optimized.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    weight_copy = _function(tree, "copy_model_weights")

    assert [ast.unparse(base) for base in optimized.bases] == ["BaselineTransformer"]
    assert [argument.arg for argument in forward.args.args] == [
        "self",
        "x",
        "valid_token_mask",
    ]
    assert "return super().forward(x, valid_token_mask)" in ast.unparse(forward)
    assert [argument.arg for argument in weight_copy.args.args] == [
        "baseline",
        "optimized",
        "strict",
    ]
    assert ast.literal_eval(weight_copy.args.defaults[0]) is True
    weight_copy_source = ast.unparse(weight_copy)
    assert "copy.deepcopy(baseline.state_dict())" in weight_copy_source
    assert "optimized.load_state_dict(state_dict, strict=strict)" in weight_copy_source


def test_candidate_base_is_observed_not_a_required_integration_obligation():
    seam = AUTHORITATIVE_CANDIDATE_INTEGRATION_CONTRACT.seam

    future_observation = CandidateSeamContract(
        evaluator_path=seam.evaluator_path,
        evaluator_sha256=seam.evaluator_sha256,
        candidate_class=seam.candidate_class,
        observed_base_class="FutureCompatibleTransformer",
        forward_parameters=seam.forward_parameters,
    )

    assert future_observation.observed_base_class == "FutureCompatibleTransformer"


def test_ast_characterizes_valid_mask_causal_and_output_contracts():
    tree = _tree()
    attention_forward = next(
        node
        for node in _class(tree, "BaselineSelfAttention").body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    transformer_forward = next(
        node
        for node in _class(tree, "BaselineTransformer").body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    attention_source = ast.unparse(attention_forward)
    transformer_source = ast.unparse(transformer_forward)
    contract = AUTHORITATIVE_CANDIDATE_INTEGRATION_CONTRACT.mask_and_output

    assert "if causal:" in attention_source
    assert "triu(diagonal=1)" in attention_source
    assert "invalid_keys = ~valid_token_mask[:, None, None, :]" in attention_source
    assert (
        "scores = scores.masked_fill(invalid_keys, float('-inf'))" in attention_source
    )
    assert (
        "output = output.masked_fill(~valid_token_mask[..., None], 0)"
        in attention_source
    )
    assert "x = x.masked_fill(~valid_token_mask[..., None], 0)" in transformer_source
    assert (
        contract.valid_mask_contract
        == "mask invalid keys and zero invalid token outputs"
    )
    assert contract.causal_contract == "causal=True masks future key positions"
    assert contract.output_contract == "tensor [batch_size, seq_len, d_model]"


def test_contract_rejects_an_executable_or_drifted_integration_state():
    contract = AUTHORITATIVE_CANDIDATE_INTEGRATION_CONTRACT

    with pytest.raises(ValueError, match="structural characterization"):
        CandidateIntegrationContract(
            seam=contract.seam,
            weight_copy=contract.weight_copy,
            mask_and_output=contract.mask_and_output,
            implementation_state="executable",
        )


def test_evaluation_contracts_have_no_framework_import_or_candidate_execution_api():
    tree = ast.parse(EVALUATION.read_text(encoding="utf-8"), filename=str(EVALUATION))
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }

    assert not any(name == "torch" or name.startswith("torch.") for name in imports)
    assert (
        not {"build_candidate", "execute_candidate", "run_candidate"} & function_names
    )
