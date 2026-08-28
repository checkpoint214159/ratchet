"""CPU-only custody tests for the supplied authoritative evaluator."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

EXPECTED_SHA256 = "5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e"
REFERENCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "reference"
    / "torch_transformer_benchmark.py"
)
RATCHET_PATH = Path(__file__).resolve().parents[2] / "ratchet"


def _source() -> str:
    return REFERENCE_PATH.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source(), filename=str(REFERENCE_PATH))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _option_default(parse_args: ast.FunctionDef, option: str) -> object:
    matches = []
    for node in ast.walk(parse_args):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != option:
            continue
        default = next(
            keyword.value for keyword in node.keywords if keyword.arg == "default"
        )
        matches.append(ast.literal_eval(default))
    assert len(matches) == 1
    return matches[0]


def test_reference_benchmark_is_byte_preserved():
    assert hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_executable_tolerance_defaults_and_or_rule_are_preserved():
    tree = _tree()

    parse_args = _function(tree, "parse_args")
    assert _option_default(parse_args, "--atol") == 0.002
    assert _option_default(parse_args, "--rtol") == 0.02

    compare_outputs = _function(tree, "compare_outputs")
    passed_mask = next(
        node
        for node in ast.walk(compare_outputs)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "passed_mask"
            for target in node.targets
        )
    )
    assert isinstance(passed_mask.value, ast.BinOp)
    assert isinstance(passed_mask.value.op, ast.BitAnd)
    assert isinstance(passed_mask.value.right, ast.BinOp)
    assert isinstance(passed_mask.value.right.op, ast.BitOr)
    assert ast.unparse(passed_mask.value.right) == "abs_ok | rel_ok"


def test_user_optimized_transformer_remains_the_designated_seam():
    tree = _tree()
    baseline = _class(tree, "BaselineTransformer")
    optimized = _class(tree, "UserOptimizedTransformer")

    assert baseline.name == "BaselineTransformer"
    assert [ast.unparse(base) for base in optimized.bases] == ["BaselineTransformer"]
    forward = next(
        node
        for node in optimized.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    assert [argument.arg for argument in forward.args.args] == [
        "self",
        "x",
        "valid_token_mask",
    ]
    assert "super().forward(x, valid_token_mask)" in ast.unparse(forward)


def test_non_cuda_timing_is_explicitly_not_accelerator_synchronized():
    source = _source()
    benchmark_once = _function(_tree(), "benchmark_once")

    assert "torch.cuda.Event(enable_timing=True)" in ast.unparse(benchmark_once)
    assert "time.perf_counter_ns()" in ast.unparse(benchmark_once)
    assert "torch.xpu.Event" not in source
    assert "torch.xpu.synchronize" not in source
    assert "torch.cuda.synchronize(device)" in source


def test_ratchet_code_does_not_import_reference_benchmark_custody():
    violations = []
    for path in RATCHET_PATH.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    f"{path}:{alias.name}"
                    for alias in node.names
                    if alias.name == "benchmarks.reference"
                    or alias.name.startswith("benchmarks.reference.")
                )
            if isinstance(node, ast.ImportFrom) and node.module == "benchmarks":
                violations.extend(
                    f"{path}:from benchmarks import {alias.name}"
                    for alias in node.names
                    if alias.name == "reference"
                )
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (
                    node.module == "benchmarks.reference"
                    or node.module.startswith("benchmarks.reference.")
                )
            ):
                violations.append(f"{path}:from {node.module} import ...")

    assert violations == []
