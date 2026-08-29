"""Structural and numerical checks for the test-only adversarial scalar pool."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import math
from pathlib import Path

import pytest

from tests.fixtures.adversarial_pool import (
    ATOL,
    NUMERICAL_NEAR_MISSES,
    RTOL,
    NumericalNearMiss,
    passes_authoritative_or,
)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "benchmarks" / "reference" / "torch_transformer_benchmark.py"
FIXTURE = ROOT / "tests" / "fixtures" / "adversarial_pool.py"
EXPECTED_SHA256 = "5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _option_default(function: ast.FunctionDef, option: str) -> object:
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == option
    ]
    assert len(calls) == 1
    return ast.literal_eval(
        next(item.value for item in calls[0].keywords if item.arg == "default")
    )


def test_pool_is_exact_finite_frozen_and_test_only():
    assert (ATOL, RTOL) == (0.002, 0.02)
    assert isinstance(NUMERICAL_NEAR_MISSES, tuple)
    assert dataclasses.is_dataclass(NumericalNearMiss)
    assert NumericalNearMiss.__dataclass_params__.frozen is True
    assert hasattr(NumericalNearMiss, "__slots__")
    assert [
        (
            item.case_id,
            item.boundary,
            item.reference,
            item.observed,
            item.expected_pass,
            item.rationale,
            item.scope,
        )
        for item in NUMERICAL_NEAR_MISSES
    ] == [
        (
            "SYN-NUM-001",
            "absolute_threshold_exactly",
            0.0,
            0.002,
            True,
            "absolute error equals atol at a zero reference",
            "test_only",
        ),
        (
            "SYN-NUM-002",
            "above_absolute_zero_relative",
            0.0,
            0.0020001,
            False,
            "absolute error exceeds atol and zero reference has no relative allowance",
            "test_only",
        ),
        (
            "SYN-NUM-003",
            "positive_relative_exactly",
            100.0,
            102.0,
            True,
            "relative error equals rtol times a positive reference",
            "test_only",
        ),
        (
            "SYN-NUM-004",
            "above_positive_relative",
            100.0,
            102.0001,
            False,
            "relative error exceeds rtol times a positive reference",
            "test_only",
        ),
        (
            "SYN-NUM-005",
            "negative_reference_abs_relative",
            -100.0,
            -102.0,
            True,
            "relative error equals rtol times a negative reference magnitude",
            "test_only",
        ),
        (
            "SYN-NUM-006",
            "additive_tolerance_trap",
            0.1,
            0.103,
            False,
            "the additive tolerance rule would pass but the required OR rule fails",
            "test_only",
        ),
    ]
    assert all(
        math.isfinite(item.reference) and math.isfinite(item.observed)
        for item in NUMERICAL_NEAR_MISSES
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        NUMERICAL_NEAR_MISSES[0].scope = "production"  # type: ignore[misc]


def test_local_predicate_matches_every_expected_outcome_and_rejects_nonfinite_values():
    assert [
        passes_authoritative_or(item.reference, item.observed)
        for item in NUMERICAL_NEAR_MISSES
    ] == [item.expected_pass for item in NUMERICAL_NEAR_MISSES]
    assert not passes_authoritative_or(float("nan"), 0.0)
    assert not passes_authoritative_or(0.0, float("inf"))


def test_additive_tolerance_would_wrongly_accept_the_explicit_trap():
    trap = NUMERICAL_NEAR_MISSES[-1]
    absolute_error = abs(trap.observed - trap.reference)

    assert absolute_error > ATOL
    assert absolute_error > RTOL * abs(trap.reference)
    assert absolute_error <= ATOL + RTOL * abs(trap.reference)
    assert not passes_authoritative_or(trap.reference, trap.observed)


def test_reference_defaults_finite_mask_and_or_structure_remain_unchanged():
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8"), filename=str(REFERENCE))
    parse_args = _function(tree, "parse_args")
    compare_outputs = _function(tree, "compare_outputs")
    names = {
        node.id for node in ast.walk(compare_outputs) if isinstance(node, ast.Name)
    }
    passed_mask = next(
        node
        for node in ast.walk(compare_outputs)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "passed_mask"
            for target in node.targets
        )
    )

    assert _option_default(parse_args, "--atol") == ATOL
    assert _option_default(parse_args, "--rtol") == RTOL
    assert "finite_mask" in names
    assert isinstance(passed_mask.value, ast.BinOp)
    assert isinstance(passed_mask.value.op, ast.BitAnd)
    assert isinstance(passed_mask.value.right, ast.BinOp)
    assert isinstance(passed_mask.value.right.op, ast.BitOr)
    assert ast.unparse(passed_mask.value.right) == "abs_ok | rel_ok"


def test_fixture_has_no_shape_fields_or_nonstandard_or_production_imports():
    tree = ast.parse(FIXTURE.read_text(encoding="utf-8"), filename=str(FIXTURE))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    annotations = {
        node.target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }

    assert imports <= {"__future__", "dataclasses"}
    assert imported_names == {"math"}
    assert (
        not {
            "shape",
            "batch_size",
            "sequence_length",
            "model_width",
            "head_count",
        }
        & annotations
    )
    assert not {"torch", "triton", "numpy", "ratchet", "benchmarks"} & imported_names
    assert "test-only" in ast.get_docstring(tree).lower()


def test_pool_has_zero_references_outside_test_surfaces_and_benchmark_hash_is_unchanged():
    references = []
    fixture_references = {
        "tests.fixtures.adversarial_pool",
        "tests/fixtures/adversarial_pool.py",
    }
    for directory in (
        ROOT / "ratchet",
        ROOT / "research" / "archive",
        ROOT / "research" / "paper",
    ):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {
                ".bib",
                ".py",
                ".json",
                ".md",
                ".tex",
            }:
                if any(
                    marker in path.read_text(encoding="utf-8")
                    for marker in fixture_references
                ):
                    references.append(path)

    assert references == []
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED_SHA256


def test_production_and_reporting_surfaces_cannot_import_or_name_synthetic_pool_data():
    imported_tests = []
    for directory in (ROOT / "ratchet", ROOT / "benchmarks"):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module]
                    if isinstance(node, ast.ImportFrom) and node.level == 0
                    else []
                )
                if any(name == "tests" or name.startswith("tests.") for name in names):
                    imported_tests.append(path)

    synthetic_ids = {item.case_id for item in NUMERICAL_NEAR_MISSES}
    synthetic_references = []
    for directory in (
        ROOT / "ratchet",
        ROOT / "benchmarks",
        ROOT / "research" / "archive",
        ROOT / "research" / "paper",
    ):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix in {
                ".bib",
                ".py",
                ".json",
                ".md",
                ".tex",
            }:
                text = path.read_text(encoding="utf-8")
                if any(identifier in text for identifier in synthetic_ids):
                    synthetic_references.append(path)

    assert imported_tests == []
    assert synthetic_references == []
