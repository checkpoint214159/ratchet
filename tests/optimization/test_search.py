"""CPU-only contracts for scoreless, bounded optimization-search planning."""

from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path

import pytest

from ratchet.optimization import (
    SearchAxis,
    SearchCache,
    SearchDefinition,
    SearchFamily,
    SearchInfeasibility,
    SearchKind,
    SearchPlan,
    SearchPoint,
    SearchProposal,
    mark_considered,
    mark_infeasible,
    next_search_point,
    plan_random_ablation,
    plan_search,
)
from tests.fixtures.search_objective import SYNTHETIC_SEARCH_CASES, synthetic_objective

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "benchmarks" / "reference" / "torch_transformer_benchmark.py"
REFERENCE_SHA256 = "5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e"


def _definition(
    *, kind: SearchKind = SearchKind.PARAMETRIC, budget: int = 4
) -> SearchDefinition:
    first = SearchFamily(
        "family-alpha",
        (
            SearchAxis("width", (1, 2)),
            SearchAxis("enabled", (True, False)),
        ),
    )
    if kind is SearchKind.PARAMETRIC:
        return SearchDefinition("IDEA-0001", "a" * 64, "b" * 64, kind, (first,), budget)
    second = SearchFamily("family-beta", (SearchAxis("variant", ("x", "y", "z")),))
    return SearchDefinition(
        "IDEA-0001", "a" * 64, "b" * 64, kind, (second, first), budget
    )


def test_parametric_plan_canonicalizes_axes_and_families_into_cartesian_points():
    plan = plan_search(_definition())
    reversed_definition = SearchDefinition(
        "IDEA-0001",
        "a" * 64,
        "b" * 64,
        SearchKind.PARAMETRIC,
        (
            SearchFamily(
                "family-alpha",
                (
                    SearchAxis("enabled", (True, False)),
                    SearchAxis("width", (1, 2)),
                ),
            ),
        ),
        4,
    )

    assert plan == plan_search(reversed_definition)
    assert [point.assignments for point in plan.points] == [
        (("enabled", False), ("width", 1)),
        (("enabled", False), ("width", 2)),
        (("enabled", True), ("width", 1)),
        (("enabled", True), ("width", 2)),
    ]
    assert plan.ordering == tuple(point.point_id for point in plan.points)
    assert _definition().scope == "planning_only"
    assert _definition().qualification_gate == "FG-01"
    assert _definition().execution_permitted is False


def test_architectural_plan_round_robins_canonical_family_streams_by_depth():
    plan = plan_search(_definition(kind=SearchKind.ARCHITECTURAL, budget=5))

    assert [point.family_id for point in plan.points] == [
        "family-alpha",
        "family-beta",
        "family-alpha",
        "family-beta",
        "family-alpha",
        "family-beta",
        "family-alpha",
    ]
    cache = SearchCache(plan.plan_digest, (), ())
    for point in plan.points[:5]:
        assert next_search_point(plan, cache).point == point
        cache = mark_considered(plan, cache, point.point_id)
    assert next_search_point(plan, cache) == SearchProposal(None, "budget_exhausted")


def test_value_permutations_canonicalize_to_the_same_plan():
    first = plan_search(_definition())
    reordered = SearchDefinition(
        "IDEA-0001",
        "a" * 64,
        "b" * 64,
        SearchKind.PARAMETRIC,
        (
            SearchFamily(
                "family-alpha",
                (
                    SearchAxis("width", (2, 1)),
                    SearchAxis("enabled", (False, True)),
                ),
            ),
        ),
        4,
    )

    assert first == plan_search(reordered)


def test_type_tagged_hashing_keeps_boolean_and_integer_assignments_distinct():
    definition = SearchDefinition(
        "IDEA-0001",
        "a" * 64,
        "b" * 64,
        SearchKind.PARAMETRIC,
        (SearchFamily("typed", (SearchAxis("choice", (True, 1)),)),),
        2,
    )
    plan = plan_search(definition)

    assert len({point.point_id for point in plan.points}) == 2
    assert plan.points[0].point_id != plan.points[1].point_id


def test_plan_and_cache_enforce_budget_exact_membership_and_atomic_infeasibility():
    plan = plan_search(_definition(budget=2))
    cache = SearchCache(plan.plan_digest, (), ())
    first = next_search_point(plan, cache)

    assert first.point == plan.points[0]
    assert first.stop_reason is None
    cache = mark_considered(plan, cache, first.point.point_id)
    second = next_search_point(plan, cache)
    assert second.point == plan.points[1]
    cache = mark_infeasible(
        plan,
        cache,
        SearchInfeasibility(
            second.point.point_id, "SYN-SEARCH-CONSTRAINT", "test only"
        ),
    )
    assert cache.considered_point_ids == tuple(sorted(cache.considered_point_ids))
    assert cache.infeasibilities[0].point_id == second.point.point_id
    assert next_search_point(plan, cache) == SearchProposal(None, "budget_exhausted")

    foreign = SearchCache(plan.plan_digest, ("c" * 64,), ())
    with pytest.raises(ValueError, match="outside"):
        next_search_point(plan, foreign)
    with pytest.raises(ValueError, match="exact plan"):
        next_search_point(plan, SearchCache("d" * 64, (), ()))
    with pytest.raises(ValueError):
        mark_considered(plan, cache, first.point.point_id)
    with pytest.raises(ValueError):
        mark_considered(plan, SearchCache(plan.plan_digest, (), ()), "c" * 64)
    with pytest.raises(ValueError):
        mark_infeasible(plan, cache, cache.infeasibilities[0])
    with pytest.raises(ValueError):
        mark_infeasible(
            plan,
            SearchCache(plan.plan_digest, (), ()),
            SearchInfeasibility("c" * 64, "constraint", "foreign"),
        )
    with pytest.raises(ValueError):
        SearchCache(plan.plan_digest, (first.point.point_id, first.point.point_id), ())


def test_space_exhaustion_and_random_ablation_are_deterministic_and_scoreless():
    one_point_definition = SearchDefinition(
        "IDEA-0001",
        "a" * 64,
        "b" * 64,
        SearchKind.PARAMETRIC,
        (SearchFamily("one", (SearchAxis("choice", (True,)),)),),
        1,
    )
    one_point_plan = plan_search(one_point_definition)
    exhausted = mark_considered(
        one_point_plan,
        SearchCache(one_point_plan.plan_digest, (), ()),
        one_point_plan.points[0].point_id,
    )
    assert next_search_point(one_point_plan, exhausted) == SearchProposal(
        None, "space_exhausted"
    )

    primary = plan_search(_definition())
    first = plan_random_ablation(primary, "SYN-SEARCH-SEED-A")
    second = plan_random_ablation(primary, "SYN-SEARCH-SEED-A")
    other = plan_random_ablation(primary, "SYN-SEARCH-SEED-B")
    assert first == second
    assert first.primary_plan_digest == primary.plan_digest
    assert first.points == primary.points
    assert first.budget == primary.budget
    assert set(first.ordering) == set(primary.ordering)
    assert first.ordering != other.ordering


@pytest.mark.parametrize(
    "definition",
    (
        SearchDefinition(
            "IDEA-0001",
            "a" * 64,
            "b" * 64,
            SearchKind.PARAMETRIC,
            (SearchFamily("only", (SearchAxis("x", (1,)),)),),
            1,
        ),
    ),
)
def test_definition_rejects_unbounded_space_bad_kind_shapes_and_over_budget(
    definition: SearchDefinition,
):
    assert plan_search(definition).budget == 1
    with pytest.raises(ValueError):
        SearchDefinition(
            "IDEA-0001",
            "a" * 64,
            "b" * 64,
            SearchKind.ARCHITECTURAL,
            (SearchFamily("only", (SearchAxis("x", (1,)),)),),
            1,
        )
    with pytest.raises(ValueError, match="budget"):
        SearchDefinition(
            "IDEA-0001",
            "a" * 64,
            "b" * 64,
            SearchKind.PARAMETRIC,
            (SearchFamily("only", (SearchAxis("x", (1,)),)),),
            257,
        )
    with pytest.raises(ValueError, match="4096"):
        plan_search(
            SearchDefinition(
                "IDEA-0001",
                "a" * 64,
                "b" * 64,
                SearchKind.PARAMETRIC,
                (SearchFamily("wide", (SearchAxis("x", tuple(range(4097))),)),),
                1,
            )
        )
    with pytest.raises(ValueError, match="budget"):
        plan_search(
            SearchDefinition(
                "IDEA-0001",
                "a" * 64,
                "b" * 64,
                SearchKind.PARAMETRIC,
                (SearchFamily("small", (SearchAxis("x", (1,)),)),),
                2,
            )
        )


def test_synthetic_fixture_is_test_only_and_search_has_no_objective_or_runtime_paths():
    assert [synthetic_objective(case) for case in SYNTHETIC_SEARCH_CASES] == [1, -2]
    source = ROOT / "ratchet" / "optimization" / "search.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    production_text = source.read_text(encoding="utf-8")
    forbidden_imports = {
        "ratchet.backends",
        "ratchet.experiments",
        "ratchet.measurement",
        "ratchet.models",
        "tests",
        "torch",
        "triton",
    }

    assert not {
        name
        for name in imports
        if any(
            name == root or name.startswith(f"{root}.") for root in forbidden_imports
        )
    }
    assert not {"objective", "score", "hardware", "measurement"} & set(
        production_text.lower().split()
    )
    for path in [
        *ROOT.glob("ratchet/**/*.py"),
        *ROOT.glob("benchmarks/**/*"),
        *ROOT.glob("research/archive/**/*"),
        *ROOT.glob("research/paper/**/*"),
    ]:
        if path.is_file() and path.suffix in {".py", ".json", ".tex", ".bib"}:
            assert "SYN-SEARCH-" not in path.read_text(encoding="utf-8")
    assert sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256


def test_public_records_reject_forged_points_and_invalid_terminal_shapes():
    with pytest.raises(ValueError):
        SearchPoint("a" * 64, "family", (("value", 1),))
    with pytest.raises(ValueError):
        SearchProposal(None, None)
    with pytest.raises(ValueError):
        SearchPlan("a" * 64, "b" * 64, "c" * 64, (), 1, ())
