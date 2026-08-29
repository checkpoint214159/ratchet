"""Pure, bounded planning records for future optimization search.

The module deliberately describes finite choices only.  It neither evaluates a
choice nor records a result, so a future execution boundary must remain explicit.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

_SHA = re.compile(r"^[0-9a-f]{64}$")
_IDEA = re.compile(r"^IDEA-[0-9]+$")
_MAX_POINTS = 4096
_MAX_BUDGET = 256
_STOP_REASONS = frozenset({"budget_exhausted", "space_exhausted"})

SearchValue = str | int | bool


def _is_value(value: object) -> bool:
    return type(value) in {str, int, bool} and (
        not isinstance(value, str) or bool(value)
    )


def _tag(value: SearchValue) -> dict[str, object]:
    if type(value) is bool:
        return {"type": "bool", "value": value}
    if type(value) is int:
        return {"type": "int", "value": value}
    if type(value) is str:
        return {"type": "str", "value": value}
    raise ValueError("search values must be tagged scalars")


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("search content is not canonical JSON") from error


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


class SearchKind(str, Enum):
    PARAMETRIC = "parametric"
    ARCHITECTURAL = "architectural"


@dataclass(frozen=True, slots=True)
class SearchAxis:
    name: str
    values: tuple[SearchValue, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not isinstance(self.values, tuple)
            or not self.values
            or not all(_is_value(value) for value in self.values)
            or len({_canonical_bytes(_tag(value)) for value in self.values})
            != len(self.values)
        ):
            raise ValueError("search axis fields are invalid")


@dataclass(frozen=True, slots=True)
class SearchFamily:
    family_id: str
    axes: tuple[SearchAxis, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.family_id, str)
            or not self.family_id
            or not isinstance(self.axes, tuple)
            or not self.axes
            or not all(isinstance(axis, SearchAxis) for axis in self.axes)
            or len({axis.name for axis in self.axes}) != len(self.axes)
        ):
            raise ValueError("search family fields are invalid")


@dataclass(frozen=True, slots=True)
class SearchDefinition:
    source_idea_id: str
    queue_projection_id: str
    protocol_digest: str
    kind: SearchKind
    families: tuple[SearchFamily, ...]
    budget: int

    def __post_init__(self) -> None:
        family_count = len(self.families) if isinstance(self.families, tuple) else 0
        if (
            not isinstance(self.budget, int)
            or isinstance(self.budget, bool)
            or not 1 <= self.budget <= _MAX_BUDGET
        ):
            raise ValueError("search budget is invalid")
        if (
            not isinstance(self.source_idea_id, str)
            or _IDEA.fullmatch(self.source_idea_id) is None
            or not _is_digest(self.queue_projection_id)
            or not _is_digest(self.protocol_digest)
            or not isinstance(self.kind, SearchKind)
            or not isinstance(self.families, tuple)
            or not all(isinstance(family, SearchFamily) for family in self.families)
            or len({family.family_id for family in self.families}) != family_count
            or (self.kind is SearchKind.PARAMETRIC and family_count != 1)
            or (self.kind is SearchKind.ARCHITECTURAL and family_count < 2)
        ):
            raise ValueError("search definition fields are invalid")

    @property
    def scope(self) -> str:
        return "planning_only"

    @property
    def qualification_gate(self) -> str:
        return "FG-01"

    @property
    def execution_permitted(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class SearchPoint:
    point_id: str
    family_id: str
    assignments: tuple[tuple[str, SearchValue], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.family_id, str)
            or not self.family_id
            or not isinstance(self.assignments, tuple)
            or not self.assignments
            or any(
                not isinstance(item, tuple) or len(item) != 2
                for item in self.assignments
            )
        ):
            raise ValueError("search point fields are invalid")
        names = tuple(name for name, _ in self.assignments)
        if (
            any(
                not isinstance(name, str) or not name or not _is_value(value)
                for name, value in self.assignments
            )
            or names != tuple(sorted(names))
            or len(set(names)) != len(names)
            or self.point_id != _point_id(self.family_id, self.assignments)
        ):
            raise ValueError("search point fields are invalid")


def _point_id(family_id: str, assignments: tuple[tuple[str, SearchValue], ...]) -> str:
    return _digest(
        {
            "family_id": family_id,
            "assignments": [
                {"name": name, "value": _tag(value)} for name, value in assignments
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class SearchInfeasibility:
    point_id: str
    constraint_id: str
    reason: str

    def __post_init__(self) -> None:
        if (
            not _is_digest(self.point_id)
            or not isinstance(self.constraint_id, str)
            or not self.constraint_id
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            raise ValueError("search infeasibility fields are invalid")


@dataclass(frozen=True, slots=True)
class SearchCache:
    plan_digest: str
    considered_point_ids: tuple[str, ...]
    infeasibilities: tuple[SearchInfeasibility, ...]

    def __post_init__(self) -> None:
        considered = self.considered_point_ids
        infeasible = self.infeasibilities
        if (
            not _is_digest(self.plan_digest)
            or not isinstance(considered, tuple)
            or not all(_is_digest(point_id) for point_id in considered)
            or len(set(considered)) != len(considered)
            or considered != tuple(sorted(considered))
            or not isinstance(infeasible, tuple)
            or not all(isinstance(record, SearchInfeasibility) for record in infeasible)
            or len({record.point_id for record in infeasible}) != len(infeasible)
            or any(record.point_id not in considered for record in infeasible)
            or infeasible
            != tuple(sorted(infeasible, key=lambda record: record.point_id))
        ):
            raise ValueError("search cache fields are invalid")


@dataclass(frozen=True, slots=True)
class SearchPlan:
    plan_digest: str
    definition_digest: str
    space_digest: str
    ordering: tuple[str, ...]
    budget: int
    points: tuple[SearchPoint, ...]
    primary_plan_digest: str | None = None
    seed: SearchValue | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.points, tuple)
            or not self.points
            or not all(isinstance(point, SearchPoint) for point in self.points)
        ):
            raise ValueError("search plan fields are invalid")
        point_ids = tuple(point.point_id for point in self.points)
        if (
            not _is_digest(self.definition_digest)
            or not _is_digest(self.space_digest)
            or len(set(point_ids)) != len(point_ids)
            or self.space_digest != _space_digest(self.points)
            or not isinstance(self.ordering, tuple)
            or set(self.ordering) != set(point_ids)
            or len(self.ordering) != len(point_ids)
            or not all(_is_digest(point_id) for point_id in self.ordering)
            or not isinstance(self.budget, int)
            or isinstance(self.budget, bool)
            or not 1 <= self.budget <= min(len(self.points), _MAX_BUDGET)
            or (self.primary_plan_digest is None) != (self.seed is None)
            or (
                self.primary_plan_digest is not None
                and not _is_digest(self.primary_plan_digest)
            )
            or (self.seed is not None and not _is_value(self.seed))
            or self.plan_digest
            != _plan_digest(
                self.definition_digest,
                self.space_digest,
                self.ordering,
                self.budget,
                self.primary_plan_digest,
                self.seed,
            )
        ):
            raise ValueError("search plan fields are invalid")


@dataclass(frozen=True, slots=True)
class SearchProposal:
    point: SearchPoint | None
    stop_reason: str | None

    def __post_init__(self) -> None:
        if (
            (self.point is None) == (self.stop_reason is None)
            or (self.point is not None and not isinstance(self.point, SearchPoint))
            or (self.stop_reason is not None and self.stop_reason not in _STOP_REASONS)
        ):
            raise ValueError("search proposal fields are invalid")


def _canonical_axis(axis: SearchAxis) -> SearchAxis:
    return SearchAxis(
        axis.name,
        tuple(sorted(axis.values, key=lambda value: _canonical_bytes(_tag(value)))),
    )


def _canonical_family(family: SearchFamily) -> SearchFamily:
    return SearchFamily(
        family.family_id,
        tuple(
            sorted(
                (_canonical_axis(axis) for axis in family.axes),
                key=lambda axis: axis.name,
            )
        ),
    )


def _canonical_families(definition: SearchDefinition) -> tuple[SearchFamily, ...]:
    return tuple(
        sorted(
            (_canonical_family(family) for family in definition.families),
            key=lambda family: family.family_id,
        )
    )


def _definition_digest(
    definition: SearchDefinition, families: tuple[SearchFamily, ...]
) -> str:
    return _digest(
        {
            "source_idea_id": definition.source_idea_id,
            "queue_projection_id": definition.queue_projection_id,
            "protocol_digest": definition.protocol_digest,
            "kind": definition.kind.value,
            "families": [
                {
                    "family_id": family.family_id,
                    "axes": [
                        {
                            "name": axis.name,
                            "values": [_tag(value) for value in axis.values],
                        }
                        for axis in family.axes
                    ],
                }
                for family in families
            ],
            "budget": definition.budget,
        }
    )


def _family_points(family: SearchFamily) -> tuple[SearchPoint, ...]:
    points: list[SearchPoint] = []
    for values in itertools.product(*(axis.values for axis in family.axes)):
        assignments = tuple(
            zip((axis.name for axis in family.axes), values, strict=True)
        )
        points.append(
            SearchPoint(
                _point_id(family.family_id, assignments), family.family_id, assignments
            )
        )
        if len(points) > _MAX_POINTS:
            raise ValueError("search space exceeds 4096 points")
    return tuple(points)


def _space_digest(points: tuple[SearchPoint, ...]) -> str:
    return _digest({"point_ids": [point.point_id for point in points]})


def _plan_digest(
    definition_digest: str,
    space_digest: str,
    ordering: tuple[str, ...],
    budget: int,
    primary_plan_digest: str | None,
    seed: SearchValue | None,
) -> str:
    return _digest(
        {
            "definition_digest": definition_digest,
            "space_digest": space_digest,
            "ordering": list(ordering),
            "budget": budget,
            "primary_plan_digest": primary_plan_digest,
            "seed": _tag(seed) if seed is not None else None,
        }
    )


def plan_search(definition: SearchDefinition) -> SearchPlan:
    """Produce one bounded, scoreless plan from canonicalized definition content."""
    if not isinstance(definition, SearchDefinition):
        raise ValueError("search definition is invalid")
    families = _canonical_families(definition)
    streams = tuple(_family_points(family) for family in families)
    if definition.kind is SearchKind.PARAMETRIC:
        points = streams[0]
    else:
        points = tuple(
            point
            for depth in range(max(len(stream) for stream in streams))
            for stream in streams
            if depth < len(stream)
            for point in (stream[depth],)
        )
    if len(points) > _MAX_POINTS:
        raise ValueError("search space exceeds 4096 points")
    if definition.budget > min(len(points), _MAX_BUDGET):
        raise ValueError("search budget exceeds available bounded points")
    definition_digest = _definition_digest(definition, families)
    space_digest = _space_digest(points)
    ordering = tuple(point.point_id for point in points)
    return SearchPlan(
        _plan_digest(
            definition_digest, space_digest, ordering, definition.budget, None, None
        ),
        definition_digest,
        space_digest,
        ordering,
        definition.budget,
        points,
    )


def plan_random_ablation(primary_plan: SearchPlan, seed: SearchValue) -> SearchPlan:
    """Derive a deterministic SHA-ordered ablation without changing the space."""
    if (
        not isinstance(primary_plan, SearchPlan)
        or primary_plan.primary_plan_digest is not None
        or not _is_value(seed)
    ):
        raise ValueError("primary search plan or seed is invalid")
    ordering = tuple(
        point.point_id
        for point in sorted(
            primary_plan.points,
            key=lambda point: (
                _digest({"seed": _tag(seed), "point_id": point.point_id}),
                point.point_id,
            ),
        )
    )
    return SearchPlan(
        _plan_digest(
            primary_plan.definition_digest,
            primary_plan.space_digest,
            ordering,
            primary_plan.budget,
            primary_plan.plan_digest,
            seed,
        ),
        primary_plan.definition_digest,
        primary_plan.space_digest,
        ordering,
        primary_plan.budget,
        primary_plan.points,
        primary_plan.plan_digest,
        seed,
    )


def _validate_cache(plan: SearchPlan, cache: SearchCache) -> None:
    if (
        not isinstance(plan, SearchPlan)
        or not isinstance(cache, SearchCache)
        or cache.plan_digest != plan.plan_digest
    ):
        raise ValueError("search cache does not belong to this exact plan")
    point_ids = {point.point_id for point in plan.points}
    cached_ids = set(cache.considered_point_ids)
    if not cached_ids <= point_ids or any(
        record.point_id not in point_ids for record in cache.infeasibilities
    ):
        raise ValueError("search cache contains points outside this exact plan")


def next_search_point(plan: SearchPlan, cache: SearchCache) -> SearchProposal:
    """Return the next unconsidered point, or one explicit terminal reason."""
    _validate_cache(plan, cache)
    considered = set(cache.considered_point_ids)
    by_id = {point.point_id: point for point in plan.points}
    for point_id in plan.ordering:
        if point_id not in considered:
            if len(cache.considered_point_ids) >= plan.budget:
                return SearchProposal(None, "budget_exhausted")
            return SearchProposal(by_id[point_id], None)
    return SearchProposal(None, "space_exhausted")


def mark_considered(plan: SearchPlan, cache: SearchCache, point_id: str) -> SearchCache:
    """Return a new exact-plan cache that records one considered point only."""
    _validate_cache(plan, cache)
    if (
        not _is_digest(point_id)
        or point_id not in {point.point_id for point in plan.points}
        or point_id in cache.considered_point_ids
    ):
        raise ValueError("search point cannot be considered")
    return SearchCache(
        cache.plan_digest,
        tuple(sorted((*cache.considered_point_ids, point_id))),
        cache.infeasibilities,
    )


def mark_infeasible(
    plan: SearchPlan, cache: SearchCache, record: SearchInfeasibility
) -> SearchCache:
    """Atomically retain one exact-plan infeasibility without a score."""
    _validate_cache(plan, cache)
    if (
        not isinstance(record, SearchInfeasibility)
        or record.point_id not in {point.point_id for point in plan.points}
        or record.point_id in cache.considered_point_ids
        or any(item.point_id == record.point_id for item in cache.infeasibilities)
    ):
        raise ValueError("search infeasibility cannot be recorded")
    return SearchCache(
        cache.plan_digest,
        tuple(sorted((*cache.considered_point_ids, record.point_id))),
        tuple(sorted((*cache.infeasibilities, record), key=lambda item: item.point_id)),
    )


__all__ = [
    "SearchAxis",
    "SearchCache",
    "SearchDefinition",
    "SearchFamily",
    "SearchInfeasibility",
    "SearchKind",
    "SearchPlan",
    "SearchPoint",
    "SearchProposal",
    "mark_considered",
    "mark_infeasible",
    "next_search_point",
    "plan_random_ablation",
    "plan_search",
]
