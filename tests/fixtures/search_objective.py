"""Test-only synthetic search fixtures; production search never imports this module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyntheticSearchCase:
    fixture_id: str
    width: int
    enabled: bool
    infeasible: bool
    scope: str = "test_only"

    def __post_init__(self) -> None:
        if (
            not self.fixture_id.startswith("SYN-SEARCH-")
            or self.width < 0
            or self.scope != "test_only"
        ):
            raise ValueError("synthetic search fixture is invalid")


SYNTHETIC_SEARCH_CASES = (
    SyntheticSearchCase("SYN-SEARCH-001", 1, True, False),
    SyntheticSearchCase("SYN-SEARCH-002", 2, False, True),
)


def synthetic_objective(case: SyntheticSearchCase) -> int:
    """A deliberately test-only scalar objective that production must not call."""
    return case.width if case.enabled and not case.infeasible else -case.width
