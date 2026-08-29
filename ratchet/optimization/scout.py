"""Citation-aware architectural scouting without candidate generation or execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_INTENT = re.compile(r"^SCOUT-[0-9]+$")
_CITATION = re.compile(r"^[a-z][a-z0-9_]*$")


class ScoutIntentState(str, Enum):
    OPEN = "open"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ScoutProposal:
    """A source-located architectural observation supplied to the planning scout."""

    intent_id: str
    technique: str
    regime: str
    citation_key: str
    source_locator: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.intent_id, str)
            or _INTENT.fullmatch(self.intent_id) is None
            or not all(
                isinstance(value, str) and value
                for value in (
                    self.technique,
                    self.regime,
                    self.citation_key,
                    self.source_locator,
                )
            )
            or _CITATION.fullmatch(self.citation_key) is None
        ):
            raise ValueError("scout proposal is invalid")


@dataclass(frozen=True, slots=True)
class ScoutIntent:
    """An auditable, FG-01-gated architectural intent rather than a candidate."""

    proposal: ScoutProposal
    state: ScoutIntentState
    reason: str | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.proposal, ScoutProposal)
            or not isinstance(self.state, ScoutIntentState)
            or (self.state is ScoutIntentState.OPEN) != (self.reason is None)
            or (
                self.reason is not None
                and (not isinstance(self.reason, str) or not self.reason)
            )
        ):
            raise ValueError("scout intent is invalid")

    @property
    def scope(self) -> str:
        return "planning_only"

    @property
    def qualification_gate(self) -> str:
        return "FG-01"

    @property
    def execution_permitted(self) -> bool:
        return False


def assess_scout_proposal(
    proposal: ScoutProposal, reviewed_citation_keys: tuple[str, ...]
) -> ScoutIntent:
    """Accept only a cited source already reviewed by the project's literature process."""

    if (
        not isinstance(proposal, ScoutProposal)
        or not isinstance(reviewed_citation_keys, tuple)
        or not all(
            isinstance(key, str) and _CITATION.fullmatch(key)
            for key in reviewed_citation_keys
        )
        or len(set(reviewed_citation_keys)) != len(reviewed_citation_keys)
    ):
        raise ValueError("reviewed citation keys are invalid")
    if proposal.citation_key not in reviewed_citation_keys:
        return ScoutIntent(proposal, ScoutIntentState.REJECTED, "citation_not_reviewed")
    return ScoutIntent(proposal, ScoutIntentState.OPEN, None)


__all__ = [
    "ScoutIntent",
    "ScoutIntentState",
    "ScoutProposal",
    "assess_scout_proposal",
]
