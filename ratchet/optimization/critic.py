"""Epoch-frozen critic records for a future measured autoresearch loop.

The current build has no empirical evidence, so this module can only retain an
explicit dormant decision. It neither reads the archive nor evaluates a candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_EPOCH = re.compile(r"^CRITIC-EPOCH-[0-9]+$")
_CANDIDATE = re.compile(r"^CAND-[A-Za-z0-9][A-Za-z0-9._-]*$")


class CriticState(str, Enum):
    DORMANT = "dormant"


@dataclass(frozen=True, slots=True)
class CriticEpoch:
    """A candidate-held-out, immutable epoch split for a future critic."""

    epoch_id: str
    training_candidate_ids: tuple[str, ...]
    held_out_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        training = self.training_candidate_ids
        held_out = self.held_out_candidate_ids
        if (
            not isinstance(self.epoch_id, str)
            or _EPOCH.fullmatch(self.epoch_id) is None
            or not isinstance(training, tuple)
            or not isinstance(held_out, tuple)
            or not training
            or not held_out
            or not all(
                isinstance(item, str) and _CANDIDATE.fullmatch(item)
                for item in training
            )
            or not all(
                isinstance(item, str) and _CANDIDATE.fullmatch(item)
                for item in held_out
            )
            or len(set(training)) != len(training)
            or len(set(held_out)) != len(held_out)
            or set(training) & set(held_out)
        ):
            raise ValueError("critic epoch must hold out whole distinct candidates")


@dataclass(frozen=True, slots=True)
class CriticDecision:
    """A non-score decision whose provenance stays tied to one frozen epoch."""

    epoch_id: str
    candidate_id: str
    state: CriticState
    reason: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.epoch_id, str)
            or _EPOCH.fullmatch(self.epoch_id) is None
            or not isinstance(self.candidate_id, str)
            or _CANDIDATE.fullmatch(self.candidate_id) is None
            or self.state is not CriticState.DORMANT
            or self.reason != "no_empirical_measurements"
        ):
            raise ValueError("critic decision is invalid")


def dormant_critic_decision(
    epoch: CriticEpoch, candidate_id: str, empirical_event_count: int
) -> CriticDecision:
    """Record that no critic score exists until measured candidate evidence is available."""

    if (
        not isinstance(epoch, CriticEpoch)
        or not isinstance(candidate_id, str)
        or _CANDIDATE.fullmatch(candidate_id) is None
        or not isinstance(empirical_event_count, int)
        or isinstance(empirical_event_count, bool)
        or empirical_event_count != 0
    ):
        raise ValueError("current critic requires a zero empirical-event catalogue")
    return CriticDecision(
        epoch.epoch_id,
        candidate_id,
        CriticState.DORMANT,
        "no_empirical_measurements",
    )


__all__ = [
    "CriticDecision",
    "CriticEpoch",
    "CriticState",
    "dormant_critic_decision",
]
