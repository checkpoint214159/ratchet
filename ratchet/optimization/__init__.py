"""Public contracts for human- and agent-directed optimization policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .controller import (
    AutoresearchInputs,
    AutoresearchOutcome,
    AutoresearchRequest,
    ControllerState,
    NoRunAutoresearchController,
    PreparedNoRunEvent,
    RepositoryAutoresearchInputs,
)
from .critic import (
    CriticDecision,
    CriticEpoch,
    CriticState,
    dormant_critic_decision,
)
from .human_queue import (
    FileHumanResearchQueue,
    HumanInputKind,
    HumanInputRecord,
    HumanInputSubmission,
    HumanQueueIntegrityError,
    HumanQueueItem,
    HumanQueueProjection,
    HumanResearchQueue,
)
from .scout import ScoutIntent, ScoutIntentState, ScoutProposal, assess_scout_proposal
from .search import (
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


class HypothesisSource(str, Enum):
    HUMAN = "human"
    LITERATURE = "literature"
    PROPOSER = "proposer"


@dataclass(frozen=True, slots=True)
class Hypothesis:
    hypothesis_id: str
    source: HypothesisSource
    statement: str

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.statement:
            raise ValueError("hypothesis fields must not be empty")


@dataclass(frozen=True, slots=True)
class OptimizationRequest:
    request_id: str
    hypothesis: Hypothesis

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")


class OptimizationController(Protocol):
    def prepare(self, request: AutoresearchRequest) -> AutoresearchOutcome: ...


__all__ = [
    "FileHumanResearchQueue",
    "AutoresearchInputs",
    "AutoresearchOutcome",
    "AutoresearchRequest",
    "ControllerState",
    "CriticDecision",
    "CriticEpoch",
    "CriticState",
    "HumanInputKind",
    "HumanInputRecord",
    "HumanInputSubmission",
    "HumanQueueIntegrityError",
    "HumanQueueItem",
    "HumanQueueProjection",
    "HumanResearchQueue",
    "Hypothesis",
    "HypothesisSource",
    "OptimizationController",
    "OptimizationRequest",
    "NoRunAutoresearchController",
    "PreparedNoRunEvent",
    "RepositoryAutoresearchInputs",
    "SearchAxis",
    "SearchCache",
    "SearchDefinition",
    "SearchFamily",
    "SearchInfeasibility",
    "SearchKind",
    "SearchPlan",
    "SearchPoint",
    "SearchProposal",
    "ScoutIntent",
    "ScoutIntentState",
    "ScoutProposal",
    "assess_scout_proposal",
    "dormant_critic_decision",
    "mark_considered",
    "mark_infeasible",
    "next_search_point",
    "plan_random_ablation",
    "plan_search",
]
