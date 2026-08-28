"""Public contracts for evidence-driven implementation selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ratchet.backends import BackendIdentity


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    regime: str
    backend: BackendIdentity

    def __post_init__(self) -> None:
        if not self.regime:
            raise ValueError("regime must not be empty")


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    implementation_id: str
    is_tuned: bool
    reason: str

    def __post_init__(self) -> None:
        if not self.implementation_id or not self.reason:
            raise ValueError("dispatch decision fields must not be empty")


class DispatchPolicy(Protocol):
    def choose(self, request: DispatchRequest) -> DispatchDecision: ...
