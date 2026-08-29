"""Fail-closed preparation of the current build's immutable no-run event.

This module prepares bytes only.  Recording those bytes, executing a workload, and
choosing a candidate remain separate future boundaries.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from ratchet.experiments import (
    EVALUATOR_CONTRACT_DIGEST,
    NO_RUN_EVENT,
    SCHEMA_VERSION,
    ArchiveIntegrityError,
    EventId,
    ExperimentId,
    canonical_json,
    unavailable_xpu_execution_environment,
    validate_event_payload,
    validate_provisional_environment,
)

from .human_queue import FileHumanResearchQueue, HumanQueueProjection

_ENVIRONMENT_ID = "ENV-0001"
_ENVIRONMENT_DIGEST = "829b9496368f264f5f5a8ddf113adbebcec54934b55570ae5e3ebdefec3f5a3c"
_PROTOCOL_ID = "PROTO-INTEL-0001"
_PROTOCOL_DIGEST = "8497aaf0f9827ca6bedaea89d59a2146157324697fa918c0d84832ec6cdaa9c5"
_MAX_STEPS = 1
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


class ControllerState(str, Enum):
    """The intentionally bounded state sequence for no-run preparation."""

    ENVIRONMENT_PENDING = "environment_pending"
    ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
    NO_RUN_PREPARED = "no_run_prepared"


@dataclass(frozen=True, slots=True)
class AutoresearchRequest:
    """Caller-provided provenance for one bounded controller preparation."""

    event_id: str
    experiment_id: str
    timestamp: str
    researcher: str
    git_commit: str
    branch: str
    max_steps: int

    def __post_init__(self) -> None:
        EventId(self.event_id)
        ExperimentId(self.experiment_id)
        if (
            not all(
                isinstance(value, str) and value
                for value in (
                    self.timestamp,
                    self.researcher,
                    self.git_commit,
                    self.branch,
                )
            )
            or _COMMIT.fullmatch(self.git_commit) is None
            or not isinstance(self.max_steps, int)
            or isinstance(self.max_steps, bool)
            or not 1 <= self.max_steps <= _MAX_STEPS
        ):
            raise ValueError("autoresearch request fields are invalid")


@dataclass(frozen=True, slots=True)
class PreparedNoRunEvent:
    """Canonical, validated event bytes that have deliberately not been recorded."""

    payload_bytes: bytes
    payload_digest: str
    environment_digest: str
    protocol_digest: str
    queue_projection_id: str
    selected_idea_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload_bytes, bytes):
            raise ValueError("prepared no-run event fields are invalid")
        try:
            payload = json.loads(self.payload_bytes)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("prepared no-run event payload is invalid") from error
        try:
            if (
                not isinstance(payload, dict)
                or canonical_json(payload) != self.payload_bytes
            ):
                raise ValueError("prepared no-run event payload is not canonical")
            validate_event_payload(payload)
        except ArchiveIntegrityError as error:
            raise ValueError("prepared no-run event payload is invalid") from error
        if (
            payload.get("event_kind") != NO_RUN_EVENT
            or not _is_digest(self.payload_digest)
            or self.payload_digest != sha256(self.payload_bytes).hexdigest()
            or self.environment_digest != _ENVIRONMENT_DIGEST
            or self.protocol_digest != _PROTOCOL_DIGEST
            or payload.get("evaluator_contract_digest") != EVALUATOR_CONTRACT_DIGEST
            or payload.get("environment_id") != _ENVIRONMENT_ID
            or payload.get("environment_artifact_digest") != self.environment_digest
            or payload.get("execution_environment")
            != unavailable_xpu_execution_environment()
            or payload.get("intended_protocol")
            != f"{_PROTOCOL_ID}@sha256:{self.protocol_digest}"
            or not _is_digest(self.queue_projection_id)
            or not isinstance(self.selected_idea_id, str)
            or self.selected_idea_id != "IDEA-0001"
        ):
            raise ValueError("prepared no-run event fields are invalid")


@dataclass(frozen=True, slots=True)
class AutoresearchOutcome:
    """The terminal result of exactly one no-run preparation step."""

    state: ControllerState
    transitions: tuple[ControllerState, ...]
    steps_consumed: int
    continuation_allowed: bool
    prepared_event: PreparedNoRunEvent

    def __post_init__(self) -> None:
        if (
            self.state is not ControllerState.NO_RUN_PREPARED
            or self.transitions
            != (
                ControllerState.ENVIRONMENT_PENDING,
                ControllerState.ENVIRONMENT_UNAVAILABLE,
                ControllerState.NO_RUN_PREPARED,
            )
            or not isinstance(self.steps_consumed, int)
            or isinstance(self.steps_consumed, bool)
            or self.steps_consumed != 1
            or self.continuation_allowed is not False
            or not isinstance(self.prepared_event, PreparedNoRunEvent)
        ):
            raise ValueError("autoresearch outcome must be terminal no-run preparation")


class AutoresearchInputs(Protocol):
    """Read-only inputs, deliberately ordered by the controller's no-run gate."""

    def environment_bytes(self) -> bytes: ...

    def queue_projection(self) -> HumanQueueProjection: ...

    def protocol_bytes(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class RepositoryAutoresearchInputs:
    """Read the fixed project facts without recording or executing anything."""

    project_root: Path

    def environment_bytes(self) -> bytes:
        return (
            self.project_root / "research" / "environment" / "ENV-0001.json"
        ).read_bytes()

    def queue_projection(self) -> HumanQueueProjection:
        return FileHumanResearchQueue(self.project_root).projection()

    def protocol_bytes(self) -> bytes:
        return (
            self.project_root / "research" / "protocols" / "PROTO-INTEL-0001.json"
        ).read_bytes()


def _json_mapping(raw: object, name: str) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise ValueError(f"{name} bytes are invalid")
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} bytes are not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} bytes are not an object")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"protocol {name} is invalid")
    return value


def _protocol(
    raw: bytes, projection: HumanQueueProjection
) -> tuple[str, str, str, tuple[str, ...], dict[str, object]]:
    value = _json_mapping(raw, "protocol")
    if sha256(raw).hexdigest() != _PROTOCOL_DIGEST:
        raise ValueError("protocol requires redirect before preparation")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("protocol_id") != _PROTOCOL_ID
        or value.get("scope") != "definition_only_future_protocol"
        or value.get("execution_status") != "not_run_hardware_unavailable"
        or value.get("execution_permitted") is not False
    ):
        raise ValueError("protocol is not the approved no-run definition")
    if not projection.items:
        raise ValueError("queue projection has no active idea")
    selected_idea_id = projection.items[0].idea_id
    if value.get("source_idea_id") != selected_idea_id:
        raise ValueError("protocol source idea must be the first active queue item")
    evaluator = value.get("evaluator_contract")
    if not isinstance(evaluator, dict) or evaluator != {
        "relative_path": "benchmarks/reference/torch_transformer_benchmark.py",
        "sha256": EVALUATOR_CONTRACT_DIGEST,
    }:
        raise ValueError("protocol evaluator contract is invalid")
    hypothesis = _nonempty_string(value.get("hypothesis"), "hypothesis")
    motivation = _nonempty_string(value.get("motivation"), "motivation")
    keys = value.get("literature_keys")
    if not isinstance(keys, list) or not all(
        isinstance(key, str) and key for key in keys
    ):
        raise ValueError("protocol literature keys are invalid")
    cases = value.get("evaluation_cases")
    if not isinstance(cases, list) or not cases or not isinstance(cases[0], dict):
        raise ValueError("protocol requires a first planned benchmark configuration")
    configuration = cases[0]
    if configuration.get("case_id") != "default":
        raise ValueError("protocol first benchmark configuration must be default")
    return (
        _PROTOCOL_DIGEST,
        hypothesis,
        motivation,
        tuple(keys),
        configuration,
    )


class NoRunAutoresearchController:
    """Prepare one canonical no-run event after the pinned environment gate."""

    def __init__(self, inputs: AutoresearchInputs) -> None:
        self._inputs = inputs

    def prepare(self, request: AutoresearchRequest) -> AutoresearchOutcome:
        environment_raw = self._inputs.environment_bytes()
        try:
            environment = _json_mapping(environment_raw, "environment")
        except ValueError as error:
            raise ValueError(
                "environment requires redirect before preparation"
            ) from error
        try:
            validate_provisional_environment(environment)
        except ArchiveIntegrityError as error:
            raise ValueError(
                "environment requires redirect before preparation"
            ) from error
        if (
            environment.get("environment_id") != _ENVIRONMENT_ID
            or sha256(environment_raw).hexdigest() != _ENVIRONMENT_DIGEST
        ):
            raise ValueError("environment requires redirect before preparation")

        projection = self._inputs.queue_projection()
        if not isinstance(projection, HumanQueueProjection):
            raise ValueError("queue projection is invalid")
        protocol_raw = self._inputs.protocol_bytes()
        (
            protocol_digest,
            hypothesis,
            motivation,
            literature_keys,
            configuration,
        ) = _protocol(protocol_raw, projection)

        reason = _nonempty_string(
            environment.get("backend_doctor_reason"), "environment reason"
        )
        environment_id = _nonempty_string(
            environment.get("environment_id"), "environment id"
        )
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "event_kind": NO_RUN_EVENT,
            "classification": "no_run",
            "event_id": request.event_id,
            "experiment_id": request.experiment_id,
            "environment_id": environment_id,
            "environment_artifact_digest": _ENVIRONMENT_DIGEST,
            "evaluator_contract_digest": EVALUATOR_CONTRACT_DIGEST,
            "timestamp": request.timestamp,
            "researcher": request.researcher,
            "git_commit": request.git_commit,
            "branch": request.branch,
            "execution_environment": unavailable_xpu_execution_environment(),
            "benchmark_configuration": configuration,
            "candidate": {
                "implementation_id": None,
                "source_digest": None,
                "state": "not_generated",
            },
            "hypothesis": hypothesis,
            "motivation": motivation,
            "literature_refs": list(literature_keys),
            "decision": "stop",
            "decision_reason": f"{environment_id}: {environment['decision']}; {reason}",
            "artifact_digests": [],
            "paper_inclusion": True,
            "intended_protocol": f"{_PROTOCOL_ID}@sha256:{protocol_digest}",
            "stop_reason": f"{environment_id}: {reason}; empirical work is not permitted",
        }
        try:
            validate_event_payload(payload)
        except ArchiveIntegrityError as error:
            raise ValueError("prepared no-run payload is invalid") from error
        payload_bytes = canonical_json(payload)
        prepared = PreparedNoRunEvent(
            payload_bytes=payload_bytes,
            payload_digest=sha256(payload_bytes).hexdigest(),
            environment_digest=_ENVIRONMENT_DIGEST,
            protocol_digest=protocol_digest,
            queue_projection_id=projection.projection_id,
            selected_idea_id=projection.items[0].idea_id,
        )
        return AutoresearchOutcome(
            state=ControllerState.NO_RUN_PREPARED,
            transitions=(
                ControllerState.ENVIRONMENT_PENDING,
                ControllerState.ENVIRONMENT_UNAVAILABLE,
                ControllerState.NO_RUN_PREPARED,
            ),
            steps_consumed=1,
            continuation_allowed=False,
            prepared_event=prepared,
        )


__all__ = [
    "AutoresearchInputs",
    "AutoresearchOutcome",
    "AutoresearchRequest",
    "ControllerState",
    "NoRunAutoresearchController",
    "PreparedNoRunEvent",
    "RepositoryAutoresearchInputs",
]
