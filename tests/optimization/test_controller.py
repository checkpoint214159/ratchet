"""CPU-only contracts for the fail-closed autoresearch preparation boundary."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from ratchet.experiments import (
    canonical_json,
    unavailable_xpu_execution_environment,
    validate_event_payload,
)
from ratchet.optimization import (
    AutoresearchOutcome,
    AutoresearchRequest,
    ControllerState,
    NoRunAutoresearchController,
    OptimizationController,
    PreparedNoRunEvent,
    RepositoryAutoresearchInputs,
)
from ratchet.optimization.human_queue import HumanQueueItem, HumanQueueProjection

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = ROOT / "research" / "environment" / "ENV-0001.json"
PROTOCOL = ROOT / "research" / "protocols" / "PROTO-INTEL-0001.json"
ARCHIVE = ROOT / "research" / "archive"


class _Inputs:
    def __init__(
        self,
        environment: bytes = ENVIRONMENT.read_bytes(),
        projection: HumanQueueProjection | None = None,
        protocol: bytes = PROTOCOL.read_bytes(),
    ) -> None:
        self.environment = environment
        self.projection = projection or _projection()
        self.protocol = protocol
        self.calls: list[str] = []

    def environment_bytes(self) -> bytes:
        self.calls.append("environment")
        return self.environment

    def queue_projection(self) -> HumanQueueProjection:
        self.calls.append("queue")
        return self.projection

    def protocol_bytes(self) -> bytes:
        self.calls.append("protocol")
        return self.protocol


def _projection(idea_id: str = "IDEA-0001") -> HumanQueueProjection:
    item = HumanQueueItem(
        idea_id=idea_id,
        statement="future investigation",
        literature_keys=("ansel2024pytorch",),
        priority=0,
        constraints=(),
        creation_sequence=1,
    )
    return HumanQueueProjection("a" * 64, (item,))


def _request() -> AutoresearchRequest:
    return AutoresearchRequest(
        event_id="EVT-000001",
        experiment_id="EXP-0001",
        timestamp="2026-08-29T01:41:39+08:00",
        researcher="ratchet",
        git_commit="31a55293ff2bbf27ccdc4f62e7b0cb15f04eed7c",
        branch="main",
        max_steps=1,
    )


def _archive_bytes() -> dict[Path, bytes]:
    return {
        path.relative_to(ARCHIVE): path.read_bytes()
        for path in sorted(ARCHIVE.rglob("*"))
        if path.is_file()
    }


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value))
    return set()


def test_prepare_returns_stable_valid_canonical_no_run_bytes_without_archive_mutation():
    before = _archive_bytes()
    controller = NoRunAutoresearchController(_Inputs())

    first = controller.prepare(_request())
    second = controller.prepare(_request())
    payload = json.loads(first.prepared_event.payload_bytes)

    assert first == second
    assert first.state is ControllerState.NO_RUN_PREPARED
    assert first.transitions == (
        ControllerState.ENVIRONMENT_PENDING,
        ControllerState.ENVIRONMENT_UNAVAILABLE,
        ControllerState.NO_RUN_PREPARED,
    )
    assert first.steps_consumed == 1
    assert first.continuation_allowed is False
    assert (
        first.prepared_event.payload_digest
        == sha256(first.prepared_event.payload_bytes).hexdigest()
    )
    assert (
        first.prepared_event.environment_digest
        == sha256(ENVIRONMENT.read_bytes()).hexdigest()
    )
    assert payload["intended_protocol"] == (
        "PROTO-INTEL-0001@sha256:" + sha256(PROTOCOL.read_bytes()).hexdigest()
    )
    assert payload["candidate"] == {
        "implementation_id": None,
        "source_digest": None,
        "state": "not_generated",
    }
    assert payload["decision"] == "stop"
    assert "PyTorch is not installed" in payload["decision_reason"]
    assert "PyTorch is not installed" in payload["stop_reason"]
    validate_event_payload(payload)
    assert _archive_bytes() == before
    events = json.loads((ARCHIVE / "manifest.json").read_text())["events"]
    assert [event["event_id"] for event in events] == ["EVT-000001"]
    assert [event["event_kind"] for event in events] == ["no_run"]


@pytest.mark.parametrize(
    "environment",
    (
        b"{}",
        ENVIRONMENT.read_bytes() + b"\n",
        json.dumps(
            json.loads(ENVIRONMENT.read_bytes()) | {"availability": "available"}
        ).encode(),
    ),
)
def test_environment_gate_fails_closed_before_queue_or_protocol(environment: bytes):
    inputs = _Inputs(environment=environment)

    with pytest.raises(ValueError):
        NoRunAutoresearchController(inputs).prepare(_request())

    assert inputs.calls == ["environment"]


def test_protocol_must_select_the_first_active_queue_idea():
    inputs = _Inputs(projection=_projection("IDEA-0002"))

    with pytest.raises(ValueError, match="first active queue item"):
        NoRunAutoresearchController(inputs).prepare(_request())

    assert inputs.calls == ["environment", "queue", "protocol"]


def test_altered_protocol_bytes_fail_closed_after_the_environment_gate():
    inputs = _Inputs(protocol=PROTOCOL.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="protocol requires redirect"):
        NoRunAutoresearchController(inputs).prepare(_request())

    assert inputs.calls == ["environment", "queue", "protocol"]


def test_payload_has_no_empirical_or_execution_fields_and_the_step_is_bounded():
    payload = json.loads(
        NoRunAutoresearchController(_Inputs())
        .prepare(_request())
        .prepared_event.payload_bytes
    )
    forbidden = {
        "measurement_status",
        "correctness_result",
        "timing_result",
        "memory_result",
        "baseline_comparison",
        "current_best_comparison",
        "speedup",
        "profile",
        "trace",
        "counter",
        "workspace",
        "compilation",
        "execution_result",
    }

    assert not (_keys(payload) & forbidden)
    with pytest.raises(ValueError, match="request"):
        replace(_request(), max_steps=2)


def test_prepared_event_constructor_binds_canonical_payload_and_its_digests():
    prepared = NoRunAutoresearchController(_Inputs()).prepare(_request()).prepared_event
    payload = json.loads(prepared.payload_bytes)

    assert PreparedNoRunEvent(**_prepared_fields(prepared)) == prepared
    with pytest.raises(ValueError, match="payload"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_bytes": prepared.payload_bytes + b"\n",
            }
        )
    with pytest.raises(ValueError, match="payload"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_bytes": canonical_json({"arbitrary": "forgery"}),
                "payload_digest": sha256(
                    canonical_json({"arbitrary": "forgery"})
                ).hexdigest(),
            }
        )
    arbitrary_list = canonical_json([])
    with pytest.raises(ValueError, match="payload"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_bytes": arbitrary_list,
                "payload_digest": sha256(arbitrary_list).hexdigest(),
            }
        )
    with pytest.raises(ValueError, match="fields"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_digest": "b" * 64,
            }
        )

    payload["environment_artifact_digest"] = "b" * 64
    forged_environment = canonical_json(payload)
    with pytest.raises(ValueError, match="fields"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_bytes": forged_environment,
                "payload_digest": sha256(forged_environment).hexdigest(),
            }
        )
    payload["environment_artifact_digest"] = prepared.environment_digest
    payload["intended_protocol"] = "PROTO-INTEL-0001@sha256:" + "b" * 64
    forged_protocol = canonical_json(payload)
    with pytest.raises(ValueError, match="fields"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_bytes": forged_protocol,
                "payload_digest": sha256(forged_protocol).hexdigest(),
            }
        )

    payload["intended_protocol"] = "PROTO-INTEL-0001@sha256:" + prepared.protocol_digest
    payload["evaluator_contract_digest"] = "b" * 64
    forged_evaluator = canonical_json(payload)
    with pytest.raises(ValueError, match="payload"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_bytes": forged_evaluator,
                "payload_digest": sha256(forged_evaluator).hexdigest(),
            }
        )
    payload["evaluator_contract_digest"] = (
        "5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e"
    )
    payload["environment_id"] = "ENV-0002"
    forged_environment_id = canonical_json(payload)
    with pytest.raises(ValueError, match="fields"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_bytes": forged_environment_id,
                "payload_digest": sha256(forged_environment_id).hexdigest(),
            }
        )
    payload["environment_id"] = "ENV-0001"
    payload["execution_environment"] = unavailable_xpu_execution_environment()
    payload["execution_environment"]["runtime"] = "forged"
    forged_execution_environment = canonical_json(payload)
    with pytest.raises(ValueError, match="fields"):
        PreparedNoRunEvent(
            **{
                **_prepared_fields(prepared),
                "payload_bytes": forged_execution_environment,
                "payload_digest": sha256(forged_execution_environment).hexdigest(),
            }
        )
    with pytest.raises(ValueError, match="fields"):
        PreparedNoRunEvent(
            **{**_prepared_fields(prepared), "selected_idea_id": "IDEA-0002"}
        )


def test_unavailable_environment_factory_returns_mutation_isolated_values():
    first = unavailable_xpu_execution_environment()
    first["runtime"] = "mutated"

    assert unavailable_xpu_execution_environment() == {
        "backend": "xpu",
        "device": "unavailable",
        "driver": "unavailable",
        "runtime": "unavailable",
        "framework": "PyTorch unavailable",
        "compiler": "unavailable",
        "architecture": "unavailable",
    }


def _prepared_fields(prepared: PreparedNoRunEvent) -> dict[str, object]:
    return {
        "payload_bytes": prepared.payload_bytes,
        "payload_digest": prepared.payload_digest,
        "environment_digest": prepared.environment_digest,
        "protocol_digest": prepared.protocol_digest,
        "queue_projection_id": prepared.queue_projection_id,
        "selected_idea_id": prepared.selected_idea_id,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("steps_consumed", True),
        ("steps_consumed", 0),
        ("continuation_allowed", True),
        ("continuation_allowed", 0),
    ),
)
def test_outcome_constructor_rejects_bool_and_equality_impostors(
    field: str, value: object
):
    outcome = NoRunAutoresearchController(_Inputs()).prepare(_request())

    with pytest.raises(ValueError, match="terminal"):
        AutoresearchOutcome(**{**_outcome_fields(outcome), field: value})


def _outcome_fields(outcome: AutoresearchOutcome) -> dict[str, object]:
    return {
        "state": outcome.state,
        "transitions": outcome.transitions,
        "steps_consumed": outcome.steps_consumed,
        "continuation_allowed": outcome.continuation_allowed,
        "prepared_event": outcome.prepared_event,
    }


def test_repository_inputs_are_fixed_path_read_only_and_public_protocol_uses_prepare():
    before = _archive_bytes()
    inputs = RepositoryAutoresearchInputs(ROOT)

    assert inputs.environment_bytes() == ENVIRONMENT.read_bytes()
    assert inputs.protocol_bytes() == PROTOCOL.read_bytes()
    assert inputs.queue_projection().items[0].idea_id == "IDEA-0001"
    assert tuple(inspect.signature(OptimizationController.prepare).parameters) == (
        "self",
        "request",
    )
    assert _archive_bytes() == before


def test_controller_source_cannot_import_or_invoke_execution_boundaries():
    source = (ROOT / "ratchet" / "optimization" / "controller.py").read_text()
    tree = ast.parse(source)
    forbidden_roots = {
        "ratchet.backends",
        "ratchet.dispatch",
        "ratchet.experiments.archive",
        "ratchet.experiments.workspaces",
        "ratchet.measurement",
        "ratchet.models",
        "ratchet.oracle",
    }
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not {
        name
        for name in imported
        if any(name == root or name.startswith(f"{root}.") for root in forbidden_roots)
    }
    assert "ratchet.experiments" in imported
    assert "ratchet.experiments.schema" not in imported
    assert "FileExperimentArchive" not in source
    assert "ExperimentWorkspace" not in source
