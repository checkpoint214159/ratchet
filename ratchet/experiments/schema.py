"""Exact archive evidence schemas and immutable public records."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Protocol

SCHEMA_VERSION = 1
EMPIRICAL_EVENT = "empirical"
NO_RUN_EVENT = "no_run"
SHA = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
EVENT = re.compile(r"^EVT-[0-9]{6}$")
EXPERIMENT = re.compile(r"^EXP-[0-9]{4}$")
ENVIRONMENT = re.compile(r"^ENV-[0-9]{4}$")
EVALUATOR_CONTRACT_DIGEST = (
    "5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e"
)


def unavailable_xpu_execution_environment() -> dict[str, str]:
    """Return the fixed unavailable-XPU provenance without sharing mutable state."""
    return {
        "backend": "xpu",
        "device": "unavailable",
        "driver": "unavailable",
        "runtime": "unavailable",
        "framework": "PyTorch unavailable",
        "compiler": "unavailable",
        "architecture": "unavailable",
    }


COMMON = frozenset(
    {
        "schema_version",
        "event_kind",
        "classification",
        "event_id",
        "experiment_id",
        "environment_id",
        "environment_artifact_digest",
        "evaluator_contract_digest",
        "timestamp",
        "researcher",
        "git_commit",
        "branch",
        "execution_environment",
        "benchmark_configuration",
        "candidate",
        "hypothesis",
        "motivation",
        "literature_refs",
        "decision",
        "decision_reason",
        "artifact_digests",
        "paper_inclusion",
    }
)
EMPIRICAL_RESULT_FIELDS = frozenset(
    {
        "correctness_result",
        "timing_result",
        "memory_result",
        "baseline_comparison",
        "current_best_comparison",
    }
)
EMPIRICAL_OK_FIELDS = (
    COMMON | frozenset({"measurement_status"}) | EMPIRICAL_RESULT_FIELDS
)
EMPIRICAL_INCORRECT_FIELDS = COMMON | frozenset(
    {"measurement_status", "correctness_result"}
)
EMPIRICAL_FAILED_FIELDS = COMMON | frozenset({"measurement_status"})
NO_RUN_FIELDS = COMMON | frozenset({"intended_protocol", "stop_reason"})
ENV_FIELDS = frozenset(
    {
        "schema_version",
        "record_kind",
        "environment_id",
        "observed_at",
        "git_commit_before_record",
        "host",
        "requested_backend",
        "availability",
        "backend_doctor_exit_code",
        "backend_doctor_reason",
        "device_interfaces",
        "probe_semantics",
        "empirical_work_permitted",
        "decision",
        "notes",
    }
)


class ArchiveIntegrityError(ValueError):
    """Raised for invalid, partial, unprovenanced, or altered archive state."""


def canonical_json(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ArchiveIntegrityError(
            "archive JSON must not contain non-finite values"
        ) from error
    return (encoded + "\n").encode()


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ArchiveIntegrityError(f"field {name} must be a non-empty string")
    return value


def digest_field(payload: Mapping[str, object], name: str) -> str:
    value = string(payload, name)
    if not SHA.fullmatch(value):
        raise ArchiveIntegrityError(f"field {name} must be a SHA-256 hex digest")
    return value


def exact(value: Mapping[str, object], fields: frozenset[str], name: str) -> None:
    if set(value) != fields:
        raise ArchiveIntegrityError(f"{name} schema fields do not match its allow-list")


def strings(payload: Mapping[str, object], name: str) -> None:
    value = payload.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ArchiveIntegrityError(f"field {name} must be a string list")


def mapping(value: object, fields: frozenset[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ArchiveIntegrityError(f"field {name} must be an object")
    exact(value, fields, name)
    return value


def number(value: object, name: str, *, positive: bool = False) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ArchiveIntegrityError(
            f"field {name} must be a {'positive' if positive else 'non-negative'} number"
        )


def nonnegative_int(value: object, name: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ArchiveIntegrityError(f"field {name} must be a non-negative integer")


def _environment(value: object) -> None:
    item = mapping(
        value,
        frozenset(
            {
                "backend",
                "device",
                "driver",
                "runtime",
                "framework",
                "compiler",
                "architecture",
            }
        ),
        "execution_environment",
    )
    for name in item:
        string(item, name)


def _configuration(value: object) -> None:
    item = mapping(
        value,
        frozenset(
            {
                "case_id",
                "batch_size",
                "sequence_length",
                "model_width",
                "head_count",
                "feed_forward_width",
                "layer_count",
                "causal",
                "padding_ratio",
                "dtype",
                "seed",
            }
        ),
        "benchmark_configuration",
    )
    for name in (
        "batch_size",
        "sequence_length",
        "model_width",
        "head_count",
        "feed_forward_width",
        "layer_count",
    ):
        nonnegative_int(item[name], f"benchmark_configuration.{name}")
        if item[name] == 0:
            raise ArchiveIntegrityError(
                f"benchmark_configuration.{name} must be positive"
            )
    nonnegative_int(item["seed"], "benchmark_configuration.seed")
    number(item["padding_ratio"], "benchmark_configuration.padding_ratio")
    if item["padding_ratio"] > 1:
        raise ArchiveIntegrityError(
            "benchmark_configuration.padding_ratio must be at most 1"
        )
    if not isinstance(item["causal"], bool):
        raise ArchiveIntegrityError("benchmark_configuration.causal must be a boolean")
    for name in ("case_id", "dtype"):
        string(item, name)


def _candidate(value: object, *, generated: bool) -> None:
    item = mapping(
        value, frozenset({"implementation_id", "source_digest", "state"}), "candidate"
    )
    if generated:
        string(item, "implementation_id")
        digest_field(item, "source_digest")
        if item["state"] != "generated":
            raise ArchiveIntegrityError("empirical candidate must be generated")
    elif item != {
        "implementation_id": None,
        "source_digest": None,
        "state": "not_generated",
    }:
        raise ArchiveIntegrityError("no-run candidate must be explicitly not generated")


def _correctness(value: object) -> None:
    item = mapping(
        value,
        frozenset(
            {"passed", "diagnostic", "atol", "rtol", "max_abs_error", "max_rel_error"}
        ),
        "correctness_result",
    )
    if not isinstance(item["passed"], bool):
        raise ArchiveIntegrityError("correctness_result.passed must be a boolean")
    if item["diagnostic"] is not None and (
        not isinstance(item["diagnostic"], str) or not item["diagnostic"]
    ):
        raise ArchiveIntegrityError(
            "correctness_result.diagnostic must be null or text"
        )
    if not item["passed"] and item["diagnostic"] is None:
        raise ArchiveIntegrityError("failed correctness requires a diagnostic")
    for name in ("atol", "rtol", "max_abs_error", "max_rel_error"):
        number(item[name], f"correctness_result.{name}")


def _statistics(value: object, name: str, samples: list[object]) -> None:
    item = mapping(
        value,
        frozenset(
            {"min_ns", "max_ns", "mean_ns", "median_ns", "p90_ns", "standard_error_ns"}
        ),
        name,
    )
    for field, statistic in item.items():
        number(statistic, f"{name}.{field}")
    numeric_samples = [int(sample) for sample in samples]
    ordered = sorted(numeric_samples)
    expected = {
        "min_ns": min(numeric_samples),
        "max_ns": max(numeric_samples),
        "mean_ns": statistics.mean(numeric_samples),
        "median_ns": statistics.median(numeric_samples),
        "p90_ns": ordered[math.ceil(len(ordered) * 0.9) - 1],
        "standard_error_ns": (
            statistics.stdev(numeric_samples) / math.sqrt(len(numeric_samples))
            if len(numeric_samples) > 1
            else 0.0
        ),
    }
    if any(
        not math.isclose(item[field], expected[field], rel_tol=1e-12, abs_tol=1e-12)
        for field in expected
    ):
        raise ArchiveIntegrityError(f"{name} does not match timing samples")


def _timing(value: object) -> None:
    item = mapping(
        value,
        frozenset(
            {
                "method",
                "synchronized",
                "samples_ns",
                "statistics",
                "warmup_calls",
                "repetitions",
                "compilation_ns",
                "first_run_ns",
                "steady_state",
            }
        ),
        "timing_result",
    )
    string(item, "method")
    if item["synchronized"] is not True:
        raise ArchiveIntegrityError("timing_result must record synchronized timing")
    for name in ("warmup_calls", "repetitions"):
        nonnegative_int(item[name], f"timing_result.{name}")
    if item["repetitions"] == 0:
        raise ArchiveIntegrityError("timing_result.repetitions must be positive")
    for name in ("compilation_ns", "first_run_ns"):
        nonnegative_int(item[name], f"timing_result.{name}", allow_none=True)
    samples = item["samples_ns"]
    if not isinstance(samples, list) or not samples:
        raise ArchiveIntegrityError("timing_result.samples_ns must be a non-empty list")
    for sample in samples:
        nonnegative_int(sample, "timing_result.samples_ns")
        if sample == 0:
            raise ArchiveIntegrityError("timing_result.samples_ns must be positive")
    if item["repetitions"] != len(samples):
        raise ArchiveIntegrityError("timing_result.repetitions must match samples")
    _statistics(item["statistics"], "timing_result.statistics", samples)
    steady = mapping(
        item["steady_state"],
        frozenset({"samples_ns", "statistics"}),
        "timing_result.steady_state",
    )
    if steady["samples_ns"] != samples:
        raise ArchiveIntegrityError("steady-state samples must match timing samples")
    _statistics(steady["statistics"], "timing_result.steady_state.statistics", samples)


def _memory(value: object) -> None:
    item = mapping(
        value,
        frozenset({"peak_allocated_bytes", "peak_reserved_bytes"}),
        "memory_result",
    )
    for name, value in item.items():
        nonnegative_int(value, f"memory_result.{name}")


def _comparison(value: object, name: str, *, current_best: bool) -> None:
    item = mapping(
        value,
        frozenset({"event_id", "improved"})
        if current_best
        else frozenset({"event_id", "speedup", "latency_ratio"}),
        name,
    )
    event_id = item["event_id"]
    if event_id is not None and (
        not isinstance(event_id, str) or not EVENT.fullmatch(event_id)
    ):
        raise ArchiveIntegrityError(f"{name}.event_id must be null or an event id")
    if current_best:
        if not isinstance(item["improved"], bool):
            raise ArchiveIntegrityError(f"{name}.improved must be a boolean")
        return
    for field in ("speedup", "latency_ratio"):
        if event_id is None and item[field] is not None:
            raise ArchiveIntegrityError(f"{name}.{field} requires a baseline event")
        if item[field] is not None:
            number(item[field], f"{name}.{field}", positive=True)


def validate_event_payload(payload: Mapping[str, object]) -> None:
    """Validate exact, versioned empirical and no-run event schemas."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ArchiveIntegrityError("event schema_version is unsupported")
    kind = payload.get("event_kind")
    if kind == EMPIRICAL_EVENT:
        if payload.get("classification") != "empirical":
            raise ArchiveIntegrityError(
                "empirical events cannot use synthetic classification"
            )
        status = payload.get("measurement_status")
        if status == "ok":
            exact(payload, EMPIRICAL_OK_FIELDS, "empirical ok")
            fields = EMPIRICAL_OK_FIELDS
            _correctness(payload["correctness_result"])
            if payload["correctness_result"]["passed"] is not True:
                raise ArchiveIntegrityError("ok empirical event must pass correctness")
            _timing(payload["timing_result"])
            _memory(payload["memory_result"])
            _comparison(
                payload["baseline_comparison"],
                "baseline_comparison",
                current_best=False,
            )
            _comparison(
                payload["current_best_comparison"],
                "current_best_comparison",
                current_best=True,
            )
        elif status == "incorrect":
            exact(payload, EMPIRICAL_INCORRECT_FIELDS, "empirical incorrect")
            fields = EMPIRICAL_INCORRECT_FIELDS
            _correctness(payload["correctness_result"])
            if payload["correctness_result"]["passed"] is not False:
                raise ArchiveIntegrityError(
                    "incorrect empirical event must fail correctness"
                )
        elif status in {"compile_error", "timeout", "crash"}:
            exact(payload, EMPIRICAL_FAILED_FIELDS, f"empirical {status}")
            fields = EMPIRICAL_FAILED_FIELDS
        else:
            raise ArchiveIntegrityError("empirical measurement_status is unsupported")
        required = fields - {
            "schema_version",
            "event_kind",
            "classification",
            "measurement_status",
            "artifact_digests",
            "literature_refs",
            "paper_inclusion",
            "evaluator_contract_digest",
            "execution_environment",
            "benchmark_configuration",
            "candidate",
            "correctness_result",
            "timing_result",
            "memory_result",
            "baseline_comparison",
            "current_best_comparison",
        }
        _candidate(payload["candidate"], generated=True)
    elif kind == NO_RUN_EVENT:
        exact(payload, NO_RUN_FIELDS, "no-run")
        if payload.get("classification") != "no_run":
            raise ArchiveIntegrityError("no-run events must use no_run classification")
        required = NO_RUN_FIELDS - {
            "schema_version",
            "event_kind",
            "classification",
            "artifact_digests",
            "literature_refs",
            "paper_inclusion",
            "evaluator_contract_digest",
            "execution_environment",
            "benchmark_configuration",
            "candidate",
        }
        _candidate(payload["candidate"], generated=False)
    else:
        raise ArchiveIntegrityError("event_kind must be empirical or no_run")
    for name in required:
        string(payload, name)
    _environment(payload["execution_environment"])
    _configuration(payload["benchmark_configuration"])
    if (
        not EVENT.fullmatch(string(payload, "event_id"))
        or not EXPERIMENT.fullmatch(string(payload, "experiment_id"))
        or not ENVIRONMENT.fullmatch(string(payload, "environment_id"))
    ):
        raise ArchiveIntegrityError("event identifiers are invalid")
    if not COMMIT.fullmatch(string(payload, "git_commit")):
        raise ArchiveIntegrityError("event git_commit must be 40 lowercase hex")
    digest_field(payload, "environment_artifact_digest")
    if digest_field(payload, "evaluator_contract_digest") != EVALUATOR_CONTRACT_DIGEST:
        raise ArchiveIntegrityError(
            "event evaluator contract digest is not the reference benchmark"
        )
    strings(payload, "literature_refs")
    strings(payload, "artifact_digests")
    if not all(SHA.fullmatch(value) for value in payload["artifact_digests"]):
        raise ArchiveIntegrityError("artifact_digests must be SHA-256 hex digests")
    if not isinstance(payload.get("paper_inclusion"), bool):
        raise ArchiveIntegrityError("paper_inclusion must be a boolean")


def validate_provisional_environment(payload: Mapping[str, object]) -> None:
    """Validate the exact unavailable-XPU provenance record."""
    exact(payload, ENV_FIELDS, "environment")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("record_kind") != "provisional_environment_observation"
    ):
        raise ArchiveIntegrityError("environment schema or record kind is unsupported")
    if not ENVIRONMENT.fullmatch(string(payload, "environment_id")):
        raise ArchiveIntegrityError("environment id must match ENV-NNNN")
    if (
        string(payload, "requested_backend") != "xpu"
        or string(payload, "availability") != "unavailable"
    ):
        raise ArchiveIntegrityError("environment must record unavailable xpu")
    code = payload.get("backend_doctor_exit_code")
    if not isinstance(code, int) or isinstance(code, bool) or code == 0:
        raise ArchiveIntegrityError("environment backend doctor must have nonzero exit")
    if not COMMIT.fullmatch(string(payload, "git_commit_before_record")):
        raise ArchiveIntegrityError("environment git commit must be 40 lowercase hex")
    if (
        payload.get("empirical_work_permitted") is not False
        or payload.get("decision") != "literature_only"
    ):
        raise ArchiveIntegrityError("environment must forbid empirical work")
    host = payload.get("host")
    if (
        not isinstance(host, dict)
        or set(host) != {"kernel", "system_python", "project_python"}
        or not all(isinstance(value, str) and value for value in host.values())
    ):
        raise ArchiveIntegrityError("environment host fields are invalid")
    if payload.get("device_interfaces") != {
        "xpu_smi": False,
        "sycl_ls": False,
        "dev_dri": False,
        "dev_dxg": False,
    }:
        raise ArchiveIntegrityError("environment device interfaces must be false")
    probes = payload.get("probe_semantics")
    if (
        not isinstance(probes, dict)
        or probes.get("backend_doctor")
        != ".venv/bin/python -m ratchet.backends --backend xpu"
        or probes.get("command_presence")
        != ["command -v xpu-smi", "command -v sycl-ls"]
        or probes.get("device_presence") != ["test -e /dev/dri", "test -e /dev/dxg"]
    ):
        raise ArchiveIntegrityError("environment probes are invalid")
    for name in ("observed_at", "backend_doctor_reason"):
        string(payload, name)
    strings(payload, "notes")


@dataclass(frozen=True, slots=True)
class EventId:
    value: str

    def __post_init__(self) -> None:
        if not EVENT.fullmatch(self.value):
            raise ValueError("event id must match EVT-NNNNNN")


@dataclass(frozen=True, slots=True)
class ExperimentId:
    value: str

    def __post_init__(self) -> None:
        if not EXPERIMENT.fullmatch(self.value):
            raise ValueError("experiment id must match EXP-NNNN")


@dataclass(frozen=True, slots=True)
class EnvironmentId:
    value: str

    def __post_init__(self) -> None:
        if not ENVIRONMENT.fullmatch(self.value):
            raise ValueError("environment id must match ENV-NNNN")


@dataclass(frozen=True, slots=True)
class ExperimentEvent:
    event_id: EventId
    experiment_id: ExperimentId
    sequence: int
    kind: str
    payload_digest: str

    def __post_init__(self) -> None:
        if (
            self.sequence < 0
            or self.kind not in {EMPIRICAL_EVENT, NO_RUN_EVENT}
            or not SHA.fullmatch(self.payload_digest)
        ):
            raise ValueError("invalid experiment event")


@dataclass(frozen=True, slots=True)
class EnvironmentArtifact:
    environment_id: EnvironmentId
    digest: str

    def __post_init__(self) -> None:
        if not SHA.fullmatch(self.digest):
            raise ValueError("environment digest must be SHA-256")


@dataclass(frozen=True, slots=True)
class CatalogueProjection:
    projection_id: str
    event_count: int
    event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.event_ids, tuple)
            or not SHA.fullmatch(self.projection_id)
            or self.event_count != len(self.event_ids)
            or len(set(self.event_ids)) != len(self.event_ids)
        ):
            raise ValueError("invalid projection")
        for event_id in self.event_ids:
            EventId(event_id)


class ExperimentCatalogue(Protocol):
    def append(self, payload: Mapping[str, object]) -> ExperimentEvent: ...
    def projection(self) -> CatalogueProjection: ...
