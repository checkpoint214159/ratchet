"""CPU-only integration tests for journaled immutable experiment facts."""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest

from ratchet.experiments import (
    EMPIRICAL_EVENT,
    NO_RUN_EVENT,
    SCHEMA_VERSION,
    ArchiveIntegrityError,
    FileExperimentArchive,
    validate_event_payload,
    validate_provisional_environment,
)

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "research" / "environment" / "ENV-0001.json"
ARCHIVE = ROOT / "research" / "archive"
COMMIT = "31a55293ff2bbf27ccdc4f62e7b0cb15f04eed7c"
EVALUATOR = "5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e"


def _common(digest: str, event_id: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_id": event_id,
        "experiment_id": "EXP-0001",
        "environment_id": "ENV-0001",
        "environment_artifact_digest": digest,
        "evaluator_contract_digest": EVALUATOR,
        "timestamp": "2026-08-29T01:41:39+08:00",
        "researcher": "ratchet",
        "git_commit": COMMIT,
        "branch": "main",
        "execution_environment": {
            "backend": "xpu",
            "device": "unavailable",
            "driver": "unavailable",
            "runtime": "unavailable",
            "framework": "PyTorch unavailable",
            "compiler": "unavailable",
            "architecture": "unavailable",
        },
        "benchmark_configuration": {
            "case_id": "reference-transformer",
            "batch_size": 1,
            "sequence_length": 16,
            "model_width": 64,
            "head_count": 4,
            "feed_forward_width": 256,
            "layer_count": 1,
            "causal": True,
            "padding_ratio": 0.0,
            "dtype": "float32",
            "seed": 0,
        },
        "candidate": {
            "implementation_id": None,
            "source_digest": None,
            "state": "not_generated",
        },
        "hypothesis": "Compare XPU eager and compiled baselines.",
        "motivation": "Compiler baseline is the first falsifiable candidate.",
        "literature_refs": ["ansel2024pytorch"],
        "decision": "defer",
        "decision_reason": "PyTorch is not installed",
        "artifact_digests": [],
        "paper_inclusion": True,
    }


def _no_run(digest: str, event_id: str = "EVT-000001") -> dict[str, object]:
    return _common(digest, event_id) | {
        "event_kind": NO_RUN_EVENT,
        "classification": "no_run",
        "intended_protocol": "correctness then synchronized XPU timing",
        "stop_reason": "qualified XPU runtime unavailable",
    }


def _empirical(digest: str) -> dict[str, object]:
    payload = _common(digest, "EVT-000002") | {
        "event_kind": EMPIRICAL_EVENT,
        "classification": "empirical",
        "measurement_status": "ok",
        "correctness_result": {
            "passed": True,
            "diagnostic": None,
            "atol": 1e-5,
            "rtol": 1e-5,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
        },
        "timing_result": {
            "method": "device events",
            "synchronized": True,
            "samples_ns": [100, 110],
            "statistics": {
                "min_ns": 100,
                "max_ns": 110,
                "mean_ns": 105,
                "median_ns": 105,
                "p90_ns": 110,
                "standard_error_ns": 5,
            },
            "warmup_calls": 2,
            "repetitions": 2,
            "compilation_ns": 1000,
            "first_run_ns": 200,
            "steady_state": {
                "samples_ns": [100, 110],
                "statistics": {
                    "min_ns": 100,
                    "max_ns": 110,
                    "mean_ns": 105,
                    "median_ns": 105,
                    "p90_ns": 110,
                    "standard_error_ns": 5,
                },
            },
        },
        "memory_result": {
            "peak_allocated_bytes": 1000,
            "peak_reserved_bytes": 2000,
        },
        "baseline_comparison": {
            "event_id": None,
            "speedup": None,
            "latency_ratio": None,
        },
        "current_best_comparison": {"event_id": None, "improved": False},
    }
    payload["candidate"] = {
        "implementation_id": "candidate-eager",
        "source_digest": "b" * 64,
        "state": "generated",
    }
    return payload


def _environment(archive: FileExperimentArchive) -> str:
    return archive.store_provisional_environment(ENV).digest


def _append_in_process(root: str, digest: str, event_id: str, result: object) -> None:
    try:
        FileExperimentArchive(Path(root)).append(_no_run(digest, event_id))
    except ArchiveIntegrityError as error:
        result.put(("error", str(error)))
    else:
        result.put(("ok", event_id))


def test_checked_in_environment_is_immutable_provenance_not_an_event():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    digest = manifest["artifacts"][0]
    artifact = ARCHIVE / "artifacts" / f"{digest}.json"

    validate_provisional_environment(json.loads(artifact.read_text()))

    assert manifest["events"] == []
    assert sha256(artifact.read_bytes()).hexdigest() == digest
    assert artifact.read_bytes() == ENV.read_bytes()


def test_no_run_schema_is_exact_and_projection_is_byte_identical(tmp_path: Path):
    archive = FileExperimentArchive(tmp_path / "archive")
    event = archive.append(_no_run(_environment(archive)))

    assert event.event_id.value == "EVT-000001"
    assert event.experiment_id.value == "EXP-0001"
    assert (
        archive.projection_bytes()
        == FileExperimentArchive(tmp_path / "archive").projection_bytes()
    )


@pytest.mark.parametrize(
    "field",
    [
        "timing",
        "memory",
        "correctness",
        "correctness_result",
        "timing_result",
        "memory_result",
        "baseline_comparison",
        "current_best_comparison",
        "speedup",
        "profile",
        "trace",
        "counter",
        "current_best",
    ],
)
def test_no_run_allow_list_excludes_all_result_and_profile_fields(
    tmp_path: Path, field: str
):
    archive = FileExperimentArchive(tmp_path / "archive")
    payload = _no_run(_environment(archive))
    payload[field] = "forbidden"

    with pytest.raises(ArchiveIntegrityError, match="allow-list"):
        archive.append(payload)


def test_empirical_schema_rejects_synthetic_and_unavailable_environment(tmp_path: Path):
    archive = FileExperimentArchive(tmp_path / "archive")
    payload = _empirical(_environment(archive))
    payload["classification"] = "synthetic"
    with pytest.raises(ArchiveIntegrityError, match="synthetic"):
        validate_event_payload(payload)

    with pytest.raises(ArchiveIntegrityError, match="unarchived artifact"):
        archive.append(_empirical(_environment(archive)))


def test_empirical_measurement_status_variants_preserve_unmeasured_failures():
    payload = _empirical("a" * 64)
    payload["measurement_status"] = "incorrect"
    for field in (
        "timing_result",
        "memory_result",
        "baseline_comparison",
        "current_best_comparison",
    ):
        del payload[field]
    payload["correctness_result"] = payload["correctness_result"] | {
        "passed": False,
        "diagnostic": "reference mismatch",
    }
    validate_event_payload(payload)

    for status in ("compile_error", "timeout", "crash"):
        failed = payload.copy()
        failed["measurement_status"] = status
        del failed["correctness_result"]
        validate_event_payload(failed)

    bad = payload.copy()
    bad["measurement_status"] = "incorrect"
    bad["correctness_result"] = bad["correctness_result"] | {"passed": True}
    with pytest.raises(ArchiveIntegrityError, match="must fail correctness"):
        validate_event_payload(bad)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("timing_result", "repetitions"), 1, "must match samples"),
        (("timing_result", "statistics", "mean_ns"), 999, "does not match"),
        (("correctness_result", "max_abs_error"), float("nan"), "non-negative"),
    ],
)
def test_empirical_evidence_rejects_nonfinite_or_nonreproducible_statistics(
    path: tuple[str, ...], value: object, message: str
):
    payload = _empirical("a" * 64)
    target: dict[str, object] = payload
    for name in path[:-1]:
        child = target[name]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value
    with pytest.raises(ArchiveIntegrityError, match=message):
        validate_event_payload(payload)


def test_candidate_source_digest_must_be_an_archived_artifact(tmp_path: Path):
    archive = FileExperimentArchive(tmp_path / "archive")
    payload = _empirical(_environment(archive))

    with pytest.raises(ArchiveIntegrityError, match="unarchived artifact"):
        archive.append(payload)


def test_multiple_events_can_share_one_experiment_and_concurrent_sequences_are_unique(
    tmp_path: Path,
):
    root = tmp_path / "archive"
    seed = FileExperimentArchive(root)
    digest = _environment(seed)
    with ThreadPoolExecutor(max_workers=2) as executor:
        events = list(
            executor.map(
                lambda identifier: FileExperimentArchive(root).append(
                    _no_run(digest, identifier)
                ),
                ("EVT-000001", "EVT-000002"),
            )
        )
    archive = FileExperimentArchive(root)
    archive.verify()
    assert {event.sequence for event in events} == {0, 1}
    assert {event.experiment_id.value for event in events} == {"EXP-0001"}


def test_multiprocess_append_races_preserve_a_canonical_contiguous_manifest(
    tmp_path: Path,
):
    root = tmp_path / "archive"
    digest = _environment(FileExperimentArchive(root))
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    identifiers = ("EVT-000001", "EVT-000002", "EVT-000003", "EVT-000004")
    processes = [
        context.Process(
            target=_append_in_process, args=(str(root), digest, event_id, results)
        )
        for event_id in identifiers
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=2) for _ in processes]
    assert sorted(outcomes) == [("ok", event_id) for event_id in identifiers]

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert [entry["sequence"] for entry in manifest["events"]] == [0, 1, 2, 3]
    assert manifest_path.read_bytes() == (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    FileExperimentArchive(root).verify()
    assert not (root / "transaction.json").exists()
    assert not list(root.rglob("*.tmp"))


def test_multiprocess_same_event_id_rejects_exactly_one_duplicate(tmp_path: Path):
    root = tmp_path / "archive"
    digest = _environment(FileExperimentArchive(root))
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [
        context.Process(
            target=_append_in_process, args=(str(root), digest, "EVT-000001", results)
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = [results.get(timeout=2) for _ in processes]
    assert sorted(kind for kind, _ in outcomes) == ["error", "ok"]
    assert sum("already exists" in message for _, message in outcomes) == 1
    archive = FileExperimentArchive(root)
    archive.verify()
    assert archive.projection().event_count == 1
    assert not (root / "transaction.json").exists()
    assert not list(root.rglob("*.tmp"))


def test_journal_recovers_an_interrupted_event_then_removes_the_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive = FileExperimentArchive(tmp_path / "archive")
    payload = _no_run(_environment(archive))
    original = FileExperimentArchive._atomic
    failed = False

    def interrupt(path: Path, data: bytes) -> None:
        nonlocal failed
        if path.name == "manifest.json" and not failed:
            failed = True
            raise OSError("simulated crash before manifest replacement")
        original(path, data)

    monkeypatch.setattr(FileExperimentArchive, "_atomic", staticmethod(interrupt))
    with pytest.raises(OSError, match="simulated crash"):
        archive.append(payload)
    monkeypatch.setattr(FileExperimentArchive, "_atomic", staticmethod(original))

    recovered = FileExperimentArchive(tmp_path / "archive")
    recovered.verify()
    assert recovered.projection().event_count == 1
    assert not (tmp_path / "archive" / "transaction.json").exists()


def test_journal_accepts_only_the_already_applied_manifest_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive = FileExperimentArchive(tmp_path / "archive")
    payload = _no_run(_environment(archive))
    original = FileExperimentArchive._atomic

    def interrupt_after_manifest(path: Path, data: bytes) -> None:
        original(path, data)
        if path.name == "manifest.json":
            raise OSError("simulated crash after manifest replacement")

    monkeypatch.setattr(
        FileExperimentArchive, "_atomic", staticmethod(interrupt_after_manifest)
    )
    with pytest.raises(OSError, match="after manifest"):
        archive.append(payload)
    monkeypatch.setattr(FileExperimentArchive, "_atomic", staticmethod(original))

    recovered = FileExperimentArchive(tmp_path / "archive")
    recovered.verify()
    assert recovered.projection().event_count == 1
    assert not (tmp_path / "archive" / "transaction.json").exists()


@pytest.mark.parametrize("tamper", ["before_manifest_digest", "target_digest"])
def test_journal_rejects_tampered_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
):
    archive = FileExperimentArchive(tmp_path / "archive")
    payload = _no_run(_environment(archive))
    original = FileExperimentArchive._atomic

    def interrupt(path: Path, data: bytes) -> None:
        if path.name == "manifest.json":
            raise OSError("leave journal for inspection")
        original(path, data)

    monkeypatch.setattr(FileExperimentArchive, "_atomic", staticmethod(interrupt))
    with pytest.raises(OSError, match="leave journal"):
        archive.append(payload)
    monkeypatch.setattr(FileExperimentArchive, "_atomic", staticmethod(original))
    journal = tmp_path / "archive" / "transaction.json"
    transaction = json.loads(journal.read_text())
    transaction[tamper] = "0" * 64
    journal.write_text(
        json.dumps(transaction, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(ArchiveIntegrityError, match="digest"):
        FileExperimentArchive(tmp_path / "archive").verify()


def test_journal_rejects_after_manifest_with_multiple_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive = FileExperimentArchive(tmp_path / "archive")
    payload = _no_run(_environment(archive))
    original = FileExperimentArchive._atomic

    def interrupt(path: Path, data: bytes) -> None:
        if path.name == "manifest.json":
            raise OSError("leave journal for inspection")
        original(path, data)

    monkeypatch.setattr(FileExperimentArchive, "_atomic", staticmethod(interrupt))
    with pytest.raises(OSError, match="leave journal"):
        archive.append(payload)
    monkeypatch.setattr(FileExperimentArchive, "_atomic", staticmethod(original))
    journal = tmp_path / "archive" / "transaction.json"
    transaction = json.loads(journal.read_text())
    transaction["after_manifest"]["artifacts"].append("f" * 64)
    after_bytes = (
        json.dumps(
            transaction["after_manifest"], sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n"
    )
    transaction["after_manifest_digest"] = sha256(after_bytes).hexdigest()
    journal.write_text(
        json.dumps(transaction, sort_keys=True, separators=(",", ":")) + "\n"
    )

    with pytest.raises(ArchiveIntegrityError, match="not one allowed"):
        FileExperimentArchive(tmp_path / "archive").verify()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("requested_backend", "cuda", "unavailable xpu"),
        ("backend_doctor_exit_code", 0, "nonzero"),
        ("git_commit_before_record", "bad", "40 lowercase hex"),
        (
            "device_interfaces",
            {"xpu_smi": False, "sycl_ls": False, "dev_dri": False, "dev_dxg": True},
            "interfaces",
        ),
    ],
)
def test_environment_validation_is_structural(field: str, value: object, message: str):
    environment = json.loads(ENV.read_text())
    environment[field] = value
    with pytest.raises(ArchiveIntegrityError, match=message):
        validate_provisional_environment(environment)
