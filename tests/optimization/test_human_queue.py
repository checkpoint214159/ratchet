"""CPU-only integrity tests for immutable human planning input."""

from __future__ import annotations

import ast
import json
import multiprocessing
import threading
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import pytest

import ratchet.optimization.human_queue as human_queue_module
from ratchet.optimization import (
    FileHumanResearchQueue,
    HumanInputKind,
    HumanInputSubmission,
    HumanQueueIntegrityError,
)

ROOT = Path(__file__).resolve().parents[2]
IDEA = ROOT / "research" / "ideas" / "IDEA-0001.json"
INTAKE = ROOT / "research" / "ideas" / "intake"
RECORD_FIELDS = {
    "schema_version",
    "input_id",
    "sequence",
    "recorded_at",
    "actor",
    "kind",
    "idea_id",
    "statement",
    "literature_keys",
    "priority",
    "redirect_to",
    "scope",
    "qualification_gate",
    "previous_digest",
    "record_digest",
}


def _queue(path: Path) -> FileHumanResearchQueue:
    return FileHumanResearchQueue(ROOT, path)


def _submission(
    kind: HumanInputKind,
    idea_id: str,
    statement: str = "planning input",
    *,
    literature_keys: tuple[str, ...] = (),
    priority: int | None = None,
    redirect_to: str | None = None,
) -> HumanInputSubmission:
    return HumanInputSubmission(
        recorded_at="2026-08-29T12:00:00Z",
        actor="researcher",
        kind=kind,
        idea_id=idea_id,
        statement=statement,
        literature_keys=literature_keys,
        priority=priority,
        redirect_to=redirect_to,
    )


def _idea(queue: FileHumanResearchQueue, idea_id: str) -> None:
    queue.append(_submission(HumanInputKind.IDEA, idea_id, f"question {idea_id}"))


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _refresh_digest(value: dict[str, object]) -> None:
    copied = {key: item for key, item in value.items() if key != "record_digest"}
    value["record_digest"] = sha256(_canonical(copied)).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _append_constraint_worker(root: str, intake: str, index: int) -> None:
    queue = FileHumanResearchQueue(Path(root), Path(intake))
    queue.append(
        HumanInputSubmission(
            recorded_at=f"2026-08-29T12:00:{index:02d}Z",
            actor=f"worker-{index}",
            kind=HumanInputKind.CONSTRAINT,
            idea_id="IDEA-0001",
            statement=f"constraint {index}",
        )
    )


def test_seed_is_exact_canonical_idea_provenance_and_all_public_records_are_frozen():
    queue = FileHumanResearchQueue(ROOT)
    records = queue.records()
    seed = records[0]
    source = json.loads(IDEA.read_text(encoding="utf-8"))
    raw = json.loads((INTAKE / "HRI-000001.json").read_text(encoding="utf-8"))

    assert sha256(IDEA.read_bytes()).hexdigest() == (
        "934ce6178629e5e72a98923444351c8de6f6ebd18c323acb2c4a72bf90a6938f"
    )
    assert set(raw) == RECORD_FIELDS
    assert raw["previous_digest"] == "0" * 64
    assert (seed.idea_id, seed.statement, seed.literature_keys) == (
        source["idea_id"],
        source["question"],
        tuple(source["literature_keys"]),
    )
    with pytest.raises(FrozenInstanceError):
        seed.idea_id = "IDEA-9999"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        queue.projection().items = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="submission"):
        HumanInputSubmission(  # type: ignore[arg-type]
            "2026-08-29T00:00:00Z", "actor", "idea", "IDEA-0002", "question"
        )


def test_append_writes_exact_schema_contiguous_digest_chain_and_no_mutable_head(
    tmp_path: Path,
):
    intake = tmp_path / "intake"
    queue = _queue(intake)
    first = queue.append(_submission(HumanInputKind.IDEA, "IDEA-0001"))
    second = queue.append(
        _submission(HumanInputKind.CONSTRAINT, "IDEA-0001", "do not execute")
    )

    assert (first.input_id, first.sequence, second.input_id, second.sequence) == (
        "HRI-000001",
        1,
        "HRI-000002",
        2,
    )
    assert second.previous_digest == first.record_digest
    assert not (intake / "head.json").exists()
    assert not (intake / "head").exists()
    assert {path.name for path in intake.iterdir()} == {
        ".human-input.lock",
        "HRI-000001.json",
        "HRI-000002.json",
    }
    assert set(json.loads((intake / "HRI-000002.json").read_text())) == RECORD_FIELDS


@pytest.mark.parametrize(
    "submission",
    (
        _submission(HumanInputKind.CONSTRAINT, "IDEA-0001"),
        _submission(HumanInputKind.LITERATURE, "IDEA-0001", literature_keys=()),
        _submission(HumanInputKind.PRIORITY, "IDEA-0001", priority=101),
        _submission(HumanInputKind.PRIORITY, "IDEA-0001", priority=-1),
        _submission(HumanInputKind.REDIRECT, "IDEA-0001", redirect_to="IDEA-0002"),
    ),
)
def test_kind_rules_reject_dangling_or_invalid_submissions(
    tmp_path: Path, submission: HumanInputSubmission
):
    with pytest.raises(HumanQueueIntegrityError):
        _queue(tmp_path / "intake").append(submission)


def test_literature_references_resolve_trackers_and_bibliography_without_marking_read(
    tmp_path: Path,
):
    queue = _queue(tmp_path / "intake")
    _idea(queue, "IDEA-0001")
    record = queue.append(
        _submission(
            HumanInputKind.LITERATURE,
            "IDEA-0001",
            "read later",
            literature_keys=("dao2022flashattention",),
        )
    )
    before = (ROOT / "papers_to_read.md").read_bytes()

    assert record.literature_keys == ("dao2022flashattention",)
    assert (ROOT / "papers_to_read.md").read_bytes() == before
    with pytest.raises(HumanQueueIntegrityError, match="resolve"):
        queue.append(
            _submission(
                HumanInputKind.LITERATURE,
                "IDEA-0001",
                "unknown",
                literature_keys=("not-a-key",),
            )
        )


@pytest.mark.parametrize(
    "field,value", (("scope", "execution"), ("qualification_gate", "FG-02"))
)
def test_planning_scope_gate_and_recursive_forbidden_fields_are_rejected(
    tmp_path: Path, field: str, value: object
):
    intake = tmp_path / "intake"
    queue = _queue(intake)
    _idea(queue, "IDEA-0001")
    raw = json.loads((intake / "HRI-000001.json").read_text())
    raw[field] = value
    _refresh_digest(raw)
    _write(intake / "HRI-000001.json", raw)

    with pytest.raises(HumanQueueIntegrityError, match="planning_only"):
        queue.records()
    raw[field] = "planning_only" if field == "scope" else "FG-01"
    raw["profile"] = {"trace": "forbidden"}
    _refresh_digest(raw)
    _write(intake / "HRI-000001.json", raw)
    with pytest.raises(HumanQueueIntegrityError):
        queue.records()


def test_chain_tamper_deletion_duplicate_rewrite_and_partial_files_are_rejected(
    tmp_path: Path,
):
    intake = tmp_path / "intake"
    queue = _queue(intake)
    _idea(queue, "IDEA-0001")
    queue.append(_submission(HumanInputKind.CONSTRAINT, "IDEA-0001", "constraint"))

    first_path = intake / "HRI-000001.json"
    first = json.loads(first_path.read_text())
    second_path = intake / "HRI-000002.json"
    second = json.loads(second_path.read_text())
    second["statement"] = "tampered"
    _write(second_path, second)
    with pytest.raises(HumanQueueIntegrityError, match="digest"):
        queue.records()

    _refresh_digest(second)
    _write(second_path, second)
    assert len(queue.records()) == 2
    rewritten = dict(first)
    rewritten["statement"] = "rewritten"
    _refresh_digest(rewritten)
    _write(first_path, rewritten)
    with pytest.raises(HumanQueueIntegrityError, match="chain"):
        queue.records()
    _write(first_path, first)
    first_path.unlink()
    with pytest.raises(HumanQueueIntegrityError, match="contiguous"):
        queue.records()

    _write(first_path, first)
    duplicate = dict(second)
    duplicate["input_id"] = "HRI-000002"
    _refresh_digest(duplicate)
    _write(intake / "HRI-000003.json", duplicate)
    with pytest.raises(HumanQueueIntegrityError, match="contiguous"):
        queue.records()

    (intake / "HRI-000003.json").unlink()
    (intake / ".HRI-000003.json.partial.tmp").write_text("partial")
    with pytest.raises(HumanQueueIntegrityError, match="partial"):
        queue.records()


def test_symlinked_intake_record_is_rejected_before_external_content_is_read(
    tmp_path: Path,
):
    intake = tmp_path / "intake"
    queue = _queue(intake)
    _idea(queue, "IDEA-0001")
    record_path = intake / "HRI-000001.json"
    external = tmp_path / "external-valid-record.json"
    external.write_bytes(record_path.read_bytes())
    record_path.unlink()
    record_path.symlink_to(external)

    with pytest.raises(HumanQueueIntegrityError, match="symbolic links"):
        queue.records()


def test_projection_accumulates_and_orders_latest_priority_literature_and_redirects(
    tmp_path: Path,
):
    queue = _queue(tmp_path / "intake")
    _idea(queue, "IDEA-0001")
    _idea(queue, "IDEA-0002")
    queue.append(_submission(HumanInputKind.CONSTRAINT, "IDEA-0001", "first"))
    queue.append(_submission(HumanInputKind.CONSTRAINT, "IDEA-0001", "second"))
    queue.append(
        _submission(
            HumanInputKind.LITERATURE,
            "IDEA-0001",
            "literature",
            literature_keys=("dao2022flashattention", "ansel2024pytorch"),
        )
    )
    queue.append(_submission(HumanInputKind.PRIORITY, "IDEA-0001", priority=10))
    queue.append(_submission(HumanInputKind.PRIORITY, "IDEA-0002", priority=50))

    before_redirect = queue.projection()
    assert [item.idea_id for item in before_redirect.items] == [
        "IDEA-0002",
        "IDEA-0001",
    ]
    first = before_redirect.items[1]
    assert first.constraints == ("first", "second")
    assert first.literature_keys == ("dao2022flashattention", "ansel2024pytorch")

    queue.append(
        _submission(
            HumanInputKind.REDIRECT,
            "IDEA-0002",
            "merge ideas",
            redirect_to="IDEA-0001",
        )
    )
    after_redirect = queue.projection()
    assert [item.idea_id for item in after_redirect.items] == ["IDEA-0001"]
    assert len(queue.records()) == 8
    assert before_redirect.projection_id != after_redirect.projection_id


def test_redirect_cycles_are_rejected(tmp_path: Path):
    queue = _queue(tmp_path / "intake")
    _idea(queue, "IDEA-0001")
    _idea(queue, "IDEA-0002")
    queue.append(
        _submission(HumanInputKind.REDIRECT, "IDEA-0001", redirect_to="IDEA-0002")
    )

    with pytest.raises(HumanQueueIntegrityError, match="cycle"):
        queue.append(
            _submission(HumanInputKind.REDIRECT, "IDEA-0002", redirect_to="IDEA-0001")
        )


def test_concurrent_appends_are_locked_contiguous_and_leave_no_partial_files(
    tmp_path: Path,
):
    intake = tmp_path / "intake"
    queue = _queue(intake)
    _idea(queue, "IDEA-0001")
    context = multiprocessing.get_context("fork")
    workers = [
        context.Process(
            target=_append_constraint_worker, args=(str(ROOT), str(intake), index)
        )
        for index in range(1, 5)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0

    records = queue.records()
    assert [record.input_id for record in records] == [
        "HRI-000001",
        "HRI-000002",
        "HRI-000003",
        "HRI-000004",
        "HRI-000005",
    ]
    assert not list(intake.glob("*.tmp"))


def test_reader_waits_for_locked_atomic_append_then_observes_complete_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    intake = tmp_path / "intake"
    queue = _queue(intake)
    _idea(queue, "IDEA-0001")
    replace_entered = threading.Event()
    release_replace = threading.Event()
    reader_requested_lock = threading.Event()
    reader_finished = threading.Event()
    original_replace = human_queue_module.os.replace
    original_flock = human_queue_module.fcntl.flock
    reader_result: list[tuple[object, ...]] = []
    reader_error: list[BaseException] = []

    def paused_replace(source: object, destination: object) -> None:
        replace_entered.set()
        assert release_replace.wait(timeout=5)
        original_replace(source, destination)

    def observed_flock(lock: object, operation: int) -> None:
        if operation == human_queue_module.fcntl.LOCK_SH:
            reader_requested_lock.set()
        original_flock(lock, operation)

    monkeypatch.setattr(human_queue_module.os, "replace", paused_replace)
    monkeypatch.setattr(human_queue_module.fcntl, "flock", observed_flock)
    writer = threading.Thread(
        target=lambda: queue.append(
            _submission(HumanInputKind.CONSTRAINT, "IDEA-0001", "locked append")
        )
    )

    def read_records() -> None:
        try:
            reader_result.append(queue.records())
        except BaseException as error:  # pragma: no cover - asserted below
            reader_error.append(error)
        finally:
            reader_finished.set()

    writer.start()
    assert replace_entered.wait(timeout=5)
    reader = threading.Thread(target=read_records)
    reader.start()
    assert reader_requested_lock.wait(timeout=5)
    assert not reader_finished.is_set()
    release_replace.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert not reader_error
    assert len(reader_result) == 1
    assert [record.input_id for record in reader_result[0]] == [
        "HRI-000001",
        "HRI-000002",
    ]


def test_queue_is_planning_only_and_has_no_execution_or_cross_context_imports():
    source = ROOT / "ratchet" / "optimization" / "human_queue.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

    assert not {name for name in imports if name.startswith("ratchet")}
    assert (
        not {
            "OptimizationRequest",
            "OptimizationController",
            "FileExperimentArchive",
            "torch",
            "triton",
            "BackendIdentity",
        }
        & names
    )
