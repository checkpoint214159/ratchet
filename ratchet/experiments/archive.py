"""Locked file storage and recoverable transactions for experiment evidence."""

from __future__ import annotations

import base64
import binascii
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterator, Mapping

from .schema import (
    EMPIRICAL_EVENT,
    EVALUATOR_CONTRACT_DIGEST,
    EVENT,
    EXPERIMENT,
    NO_RUN_EVENT,
    SCHEMA_VERSION,
    SHA,
    ArchiveIntegrityError,
    CatalogueProjection,
    EnvironmentArtifact,
    EnvironmentId,
    EventId,
    ExperimentEvent,
    ExperimentId,
    canonical_json,
    digest,
    digest_field,
    string,
    unavailable_xpu_execution_environment,
    validate_event_payload,
    validate_provisional_environment,
)


class FileExperimentArchive:
    """An fsync'd journal recovers each event/artifact plus manifest transaction."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._events = root / "events"
        self._artifacts = root / "artifacts"
        self._manifest = root / "manifest.json"
        self._journal = root / "transaction.json"
        self._lock = root / ".archive.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._root.mkdir(parents=True, exist_ok=True)
        with self._lock.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            FileExperimentArchive._fsync_directory(path.parent)
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _manifest_data(self) -> dict[str, object]:
        if not self._manifest.exists():
            return {"schema_version": SCHEMA_VERSION, "events": [], "artifacts": []}
        raw = self._manifest.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ArchiveIntegrityError(
                "archive manifest is partial or invalid"
            ) from error
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "events", "artifacts"}
            or value.get("schema_version") != SCHEMA_VERSION
            or not isinstance(value["events"], list)
            or not isinstance(value["artifacts"], list)
            or raw != canonical_json(value)
        ):
            raise ArchiveIntegrityError("archive manifest is invalid or mutated")
        return value

    @staticmethod
    def _target(operation: str, value: object) -> PurePosixPath:
        if not isinstance(value, str):
            raise ArchiveIntegrityError("archive transaction target path is invalid")
        path = PurePosixPath(value)
        if (
            len(path.parts) != 2
            or path.is_absolute()
            or any(part in {".", ".."} for part in path.parts)
        ):
            raise ArchiveIntegrityError("archive transaction target path is invalid")
        directory, filename = path.parts
        valid = (
            operation == "append_event"
            and directory == "events"
            and filename.endswith(".json")
            and EVENT.fullmatch(filename[:-5])
        )
        valid = (
            valid
            or operation == "store_artifact"
            and directory == "artifacts"
            and filename.endswith(".json")
            and SHA.fullmatch(filename[:-5])
        )
        if not valid:
            raise ArchiveIntegrityError("archive transaction target path is invalid")
        return path

    @staticmethod
    def _manifest_shape(value: object) -> dict[str, object]:
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "events", "artifacts"}
            or value.get("schema_version") != SCHEMA_VERSION
            or not isinstance(value["events"], list)
            or not isinstance(value["artifacts"], list)
        ):
            raise ArchiveIntegrityError("archive transaction manifest is invalid")
        return value

    def _transition(
        self,
        operation: str,
        target: PurePosixPath,
        data: bytes,
        before: object,
        after: object,
    ) -> None:
        prior, next_manifest = self._manifest_shape(before), self._manifest_shape(after)
        prior_events, prior_artifacts = prior["events"], prior["artifacts"]
        events, artifacts = next_manifest["events"], next_manifest["artifacts"]
        assert (
            isinstance(prior_events, list)
            and isinstance(prior_artifacts, list)
            and isinstance(events, list)
            and isinstance(artifacts, list)
        )
        if operation == "store_artifact":
            target_digest = digest(data)
            if target.as_posix() != f"artifacts/{target_digest}.json":
                raise ArchiveIntegrityError(
                    "transaction artifact target does not match bytes"
                )
            if events != prior_events or artifacts != [*prior_artifacts, target_digest]:
                raise ArchiveIntegrityError(
                    "transaction after-manifest is not one allowed artifact store"
                )
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as error:
            raise ArchiveIntegrityError(
                "transaction event target is invalid"
            ) from error
        if not isinstance(payload, dict) or data != canonical_json(payload):
            raise ArchiveIntegrityError("transaction event target is invalid")
        validate_event_payload(payload)
        event_id = string(payload, "event_id")
        if target.as_posix() != f"events/{event_id}.json":
            raise ArchiveIntegrityError(
                "transaction event target does not match payload"
            )
        entry = {
            "event_id": event_id,
            "experiment_id": string(payload, "experiment_id"),
            "sequence": len(prior_events),
            "event_kind": string(payload, "event_kind"),
            "payload_digest": digest(data),
        }
        if artifacts != prior_artifacts or events != [*prior_events, entry]:
            raise ArchiveIntegrityError(
                "transaction after-manifest is not one allowed event append"
            )

    def _read_transaction(
        self,
    ) -> tuple[PurePosixPath, bytes, Mapping[str, object], Mapping[str, object], str]:
        raw = self._journal.read_bytes()
        try:
            transaction = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ArchiveIntegrityError("archive transaction is partial") from error
        fields = {
            "schema_version",
            "operation",
            "target_path",
            "target_data",
            "target_digest",
            "before_manifest",
            "before_manifest_digest",
            "after_manifest",
            "after_manifest_digest",
        }
        if (
            not isinstance(transaction, dict)
            or set(transaction) != fields
            or transaction.get("schema_version") != SCHEMA_VERSION
            or transaction.get("operation") not in {"append_event", "store_artifact"}
            or not isinstance(transaction.get("target_data"), str)
            or not isinstance(transaction.get("before_manifest"), dict)
            or not isinstance(transaction.get("after_manifest"), dict)
            or raw != canonical_json(transaction)
        ):
            raise ArchiveIntegrityError("archive transaction is invalid")
        for field in (
            "target_digest",
            "before_manifest_digest",
            "after_manifest_digest",
        ):
            digest_field(transaction, field)
        try:
            data = base64.b64decode(transaction["target_data"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise ArchiveIntegrityError(
                "archive transaction target data is invalid"
            ) from error
        if digest(data) != transaction["target_digest"]:
            raise ArchiveIntegrityError(
                "archive transaction target digest does not match bytes"
            )
        target = self._target(transaction["operation"], transaction["target_path"])
        before, after = (
            self._manifest_shape(transaction["before_manifest"]),
            self._manifest_shape(transaction["after_manifest"]),
        )
        if digest(canonical_json(before)) != transaction["before_manifest_digest"]:
            raise ArchiveIntegrityError(
                "archive transaction before manifest digest is invalid"
            )
        if digest(canonical_json(after)) != transaction["after_manifest_digest"]:
            raise ArchiveIntegrityError(
                "archive transaction after manifest digest is invalid"
            )
        self._transition(transaction["operation"], target, data, before, after)
        return (
            target,
            data,
            before,
            after,
            string(transaction, "before_manifest_digest"),
        )

    def _recover_locked(self) -> None:
        if not self._journal.exists():
            return
        current = self._manifest_data()
        target, data, before, after, before_digest = self._read_transaction()
        current_digest, after_digest = (
            digest(canonical_json(current)),
            digest(canonical_json(after)),
        )
        path = self._root.joinpath(*target.parts)
        if current_digest not in {before_digest, after_digest}:
            raise ArchiveIntegrityError(
                "archive transaction manifest digest does not match"
            )
        if current_digest == after_digest:
            if not path.is_file() or path.read_bytes() != data:
                raise ArchiveIntegrityError(
                    "applied transaction target was mutated or missing"
                )
        else:
            if path.exists() and path.read_bytes() != data:
                raise ArchiveIntegrityError("archive transaction target was mutated")
            if not path.exists():
                self._atomic(path, data)
            self._atomic(self._manifest, canonical_json(after))
        if self._manifest_data() != after:
            raise ArchiveIntegrityError(
                "archive transaction did not reach its applied manifest"
            )
        self._verify_state(after)
        self._journal.unlink()
        self._fsync_directory(self._root)

    def _transaction(
        self, operation: str, path: Path, data: bytes, after: Mapping[str, object]
    ) -> None:
        before = self._manifest_data()
        target = self._target(operation, path.relative_to(self._root).as_posix())
        self._transition(operation, target, data, before, after)
        transaction = {
            "schema_version": SCHEMA_VERSION,
            "operation": operation,
            "target_path": target.as_posix(),
            "target_data": base64.b64encode(data).decode(),
            "target_digest": digest(data),
            "before_manifest": before,
            "before_manifest_digest": digest(canonical_json(before)),
            "after_manifest": dict(after),
            "after_manifest_digest": digest(canonical_json(after)),
        }
        self._atomic(self._journal, canonical_json(transaction))
        self._recover_locked()

    def _verify_state(self, manifest: Mapping[str, object]) -> None:
        events, artifacts = manifest["events"], manifest["artifacts"]
        assert isinstance(events, list) and isinstance(artifacts, list)
        if len(set(artifacts)) != len(artifacts) or not all(
            isinstance(value, str) and SHA.fullmatch(value) for value in artifacts
        ):
            raise ArchiveIntegrityError("artifact digest manifest is invalid")
        actual_artifacts = (
            {path.name for path in self._artifacts.iterdir()}
            if self._artifacts.exists()
            else set()
        )
        if actual_artifacts != {f"{value}.json" for value in artifacts}:
            raise ArchiveIntegrityError(
                "archive has unexpected or missing artifact files"
            )
        for value in artifacts:
            if digest((self._artifacts / f"{value}.json").read_bytes()) != value:
                raise ArchiveIntegrityError("archive artifact bytes were mutated")
        names, seen = set(), set()
        for sequence, entry in enumerate(events):
            if not isinstance(entry, dict) or set(entry) != {
                "event_id",
                "experiment_id",
                "sequence",
                "event_kind",
                "payload_digest",
            }:
                raise ArchiveIntegrityError("archive manifest event is invalid")
            event_id, experiment_id, payload_digest = (
                entry.get("event_id"),
                entry.get("experiment_id"),
                entry.get("payload_digest"),
            )
            if (
                not isinstance(event_id, str)
                or not EVENT.fullmatch(event_id)
                or event_id in seen
                or not isinstance(experiment_id, str)
                or not EXPERIMENT.fullmatch(experiment_id)
                or entry.get("sequence") != sequence
                or entry.get("event_kind") not in {EMPIRICAL_EVENT, NO_RUN_EVENT}
                or not isinstance(payload_digest, str)
                or not SHA.fullmatch(payload_digest)
            ):
                raise ArchiveIntegrityError("archive manifest event is invalid")
            path = self._events / f"{event_id}.json"
            if not path.is_file() or digest(path.read_bytes()) != payload_digest:
                raise ArchiveIntegrityError("archive event bytes were mutated")
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError as error:
                raise ArchiveIntegrityError("archive event is partial") from error
            if not isinstance(payload, dict) or path.read_bytes() != canonical_json(
                payload
            ):
                raise ArchiveIntegrityError("archive event bytes were mutated")
            validate_event_payload(payload)
            if (
                payload.get("event_id") != event_id
                or payload.get("experiment_id") != experiment_id
                or payload.get("event_kind") != entry.get("event_kind")
            ):
                raise ArchiveIntegrityError(
                    "archive event manifest does not match payload"
                )
            self._provenance(manifest, payload)
            names.add(path.name)
            seen.add(event_id)
        actual_events = (
            {path.name for path in self._events.iterdir()}
            if self._events.exists()
            else set()
        )
        if actual_events != names:
            raise ArchiveIntegrityError("archive has unexpected or missing event files")

    def _provenance(
        self, manifest: Mapping[str, object], payload: Mapping[str, object]
    ) -> None:
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, list)
        environment_digest = digest_field(payload, "environment_artifact_digest")
        candidate = payload.get("candidate")
        candidate_source_digest = (
            candidate.get("source_digest") if isinstance(candidate, dict) else None
        )
        if candidate_source_digest is not None and not isinstance(
            candidate_source_digest, str
        ):
            raise ArchiveIntegrityError("candidate source digest is invalid")
        if any(
            value not in artifacts
            for value in [
                environment_digest,
                *payload.get("artifact_digests", []),
                *([candidate_source_digest] if candidate_source_digest else []),
            ]
        ):
            raise ArchiveIntegrityError("event references an unarchived artifact")
        try:
            environment = json.loads(
                (self._artifacts / f"{environment_digest}.json").read_text()
            )
        except json.JSONDecodeError as error:
            raise ArchiveIntegrityError("environment provenance is invalid") from error
        if not isinstance(environment, dict):
            raise ArchiveIntegrityError("environment provenance is invalid")
        validate_provisional_environment(environment)
        if environment.get("environment_id") != payload.get("environment_id"):
            raise ArchiveIntegrityError("environment provenance does not match")
        if (
            payload.get("execution_environment")
            != unavailable_xpu_execution_environment()
        ):
            raise ArchiveIntegrityError(
                "event execution environment does not match ENV-0001 provenance"
            )
        if payload.get("evaluator_contract_digest") != EVALUATOR_CONTRACT_DIGEST:
            raise ArchiveIntegrityError("event evaluator contract digest is invalid")
        self._comparison_provenance(manifest, payload)
        if payload.get("event_kind") == EMPIRICAL_EVENT and (
            environment.get("empirical_work_permitted") is False
            or environment.get("decision") == "literature_only"
        ):
            raise ArchiveIntegrityError(
                "environment provenance forbids empirical events"
            )

    def _comparison_provenance(
        self, manifest: Mapping[str, object], payload: Mapping[str, object]
    ) -> None:
        if payload.get("event_kind") != EMPIRICAL_EVENT:
            return
        events = manifest["events"]
        assert isinstance(events, list)
        references = []
        for field in ("baseline_comparison", "current_best_comparison"):
            comparison = payload.get(field)
            if isinstance(comparison, dict) and comparison.get("event_id") is not None:
                references.append(comparison["event_id"])
        for event_id in references:
            entry = next(
                (
                    item
                    for item in events
                    if isinstance(item, dict) and item.get("event_id") == event_id
                ),
                None,
            )
            if entry is None:
                raise ArchiveIntegrityError(
                    "comparison references an unknown prior event"
                )
            current_entry = next(
                (
                    item
                    for item in events
                    if isinstance(item, dict)
                    and item.get("event_id") == payload.get("event_id")
                ),
                None,
            )
            if isinstance(current_entry, dict) and entry.get(
                "sequence"
            ) >= current_entry.get("sequence"):
                raise ArchiveIntegrityError("comparison must reference a prior event")
            reference_path = self._events / f"{event_id}.json"
            reference = json.loads(reference_path.read_text())
            if (
                not isinstance(reference, dict)
                or reference.get("event_kind") != EMPIRICAL_EVENT
                or reference.get("measurement_status") != "ok"
                or any(
                    reference.get(field) != payload.get(field)
                    for field in (
                        "evaluator_contract_digest",
                        "benchmark_configuration",
                        "execution_environment",
                    )
                )
            ):
                raise ArchiveIntegrityError(
                    "comparison reference does not match evaluator configuration"
                )

    def _verify_locked(self) -> None:
        self._recover_locked()
        self._verify_state(self._manifest_data())

    def store_provisional_environment(self, source: Path) -> EnvironmentArtifact:
        raw = source.read_bytes()
        try:
            environment = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ArchiveIntegrityError(
                "provisional environment is unreadable"
            ) from error
        if not isinstance(environment, dict):
            raise ArchiveIntegrityError("provisional environment must be an object")
        validate_provisional_environment(environment)
        value, environment_id = (
            digest(raw),
            EnvironmentId(string(environment, "environment_id")),
        )
        with self._locked():
            self._verify_locked()
            manifest = self._manifest_data()
            artifacts = manifest["artifacts"]
            assert isinstance(artifacts, list)
            if value not in artifacts:
                artifacts.append(value)
                self._transaction(
                    "store_artifact", self._artifacts / f"{value}.json", raw, manifest
                )
        return EnvironmentArtifact(environment_id, value)

    def store_artifact(self, data: bytes) -> str:
        value = digest(data)
        with self._locked():
            self._verify_locked()
            manifest = self._manifest_data()
            artifacts = manifest["artifacts"]
            assert isinstance(artifacts, list)
            if value not in artifacts:
                artifacts.append(value)
                self._transaction(
                    "store_artifact", self._artifacts / f"{value}.json", data, manifest
                )
        return value

    def append(self, payload: Mapping[str, object]) -> ExperimentEvent:
        validate_event_payload(payload)
        with self._locked():
            self._verify_locked()
            manifest = self._manifest_data()
            self._provenance(manifest, payload)
            events = manifest["events"]
            assert isinstance(events, list)
            event_id, experiment_id = (
                string(payload, "event_id"),
                string(payload, "experiment_id"),
            )
            if any(
                isinstance(item, dict) and item.get("event_id") == event_id
                for item in events
            ):
                raise ArchiveIntegrityError(f"event_id already exists: {event_id}")
            data = canonical_json(payload)
            value = digest(data)
            events.append(
                {
                    "event_id": event_id,
                    "experiment_id": experiment_id,
                    "sequence": len(events),
                    "event_kind": payload["event_kind"],
                    "payload_digest": value,
                }
            )
            self._transaction(
                "append_event", self._events / f"{event_id}.json", data, manifest
            )
            return ExperimentEvent(
                EventId(event_id),
                ExperimentId(experiment_id),
                len(events) - 1,
                string(payload, "event_kind"),
                value,
            )

    def verify(self) -> None:
        with self._locked():
            self._verify_locked()

    def _projection_bytes_locked(self) -> bytes:
        self._verify_locked()
        manifest = self._manifest_data()
        events = manifest["events"]
        assert isinstance(events, list)
        return canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "events": [
                    {
                        "event": entry,
                        "payload": json.loads(
                            (self._events / f"{entry['event_id']}.json").read_text()
                        ),
                    }
                    for entry in events
                ],
            }
        )

    def projection_bytes(self) -> bytes:
        with self._locked():
            return self._projection_bytes_locked()

    def projection(self) -> CatalogueProjection:
        with self._locked():
            projection = self._projection_bytes_locked()
            events = self._manifest_data()["events"]
            assert isinstance(events, list)
            event_ids = tuple(string(entry, "event_id") for entry in events)
            return CatalogueProjection(digest(projection), len(events), event_ids)
