"""Immutable human planning input; this module never initiates execution."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Iterator, Mapping, Protocol

SCHEMA_VERSION = 1
PLANNING_SCOPE = "planning_only"
QUALIFICATION_GATE = "FG-01"
ZERO_DIGEST = "0" * 64
IDEA_0001_SHA256 = "934ce6178629e5e72a98923444351c8de6f6ebd18c323acb2c4a72bf90a6938f"
_INPUT_ID = re.compile(r"^HRI-([0-9]{6})$")
_IDEA_ID = re.compile(r"^IDEA-[0-9]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BIB_KEY = re.compile(r"@\w+\{([^,]+),")
_TRACKER_KEY = re.compile(r"^\| `([^`]+)`", re.MULTILINE)
_RECORD_FIELDS = frozenset(
    {
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
)
_FORBIDDEN_FRAGMENTS = (
    "execution",
    "environment",
    "event",
    "candidate",
    "empirical",
    "measurement",
    "result",
    "profile",
    "trace",
    "counter",
    "artifact",
)


class HumanQueueIntegrityError(ValueError):
    """Raised when a human-input queue is incomplete, mutable, or invalid."""


class HumanInputKind(str, Enum):
    IDEA = "idea"
    CONSTRAINT = "constraint"
    LITERATURE = "literature"
    PRIORITY = "priority"
    REDIRECT = "redirect"


@dataclass(frozen=True, slots=True)
class HumanInputSubmission:
    recorded_at: str
    actor: str
    kind: HumanInputKind
    idea_id: str
    statement: str
    literature_keys: tuple[str, ...] = ()
    priority: int | None = None
    redirect_to: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.kind, HumanInputKind)
            or not _IDEA_ID.fullmatch(self.idea_id)
            or not self.recorded_at
            or not self.actor
            or not self.statement
            or not isinstance(self.literature_keys, tuple)
            or not all(isinstance(key, str) and key for key in self.literature_keys)
            or (
                self.priority is not None
                and (
                    not isinstance(self.priority, int)
                    or isinstance(self.priority, bool)
                )
            )
            or (self.redirect_to is not None and not isinstance(self.redirect_to, str))
        ):
            raise ValueError("human input submission fields are invalid")


@dataclass(frozen=True, slots=True)
class HumanInputRecord:
    schema_version: int
    input_id: str
    sequence: int
    recorded_at: str
    actor: str
    kind: HumanInputKind
    idea_id: str
    statement: str
    literature_keys: tuple[str, ...]
    priority: int | None
    redirect_to: str | None
    scope: str
    qualification_gate: str
    previous_digest: str
    record_digest: str

    def __post_init__(self) -> None:
        identifier = _INPUT_ID.fullmatch(self.input_id)
        if (
            self.schema_version != SCHEMA_VERSION
            or identifier is None
            or int(identifier.group(1)) != self.sequence
            or not isinstance(self.kind, HumanInputKind)
            or not _IDEA_ID.fullmatch(self.idea_id)
            or not self.recorded_at
            or not self.actor
            or not self.statement
            or not isinstance(self.literature_keys, tuple)
            or not all(isinstance(key, str) and key for key in self.literature_keys)
            or (
                self.priority is not None
                and (
                    not isinstance(self.priority, int)
                    or isinstance(self.priority, bool)
                )
            )
            or (self.redirect_to is not None and not isinstance(self.redirect_to, str))
            or self.scope != PLANNING_SCOPE
            or self.qualification_gate != QUALIFICATION_GATE
            or not _SHA256.fullmatch(self.previous_digest)
            or not _SHA256.fullmatch(self.record_digest)
        ):
            raise ValueError("human input record fields are invalid")


@dataclass(frozen=True, slots=True)
class HumanQueueItem:
    idea_id: str
    statement: str
    literature_keys: tuple[str, ...]
    priority: int
    constraints: tuple[str, ...]
    creation_sequence: int

    def __post_init__(self) -> None:
        if (
            not _IDEA_ID.fullmatch(self.idea_id)
            or not self.statement
            or not isinstance(self.literature_keys, tuple)
            or not isinstance(self.constraints, tuple)
            or not all(
                isinstance(value, str) and value for value in self.literature_keys
            )
            or not all(isinstance(value, str) and value for value in self.constraints)
            or not isinstance(self.priority, int)
            or isinstance(self.priority, bool)
            or self.creation_sequence <= 0
        ):
            raise ValueError("human queue item fields are invalid")


@dataclass(frozen=True, slots=True)
class HumanQueueProjection:
    projection_id: str
    items: tuple[HumanQueueItem, ...]

    def __post_init__(self) -> None:
        if (
            not _SHA256.fullmatch(self.projection_id)
            or not isinstance(self.items, tuple)
            or not all(isinstance(item, HumanQueueItem) for item in self.items)
        ):
            raise ValueError("human queue projection fields are invalid")


class HumanResearchQueue(Protocol):
    def append(self, submission: HumanInputSubmission) -> HumanInputRecord: ...

    def records(self) -> tuple[HumanInputRecord, ...]: ...

    def projection(self) -> HumanQueueProjection: ...


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise HumanQueueIntegrityError(
            "human input JSON must be finite and canonical"
        ) from error


def _digest(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _record_digest(value: Mapping[str, object]) -> str:
    return _digest({key: item for key, item in value.items() if key != "record_digest"})


def _record_mapping(record: HumanInputRecord) -> dict[str, object]:
    return {
        "schema_version": record.schema_version,
        "input_id": record.input_id,
        "sequence": record.sequence,
        "recorded_at": record.recorded_at,
        "actor": record.actor,
        "kind": record.kind.value,
        "idea_id": record.idea_id,
        "statement": record.statement,
        "literature_keys": list(record.literature_keys),
        "priority": record.priority,
        "redirect_to": record.redirect_to,
        "scope": record.scope,
        "qualification_gate": record.qualification_gate,
        "previous_digest": record.previous_digest,
        "record_digest": record.record_digest,
    }


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise HumanQueueIntegrityError(f"{name} must be a non-empty string")
    return value


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value))
    return set()


class FileHumanResearchQueue:
    """Locked, append-only HRI files rooted at ``research/ideas/intake``."""

    def __init__(self, project_root: Path, intake_root: Path | None = None) -> None:
        self._project_root = project_root
        self._root = intake_root or project_root / "research" / "ideas" / "intake"
        self._lock = self._root / ".human-input.lock"

    @contextmanager
    def _locked(self, mode: int) -> Iterator[None]:
        self._root.mkdir(parents=True, exist_ok=True)
        with self._lock.open("a+b") as lock:
            fcntl.flock(lock.fileno(), mode)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _atomic(cls, path: Path, data: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            cls._fsync_directory(path.parent)
        except Exception:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def _literature_keys(self) -> set[str]:
        tracker_keys: set[str] = set()
        for name in ("papers_read.md", "papers_to_read.md"):
            path = self._project_root / name
            if not path.is_file():
                raise HumanQueueIntegrityError(f"literature tracker is missing: {name}")
            tracker_keys.update(_TRACKER_KEY.findall(path.read_text(encoding="utf-8")))
        bibliography = self._project_root / "research" / "paper" / "bibliography.bib"
        if not bibliography.is_file():
            raise HumanQueueIntegrityError("literature bibliography is missing")
        bibliography_keys = set(
            _BIB_KEY.findall(bibliography.read_text(encoding="utf-8"))
        )
        return tracker_keys & bibliography_keys

    def _validate_literature(self, keys: tuple[str, ...]) -> None:
        if not all(isinstance(key, str) and key for key in keys):
            raise HumanQueueIntegrityError("literature keys must be non-empty strings")
        unknown = set(keys) - self._literature_keys()
        if unknown:
            raise HumanQueueIntegrityError(
                f"literature keys must resolve in tracker and bibliography: {sorted(unknown)}"
            )

    @staticmethod
    def _validate_kind(
        record: HumanInputRecord,
        known_ideas: set[str],
        redirects: Mapping[str, str],
    ) -> None:
        if not _IDEA_ID.fullmatch(record.idea_id):
            raise HumanQueueIntegrityError("idea_id must match IDEA-N")
        if record.kind is HumanInputKind.IDEA:
            if (
                record.idea_id in known_ideas
                or record.priority is not None
                or record.redirect_to is not None
            ):
                raise HumanQueueIntegrityError(
                    "idea must create one new idea without priority or redirect"
                )
            return
        if record.idea_id not in known_ideas:
            raise HumanQueueIntegrityError(
                "human input must reference an existing idea"
            )
        if record.kind is HumanInputKind.CONSTRAINT:
            if (
                record.literature_keys
                or record.priority is not None
                or record.redirect_to is not None
            ):
                raise HumanQueueIntegrityError(
                    "constraint may carry only its statement"
                )
            return
        if record.kind is HumanInputKind.LITERATURE:
            if (
                not record.literature_keys
                or record.priority is not None
                or record.redirect_to is not None
            ):
                raise HumanQueueIntegrityError("literature input requires keys only")
            return
        if record.kind is HumanInputKind.PRIORITY:
            if (
                record.literature_keys
                or record.redirect_to is not None
                or not isinstance(record.priority, int)
                or isinstance(record.priority, bool)
                or not 0 <= record.priority <= 100
            ):
                raise HumanQueueIntegrityError(
                    "priority must be an integer from 0 through 100"
                )
            return
        if record.kind is HumanInputKind.REDIRECT:
            target = record.redirect_to
            if (
                record.literature_keys
                or record.priority is not None
                or not isinstance(target, str)
                or target not in known_ideas
                or target == record.idea_id
            ):
                raise HumanQueueIntegrityError(
                    "redirect requires a distinct existing target"
                )
            if record.idea_id in redirects:
                raise HumanQueueIntegrityError(
                    "redirect source may be redirected only once"
                )
            graph = dict(redirects)
            graph[record.idea_id] = target
            current = record.idea_id
            seen: set[str] = set()
            while current in graph:
                if current in seen:
                    raise HumanQueueIntegrityError("redirects must not form a cycle")
                seen.add(current)
                current = graph[current]
            return
        raise HumanQueueIntegrityError("human input kind is unsupported")

    def _record_from_value(
        self,
        value: object,
        expected_sequence: int,
        previous_digest: str,
        known_ideas: set[str],
        redirects: Mapping[str, str],
    ) -> HumanInputRecord:
        if not isinstance(value, dict) or set(value) != _RECORD_FIELDS:
            raise HumanQueueIntegrityError("human input schema fields do not match")
        forbidden = {
            key
            for key in _keys(value)
            if any(fragment in key.lower() for fragment in _FORBIDDEN_FRAGMENTS)
        }
        if forbidden:
            raise HumanQueueIntegrityError(
                f"human input contains forbidden fields: {sorted(forbidden)}"
            )
        input_id = _string(value.get("input_id"), "input_id")
        identifier = _INPUT_ID.fullmatch(input_id)
        if identifier is None or int(identifier.group(1)) != expected_sequence:
            raise HumanQueueIntegrityError("human input files must be contiguous")
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or value.get("sequence") != expected_sequence
        ):
            raise HumanQueueIntegrityError(
                "human input schema version or sequence is invalid"
            )
        kind_value = value.get("kind")
        try:
            kind = HumanInputKind(kind_value)
        except (TypeError, ValueError) as error:
            raise HumanQueueIntegrityError("human input kind is invalid") from error
        literature = value.get("literature_keys")
        if not isinstance(literature, list):
            raise HumanQueueIntegrityError("literature_keys must be a list")
        priority = value.get("priority")
        redirect_to = value.get("redirect_to")
        if priority is not None and (
            not isinstance(priority, int) or isinstance(priority, bool)
        ):
            raise HumanQueueIntegrityError("priority must be an integer or null")
        if redirect_to is not None and not isinstance(redirect_to, str):
            raise HumanQueueIntegrityError("redirect_to must be a string or null")
        scope = _string(value.get("scope"), "scope")
        qualification_gate = _string(
            value.get("qualification_gate"), "qualification_gate"
        )
        if scope != PLANNING_SCOPE or qualification_gate != QUALIFICATION_GATE:
            raise HumanQueueIntegrityError(
                "human input must remain planning_only behind FG-01"
            )
        record = HumanInputRecord(
            schema_version=SCHEMA_VERSION,
            input_id=input_id,
            sequence=expected_sequence,
            recorded_at=_string(value.get("recorded_at"), "recorded_at"),
            actor=_string(value.get("actor"), "actor"),
            kind=kind,
            idea_id=_string(value.get("idea_id"), "idea_id"),
            statement=_string(value.get("statement"), "statement"),
            literature_keys=tuple(literature),
            priority=priority,
            redirect_to=redirect_to,
            scope=scope,
            qualification_gate=qualification_gate,
            previous_digest=_string(value.get("previous_digest"), "previous_digest"),
            record_digest=_string(value.get("record_digest"), "record_digest"),
        )
        if record.previous_digest != previous_digest or not _SHA256.fullmatch(
            record.previous_digest
        ):
            raise HumanQueueIntegrityError("human input digest chain is invalid")
        if not _SHA256.fullmatch(
            record.record_digest
        ) or record.record_digest != _record_digest(value):
            raise HumanQueueIntegrityError("human input record digest is invalid")
        self._validate_literature(record.literature_keys)
        self._validate_kind(record, known_ideas, redirects)
        return record

    def _validate_seed(self, record: HumanInputRecord) -> None:
        if record.sequence != 1:
            return
        source = self._project_root / "research" / "ideas" / "IDEA-0001.json"
        if (
            self._root != self._project_root / "research" / "ideas" / "intake"
            or not source.is_file()
        ):
            return
        if sha256(source.read_bytes()).hexdigest() != IDEA_0001_SHA256:
            raise HumanQueueIntegrityError("IDEA-0001 custody digest is invalid")
        idea = json.loads(source.read_text(encoding="utf-8"))
        if (
            record.kind is not HumanInputKind.IDEA
            or record.idea_id != idea.get("idea_id")
            or record.statement != idea.get("question")
            or list(record.literature_keys) != idea.get("literature_keys")
        ):
            raise HumanQueueIntegrityError(
                "HRI-000001 must preserve IDEA-0001 semantics"
            )

    def _records_locked(self) -> tuple[HumanInputRecord, ...]:
        if not self._root.exists():
            return ()
        paths: list[Path] = []
        for path in self._root.iterdir():
            if path.is_symlink():
                raise HumanQueueIntegrityError(
                    "human intake must not contain symbolic links"
                )
            if path.name == self._lock.name:
                if not path.is_file():
                    raise HumanQueueIntegrityError(
                        "human intake has an unexpected or partial file"
                    )
                continue
            if (
                not path.is_file()
                or _INPUT_ID.fullmatch(path.stem) is None
                or path.suffix != ".json"
            ):
                raise HumanQueueIntegrityError(
                    "human intake has an unexpected or partial file"
                )
            paths.append(path)
        paths.sort(key=lambda path: path.name)
        records: list[HumanInputRecord] = []
        known_ideas: set[str] = set()
        redirects: dict[str, str] = {}
        previous = ZERO_DIGEST
        for sequence, path in enumerate(paths, start=1):
            try:
                raw = path.read_bytes()
                value = json.loads(raw)
            except (OSError, json.JSONDecodeError) as error:
                raise HumanQueueIntegrityError(
                    "human input file is partial or invalid"
                ) from error
            if raw != _canonical_json(value):
                raise HumanQueueIntegrityError("human input file is not canonical")
            record = self._record_from_value(
                value, sequence, previous, known_ideas, redirects
            )
            self._validate_seed(record)
            records.append(record)
            if record.kind is HumanInputKind.IDEA:
                known_ideas.add(record.idea_id)
            elif record.kind is HumanInputKind.REDIRECT:
                assert record.redirect_to is not None
                redirects[record.idea_id] = record.redirect_to
            previous = record.record_digest
        return tuple(records)

    def records(self) -> tuple[HumanInputRecord, ...]:
        if not self._root.exists():
            return ()
        with self._locked(fcntl.LOCK_SH):
            return self._records_locked()

    def append(self, submission: HumanInputSubmission) -> HumanInputRecord:
        with self._locked(fcntl.LOCK_EX):
            records = self._records_locked()
            sequence = len(records) + 1
            prior = records[-1].record_digest if records else ZERO_DIGEST
            raw: dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "input_id": f"HRI-{sequence:06d}",
                "sequence": sequence,
                "recorded_at": submission.recorded_at,
                "actor": submission.actor,
                "kind": submission.kind.value,
                "idea_id": submission.idea_id,
                "statement": submission.statement,
                "literature_keys": list(submission.literature_keys),
                "priority": submission.priority,
                "redirect_to": submission.redirect_to,
                "scope": PLANNING_SCOPE,
                "qualification_gate": QUALIFICATION_GATE,
                "previous_digest": prior,
                "record_digest": "",
            }
            raw["record_digest"] = _record_digest(raw)
            known_ideas = {
                record.idea_id
                for record in records
                if record.kind is HumanInputKind.IDEA
            }
            redirects = {
                record.idea_id: record.redirect_to
                for record in records
                if record.kind is HumanInputKind.REDIRECT
                and record.redirect_to is not None
            }
            record = self._record_from_value(
                raw, sequence, prior, known_ideas, redirects
            )
            self._atomic(self._root / f"{record.input_id}.json", _canonical_json(raw))
            return record

    def projection(self) -> HumanQueueProjection:
        records = self.records()
        ideas: dict[str, HumanInputRecord] = {}
        priorities: dict[str, int] = {}
        constraints: dict[str, list[str]] = {}
        literature: dict[str, list[str]] = {}
        redirected: set[str] = set()
        for record in records:
            if record.kind is HumanInputKind.IDEA:
                ideas[record.idea_id] = record
                literature[record.idea_id] = list(record.literature_keys)
            elif record.kind is HumanInputKind.CONSTRAINT:
                constraints.setdefault(record.idea_id, []).append(record.statement)
            elif record.kind is HumanInputKind.LITERATURE:
                literature.setdefault(record.idea_id, []).extend(record.literature_keys)
            elif record.kind is HumanInputKind.PRIORITY:
                assert record.priority is not None
                priorities[record.idea_id] = record.priority
            elif record.kind is HumanInputKind.REDIRECT:
                redirected.add(record.idea_id)
        items = []
        for idea_id, record in ideas.items():
            if idea_id in redirected:
                continue
            seen: set[str] = set()
            keys = tuple(
                key
                for key in literature.get(idea_id, [])
                if not (key in seen or seen.add(key))
            )
            items.append(
                HumanQueueItem(
                    idea_id=idea_id,
                    statement=record.statement,
                    literature_keys=keys,
                    priority=priorities.get(idea_id, 0),
                    constraints=tuple(constraints.get(idea_id, [])),
                    creation_sequence=record.sequence,
                )
            )
        ordered = tuple(
            sorted(
                items,
                key=lambda item: (-item.priority, item.creation_sequence, item.idea_id),
            )
        )
        projection_id = _digest([_record_mapping(record) for record in records])
        return HumanQueueProjection(projection_id, ordered)


__all__ = [
    "FileHumanResearchQueue",
    "HumanInputKind",
    "HumanInputRecord",
    "HumanInputSubmission",
    "HumanQueueIntegrityError",
    "HumanQueueItem",
    "HumanQueueProjection",
    "HumanResearchQueue",
]
