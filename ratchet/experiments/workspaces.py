"""Safe, provenance-bound local Git worktree lifecycle for experiments.

This module intentionally has no archive or candidate-execution dependency.  It only
creates isolated local worktrees and combines already-finalized commits through a
small, literal-argument Git adapter.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Sequence

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_EXPERIMENT = re.compile(r"^EXP-[0-9]{4}$")
_PROTOCOL = re.compile(r"^PROTO-[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-[0-9]{4}$")
_LANE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_WORKSPACE_REF = re.compile(
    r"^refs/heads/ratchet/experiments/(?:"
    r"exp-[0-9]{4}/proto-[a-z][a-z0-9-]*/[a-z][a-z0-9-]{0,31}"
    r"|integration/proto-[a-z][a-z0-9-]*/[0-9a-f]{16})$"
)
_ZERO_COMMIT = "0" * 40


class WorkspaceLifecycleError(RuntimeError):
    """Raised when the local Git lifecycle cannot safely continue."""


def _protocol_slug(protocol_id: str) -> str:
    return protocol_id.removeprefix("PROTO-").lower()


def _branch(experiment_id: str, protocol_id: str, lane: str) -> str:
    return (
        f"ratchet/experiments/{experiment_id.lower()}/"
        f"proto-{_protocol_slug(protocol_id)}/{lane}"
    )


def _path_name(experiment_id: str, protocol_id: str, lane: str) -> str:
    return f"{experiment_id.lower()}--proto-{_protocol_slug(protocol_id)}--{lane}"


def _canonical_digest(value: object) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _validate_spec_fields(
    experiment_id: str,
    protocol_id: str,
    protocol_digest: str,
    lane: str,
    base_commit: str,
) -> None:
    if not isinstance(experiment_id, str) or not _EXPERIMENT.fullmatch(experiment_id):
        raise ValueError("experiment_id must match EXP-NNNN")
    if not isinstance(protocol_id, str) or not _PROTOCOL.fullmatch(protocol_id):
        raise ValueError("protocol_id is invalid")
    if not isinstance(protocol_digest, str) or not _DIGEST.fullmatch(protocol_digest):
        raise ValueError("protocol_digest must be a lowercase SHA-256 digest")
    if not isinstance(lane, str) or not _LANE.fullmatch(lane):
        raise ValueError("lane is invalid")
    if not isinstance(base_commit, str) or not _COMMIT.fullmatch(base_commit):
        raise ValueError("base_commit must be a lowercase full Git SHA")


@dataclass(frozen=True, slots=True)
class ExperimentWorkspaceSpec:
    """Validated immutable inputs for one independently-developed experiment."""

    experiment_id: str
    protocol_id: str
    protocol_digest: str
    lane: str
    base_commit: str

    def __post_init__(self) -> None:
        _validate_spec_fields(
            self.experiment_id,
            self.protocol_id,
            self.protocol_digest,
            self.lane,
            self.base_commit,
        )


@dataclass(frozen=True, slots=True)
class ExperimentWorkspace:
    """A generated branch and direct-child worktree path for a specification."""

    experiment_id: str
    protocol_id: str
    protocol_digest: str
    lane: str
    base_commit: str
    branch: str
    path: Path

    def __post_init__(self) -> None:
        _validate_spec_fields(
            self.experiment_id,
            self.protocol_id,
            self.protocol_digest,
            self.lane,
            self.base_commit,
        )
        if self.branch != _branch(self.experiment_id, self.protocol_id, self.lane):
            raise ValueError("workspace branch is not the generated branch")
        if not isinstance(self.path, Path) or self.path.name != _path_name(
            self.experiment_id, self.protocol_id, self.lane
        ):
            raise ValueError("workspace path is not the generated direct-child path")


@dataclass(frozen=True, slots=True)
class WorkspaceProvenance:
    """Clean, descendant commit provenance for a finalized workspace."""

    experiment_id: str
    protocol_id: str
    protocol_digest: str
    lane: str
    branch: str
    base_commit: str
    head_commit: str
    changed_paths: tuple[str, ...]
    provenance_digest: str

    def __post_init__(self) -> None:
        _validate_spec_fields(
            self.experiment_id,
            self.protocol_id,
            self.protocol_digest,
            self.lane,
            self.base_commit,
        )
        if self.branch != _branch(self.experiment_id, self.protocol_id, self.lane):
            raise ValueError("provenance branch is not the generated branch")
        if not isinstance(self.head_commit, str) or not _COMMIT.fullmatch(
            self.head_commit
        ):
            raise ValueError("head_commit must be a lowercase full Git SHA")
        if (
            not isinstance(self.changed_paths, tuple)
            or tuple(sorted(set(self.changed_paths))) != self.changed_paths
        ):
            raise ValueError("changed_paths must be an immutable ordered unique tuple")
        for changed_path in self.changed_paths:
            path = PurePosixPath(changed_path)
            if (
                not isinstance(changed_path, str)
                or not changed_path
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError("changed_paths contains an invalid repository path")
        expected = _canonical_digest(
            {
                "base_commit": self.base_commit,
                "branch": self.branch,
                "changed_paths": self.changed_paths,
                "experiment_id": self.experiment_id,
                "head_commit": self.head_commit,
                "lane": self.lane,
                "protocol_digest": self.protocol_digest,
                "protocol_id": self.protocol_id,
            }
        )
        if self.provenance_digest != expected:
            raise ValueError("provenance_digest does not match canonical provenance")


@dataclass(frozen=True, slots=True)
class ConsolidationResult:
    """Outcome of deterministic, compare-create integration of source commits."""

    status: Literal["consolidated", "conflict", "already_exists"]
    integration_branch: str | None
    integration_commit: str | None
    source_provenance: tuple[WorkspaceProvenance, ...]
    conflicts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in {"consolidated", "conflict", "already_exists"}:
            raise ValueError("consolidation status is invalid")
        if not isinstance(self.source_provenance, tuple) or not self.source_provenance:
            raise ValueError("source_provenance must be a non-empty immutable tuple")
        if not isinstance(self.conflicts, tuple):
            raise ValueError("conflicts must be immutable")
        if self.status in {"consolidated", "already_exists"}:
            if (
                not isinstance(self.integration_branch, str)
                or not isinstance(self.integration_commit, str)
                or not _COMMIT.fullmatch(self.integration_commit)
                or self.conflicts
            ):
                raise ValueError("successful consolidation is incomplete")
        elif (
            self.integration_branch is not None
            or self.integration_commit is not None
            or not self.conflicts
        ):
            raise ValueError("conflicted consolidation must name only conflicts")


@dataclass(frozen=True, slots=True)
class WorkspaceCleanupResult:
    """Cleanup result which always retains the experiment branch."""

    status: Literal["removed", "retained"]
    branch_retained: bool
    reason: str

    def __post_init__(self) -> None:
        if self.status not in {"removed", "retained"} or not self.branch_retained:
            raise ValueError("cleanup must retain the experiment branch")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("cleanup reason is required")


class ExperimentWorkspaceLifecycle(Protocol):
    """Public lifecycle boundary; implementations never execute candidates."""

    def create(self, spec: ExperimentWorkspaceSpec) -> ExperimentWorkspace: ...

    def finalize(self, workspace: ExperimentWorkspace) -> WorkspaceProvenance: ...

    def consolidate(
        self, provenance: Sequence[WorkspaceProvenance]
    ) -> ConsolidationResult: ...

    def cleanup(
        self,
        workspace: ExperimentWorkspace,
        provenance: WorkspaceProvenance,
        consolidation: ConsolidationResult | None,
    ) -> WorkspaceCleanupResult: ...


@dataclass(frozen=True, slots=True)
class _GitResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True, slots=True)
class _GitBytesResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True, slots=True)
class _DirectoryIdentity:
    """Stable identity for a manager-owned directory between lifecycle calls."""

    path: Path
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _IntegrationDerivation:
    """Deterministic integration commit or validated merge conflict paths."""

    commit: str | None
    conflicts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.commit is None:
            if not self.conflicts:
                raise ValueError("conflicted derivation must name conflicts")
        elif not _COMMIT.fullmatch(self.commit) or self.conflicts:
            raise ValueError("successful derivation is invalid")


class _LocalGit:
    """Internal subprocess adapter using literal argv and no shell or remote commands."""

    def __init__(self, repository: Path) -> None:
        self._repository = repository

    def run(self, arguments: Sequence[str], *, cwd: Path | None = None) -> _GitResult:
        self._validate_arguments(arguments)
        command = ("git", "-C", str(cwd or self._repository), *arguments)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            env=None,
        )
        return _GitResult(completed.returncode, completed.stdout, completed.stderr)

    def read_blob(self, argument: str) -> _GitBytesResult:
        arguments = ("show", argument)
        self._validate_arguments(arguments)
        command = ("git", "-C", str(self._repository), *arguments)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=False,
            shell=False,
            env=None,
        )
        return _GitBytesResult(completed.returncode, completed.stdout, completed.stderr)

    def run_with_identity(self, arguments: Sequence[str]) -> _GitResult:
        if (
            len(arguments) != 8
            or arguments[0] != "commit-tree"
            or arguments[2] != "-p"
            or arguments[4] != "-p"
            or arguments[6] != "-m"
        ):
            raise WorkspaceLifecycleError("Git operation is not allowlisted")
        command = ("git", "-C", str(self._repository), *arguments)
        environment = os.environ.copy()
        environment.update(
            {
                "GIT_AUTHOR_NAME": "Ratchet Workspace",
                "GIT_AUTHOR_EMAIL": "workspace@ratchet.invalid",
                "GIT_AUTHOR_DATE": "946684800 +0000",
                "GIT_COMMITTER_NAME": "Ratchet Workspace",
                "GIT_COMMITTER_EMAIL": "workspace@ratchet.invalid",
                "GIT_COMMITTER_DATE": "946684800 +0000",
            }
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            env=environment,
        )
        return _GitResult(completed.returncode, completed.stdout, completed.stderr)

    @staticmethod
    def _validate_arguments(arguments: Sequence[str]) -> None:
        if not arguments or not all(isinstance(item, str) for item in arguments):
            raise WorkspaceLifecycleError("Git operation is not allowlisted")
        command = arguments[0]
        revision = re.compile(r"^[0-9a-f]{40}\.\.\.[0-9a-f]{40}$")
        ref_commit = re.compile(
            r"^refs/heads/ratchet/experiments/(?:"
            r"exp-[0-9]{4}/proto-[a-z][a-z0-9-]*/[a-z][a-z0-9-]{0,31}"
            r"|integration/proto-[a-z][a-z0-9-]*/[0-9a-f]{16})\^\{commit\}$"
        )
        allowed = (
            arguments == ("rev-parse", "--show-toplevel")
            or command == "rev-parse"
            and len(arguments) == 3
            and arguments[1] == "--verify"
            and (
                arguments[2] == "HEAD^{commit}"
                or re.fullmatch(r"[0-9a-f]{40}\^\{commit\}", arguments[2])
                or ref_commit.fullmatch(arguments[2]) is not None
            )
            or command == "cat-file"
            and len(arguments) == 3
            and arguments[1] == "-e"
            and re.fullmatch(r"[0-9a-f]{40}\^\{commit\}", arguments[2]) is not None
            or arguments == ("status", "--porcelain")
            or arguments == ("worktree", "list", "--porcelain")
            or command == "worktree"
            and len(arguments) == 6
            and arguments[1:3] == ("add", "-b")
            and _WORKSPACE_REF.fullmatch(f"refs/heads/{arguments[3]}") is not None
            and Path(arguments[4]).is_absolute()
            and _COMMIT.fullmatch(arguments[5]) is not None
            or command == "worktree"
            and len(arguments) == 3
            and arguments[1] == "remove"
            and Path(arguments[2]).is_absolute()
            or command == "show-ref"
            and arguments[:3] == ("show-ref", "--verify", "--quiet")
            and len(arguments) == 4
            and _WORKSPACE_REF.fullmatch(arguments[3]) is not None
            or command == "show-ref"
            and arguments[:2] == ("show-ref", "--verify")
            and len(arguments) == 3
            and _WORKSPACE_REF.fullmatch(arguments[2]) is not None
            or command == "diff"
            and len(arguments) == 3
            and arguments[1] == "--check"
            and revision.fullmatch(arguments[2]) is not None
            or command == "diff"
            and len(arguments) == 4
            and arguments[1:3] == ("--name-only", "-z")
            and revision.fullmatch(arguments[3]) is not None
            or command == "merge-base"
            and len(arguments) == 4
            and arguments[1] == "--is-ancestor"
            and _COMMIT.fullmatch(arguments[2]) is not None
            and _COMMIT.fullmatch(arguments[3]) is not None
            or command == "merge-tree"
            and len(arguments) == 7
            and arguments[1:5] == ("--write-tree", "--name-only", "-z", "--no-messages")
            and _COMMIT.fullmatch(arguments[5]) is not None
            and _COMMIT.fullmatch(arguments[6]) is not None
            or command == "update-ref"
            and len(arguments) == 4
            and _WORKSPACE_REF.fullmatch(arguments[1]) is not None
            and "/integration/" in arguments[1]
            and _COMMIT.fullmatch(arguments[2]) is not None
            and arguments[3] == _ZERO_COMMIT
            or command == "show"
            and len(arguments) == 2
            and re.fullmatch(
                r"[0-9a-f]{40}:research/protocols/PROTO-[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*-[0-9]{4}\.json",
                arguments[1],
            )
            is not None
        )
        if not allowed:
            raise WorkspaceLifecycleError("Git operation is not allowlisted")


class ExperimentWorkspaceManager:
    """Lifecycle manager for explicit, external, non-symlink worktree roots."""

    def __init__(self, repository: Path, worktree_root: Path) -> None:
        self._repository_identity = self._capture_directory(repository, "repository")
        self._worktree_root_identity = self._capture_directory(
            worktree_root, "worktree_root"
        )
        self._repository = self._repository_identity.path
        self._worktree_root = self._worktree_root_identity.path
        if (
            self._repository == self._worktree_root
            or self._repository in self._worktree_root.parents
            or self._worktree_root in self._repository.parents
        ):
            raise ValueError("worktree_root must be external to the repository")
        self._git = _LocalGit(self._repository)
        top_level = self._require(("rev-parse", "--show-toplevel")).stdout.strip()
        if Path(top_level).resolve(strict=True) != self._repository:
            raise ValueError("repository must be the Git top-level")

    @staticmethod
    def _capture_directory(path: Path, label: str) -> _DirectoryIdentity:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError(f"{label} must be an explicit non-symlink path")
        try:
            metadata = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValueError(
                f"{label} must be an existing non-symlink directory"
            ) from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or resolved != path
        ):
            raise ValueError(
                f"{label} must be an explicit existing non-symlink directory"
            )
        return _DirectoryIdentity(path, metadata.st_dev, metadata.st_ino)

    @staticmethod
    def _revalidate_directory(identity: _DirectoryIdentity, label: str) -> None:
        try:
            metadata = identity.path.lstat()
            resolved = identity.path.resolve(strict=True)
        except OSError as error:
            raise WorkspaceLifecycleError(
                f"{label} directory identity changed"
            ) from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_dev != identity.device
            or metadata.st_ino != identity.inode
            or resolved != identity.path
        ):
            raise WorkspaceLifecycleError(f"{label} directory identity changed")

    def _revalidate_roots(self) -> None:
        self._revalidate_directory(self._repository_identity, "repository")
        self._revalidate_directory(self._worktree_root_identity, "worktree_root")

    @staticmethod
    def _failure(result: _GitResult) -> str:
        return (result.stderr or result.stdout).strip() or "Git command failed"

    def _require(
        self, arguments: Sequence[str], *, cwd: Path | None = None
    ) -> _GitResult:
        result = self._git.run(arguments, cwd=cwd)
        if result.returncode != 0:
            raise WorkspaceLifecycleError(self._failure(result))
        return result

    def _commit_exists(self, commit: str) -> None:
        resolved = self._require(
            ("rev-parse", "--verify", f"{commit}^{{commit}}")
        ).stdout.strip()
        if resolved != commit:
            raise WorkspaceLifecycleError(
                "commit does not resolve to its supplied full SHA"
            )
        self._require(("cat-file", "-e", f"{commit}^{{commit}}"))

    def _resolve_ref(self, reference: str) -> str:
        resolved = self._require(
            ("rev-parse", "--verify", f"{reference}^{{commit}}")
        ).stdout.strip()
        if not _COMMIT.fullmatch(resolved):
            raise WorkspaceLifecycleError(
                "Git ref did not resolve to a full commit SHA"
            )
        return resolved

    def _verify_source_ref(self, item: WorkspaceProvenance) -> None:
        reference = f"refs/heads/{item.branch}"
        resolved = self._resolve_ref(reference)
        if resolved != item.head_commit:
            raise WorkspaceLifecycleError(
                "source branch does not match provenance head"
            )

    def _bind_protocol(self, spec: ExperimentWorkspaceSpec) -> None:
        path = f"research/protocols/{spec.protocol_id}.json"
        result = self._git.read_blob(f"{spec.base_commit}:{path}")
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise WorkspaceLifecycleError(
                detail or "protocol blob is absent at base_commit"
            )
        try:
            payload = json.loads(result.stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WorkspaceLifecycleError(
                "protocol blob is not valid UTF-8 JSON"
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("protocol_id") != spec.protocol_id
        ):
            raise WorkspaceLifecycleError(
                "protocol blob does not name the supplied protocol_id"
            )
        if sha256(result.stdout).hexdigest() != spec.protocol_digest:
            raise WorkspaceLifecycleError(
                "protocol_digest does not match the base-commit blob"
            )

    def _verify_provenance_against_git(self, item: WorkspaceProvenance) -> None:
        """Reject self-consistent records that do not describe their Git commits."""
        self._commit_exists(item.base_commit)
        self._bind_protocol(
            ExperimentWorkspaceSpec(
                item.experiment_id,
                item.protocol_id,
                item.protocol_digest,
                item.lane,
                item.base_commit,
            )
        )
        self._commit_exists(item.head_commit)
        self._verify_source_ref(item)
        result = self._git.run(
            ("merge-base", "--is-ancestor", item.base_commit, item.head_commit)
        )
        if result.returncode != 0:
            if result.returncode == 1:
                raise WorkspaceLifecycleError(
                    "source provenance is not a base descendant"
                )
            raise WorkspaceLifecycleError(self._failure(result))
        revision = f"{item.base_commit}...{item.head_commit}"
        self._require(("diff", "--check", revision))
        names = self._require(("diff", "--name-only", "-z", revision)).stdout
        actual_paths = tuple(sorted(path for path in names.split("\0") if path))
        if actual_paths != item.changed_paths:
            raise WorkspaceLifecycleError(
                "source provenance changed_paths do not match its Git diff"
            )

    def _validate_workspace(self, workspace: ExperimentWorkspace) -> None:
        if (
            self._worktree_root.is_symlink()
            or workspace.path.parent != self._worktree_root
            or workspace.path.is_symlink()
        ):
            raise ValueError(
                "workspace must be a non-symlink direct child of worktree_root"
            )

    def _registered(self, path: Path) -> bool:
        result = self._require(("worktree", "list", "--porcelain"))
        expected = str(path.resolve())
        return any(
            line.removeprefix("worktree ") == expected
            for line in result.stdout.splitlines()
            if line.startswith("worktree ")
        )

    def create(self, spec: ExperimentWorkspaceSpec) -> ExperimentWorkspace:
        self._revalidate_roots()
        self._commit_exists(spec.base_commit)
        self._bind_protocol(spec)
        branch = _branch(spec.experiment_id, spec.protocol_id, spec.lane)
        path = self._worktree_root / _path_name(
            spec.experiment_id, spec.protocol_id, spec.lane
        )
        workspace = ExperimentWorkspace(
            spec.experiment_id,
            spec.protocol_id,
            spec.protocol_digest,
            spec.lane,
            spec.base_commit,
            branch,
            path,
        )
        if path.exists() or path.is_symlink():
            raise WorkspaceLifecycleError("generated workspace path already exists")
        existing = self._git.run(
            ("show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
        )
        if existing.returncode == 0:
            raise WorkspaceLifecycleError("generated experiment branch already exists")
        if existing.returncode not in {0, 1}:
            raise WorkspaceLifecycleError(self._failure(existing))
        self._revalidate_roots()
        self._require(("worktree", "add", "-b", branch, str(path), spec.base_commit))
        if not self._registered(path):
            raise WorkspaceLifecycleError(
                "Git did not register the generated workspace"
            )
        return workspace

    def finalize(self, workspace: ExperimentWorkspace) -> WorkspaceProvenance:
        self._revalidate_roots()
        self._validate_workspace(workspace)
        if not workspace.path.is_dir() or not self._registered(workspace.path):
            raise WorkspaceLifecycleError(
                "workspace is not a registered local worktree"
            )
        status = self._require(("status", "--porcelain"), cwd=workspace.path).stdout
        if status:
            raise WorkspaceLifecycleError("workspace must be clean before finalization")
        self._commit_exists(workspace.base_commit)
        self._bind_protocol(
            ExperimentWorkspaceSpec(
                workspace.experiment_id,
                workspace.protocol_id,
                workspace.protocol_digest,
                workspace.lane,
                workspace.base_commit,
            )
        )
        head = self._require(
            ("rev-parse", "--verify", "HEAD^{commit}"), cwd=workspace.path
        ).stdout.strip()
        if not _COMMIT.fullmatch(head):
            raise WorkspaceLifecycleError("workspace HEAD is not a full commit SHA")
        branch_head = self._require(
            ("show-ref", "--verify", f"refs/heads/{workspace.branch}")
        ).stdout.split(maxsplit=1)[0]
        if branch_head != head:
            raise WorkspaceLifecycleError(
                "workspace HEAD does not match its generated branch"
            )
        descendant = self._git.run(
            ("merge-base", "--is-ancestor", workspace.base_commit, head)
        )
        if descendant.returncode != 0:
            if descendant.returncode == 1:
                raise WorkspaceLifecycleError(
                    "workspace HEAD must descend from base_commit"
                )
            raise WorkspaceLifecycleError(self._failure(descendant))
        self._require(("diff", "--check", f"{workspace.base_commit}...{head}"))
        names = self._require(
            ("diff", "--name-only", "-z", f"{workspace.base_commit}...{head}")
        ).stdout
        changed_paths = tuple(sorted(path for path in names.split("\0") if path))
        digest = _canonical_digest(
            {
                "base_commit": workspace.base_commit,
                "changed_paths": changed_paths,
                "experiment_id": workspace.experiment_id,
                "head_commit": head,
                "lane": workspace.lane,
                "protocol_digest": workspace.protocol_digest,
                "protocol_id": workspace.protocol_id,
                "branch": workspace.branch,
            }
        )
        return WorkspaceProvenance(
            workspace.experiment_id,
            workspace.protocol_id,
            workspace.protocol_digest,
            workspace.lane,
            workspace.branch,
            workspace.base_commit,
            head,
            changed_paths,
            digest,
        )

    @staticmethod
    def _ordered(
        provenance: Sequence[WorkspaceProvenance],
    ) -> tuple[WorkspaceProvenance, ...]:
        if not isinstance(provenance, (tuple, list)) or not provenance:
            raise ValueError("consolidation requires non-empty provenance")
        ordered = tuple(
            sorted(
                provenance,
                key=lambda item: (item.experiment_id, item.lane, item.head_commit),
            )
        )
        first = ordered[0]
        if any(
            item.base_commit != first.base_commit
            or item.protocol_id != first.protocol_id
            or item.protocol_digest != first.protocol_digest
            for item in ordered
        ):
            raise ValueError("consolidation provenance must share base and protocol")
        if len({(item.experiment_id, item.lane) for item in ordered}) != len(ordered):
            raise ValueError(
                "consolidation provenance must have unique experiment lanes"
            )
        return ordered

    @staticmethod
    def _integration_branch(ordered: tuple[WorkspaceProvenance, ...]) -> str:
        first = ordered[0]
        group_digest = _canonical_digest(
            {
                "base_commit": first.base_commit,
                "protocol_digest": first.protocol_digest,
                "sources": [item.provenance_digest for item in ordered],
            }
        )
        return (
            f"ratchet/experiments/integration/proto-{_protocol_slug(first.protocol_id)}/"
            f"{group_digest[:16]}"
        )

    @staticmethod
    def _conflict_paths(output: str) -> tuple[str, ...]:
        paths = tuple(path for path in output.split("\0") if path)
        if paths and _COMMIT.fullmatch(paths[0]):
            paths = paths[1:]
        if not paths:
            raise WorkspaceLifecycleError(
                "merge-tree conflict did not name repository paths"
            )
        validated: list[str] = []
        for item in paths:
            path = PurePosixPath(item)
            if (
                path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or str(path) != item
            ):
                raise WorkspaceLifecycleError(
                    "merge-tree conflict named an invalid repository path"
                )
            validated.append(item)
        return tuple(sorted(set(validated)))

    def _derive_integration_commit(
        self, ordered: tuple[WorkspaceProvenance, ...]
    ) -> _IntegrationDerivation:
        """Derive the deterministic commit without creating or updating a ref."""
        current = ordered[0].base_commit
        for item in ordered:
            merged = self._git.run(
                (
                    "merge-tree",
                    "--write-tree",
                    "--name-only",
                    "-z",
                    "--no-messages",
                    current,
                    item.head_commit,
                )
            )
            if merged.returncode == 1:
                return _IntegrationDerivation(None, self._conflict_paths(merged.stdout))
            if merged.returncode != 0:
                raise WorkspaceLifecycleError(self._failure(merged))
            tree = merged.stdout.rstrip("\0\n")
            if not _COMMIT.fullmatch(tree):
                raise WorkspaceLifecycleError("merge-tree did not return a tree SHA")
            message = f"Ratchet integration {item.experiment_id} {item.lane}\n"
            commit = self._git.run_with_identity(
                (
                    "commit-tree",
                    tree,
                    "-p",
                    current,
                    "-p",
                    item.head_commit,
                    "-m",
                    message,
                )
            )
            if commit.returncode != 0:
                raise WorkspaceLifecycleError(self._failure(commit))
            current = commit.stdout.strip()
            if not _COMMIT.fullmatch(current):
                raise WorkspaceLifecycleError("commit-tree did not return a commit SHA")
        return _IntegrationDerivation(current, ())

    def consolidate(
        self, provenance: Sequence[WorkspaceProvenance]
    ) -> ConsolidationResult:
        self._revalidate_roots()
        ordered = self._ordered(provenance)
        for item in ordered:
            self._verify_provenance_against_git(item)
        derivation = self._derive_integration_commit(ordered)
        if derivation.commit is None:
            return ConsolidationResult(
                "conflict", None, None, ordered, derivation.conflicts
            )
        branch = self._integration_branch(ordered)
        reference = f"refs/heads/{branch}"
        self._revalidate_roots()
        created = self._git.run(
            ("update-ref", reference, derivation.commit, _ZERO_COMMIT)
        )
        if created.returncode != 0:
            existing = self._resolve_ref(reference)
            if existing != derivation.commit:
                raise WorkspaceLifecycleError(
                    "existing integration ref does not match deterministic integration commit"
                )
            return ConsolidationResult("already_exists", branch, existing, ordered, ())
        return ConsolidationResult(
            "consolidated", branch, derivation.commit, ordered, ()
        )

    def cleanup(
        self,
        workspace: ExperimentWorkspace,
        provenance: WorkspaceProvenance,
        consolidation: ConsolidationResult | None,
    ) -> WorkspaceCleanupResult:
        self._revalidate_roots()
        self._validate_workspace(workspace)
        expected = (
            workspace.experiment_id,
            workspace.protocol_id,
            workspace.protocol_digest,
            workspace.lane,
            workspace.branch,
            workspace.base_commit,
        )
        actual = (
            provenance.experiment_id,
            provenance.protocol_id,
            provenance.protocol_digest,
            provenance.lane,
            provenance.branch,
            provenance.base_commit,
        )
        if expected != actual:
            return WorkspaceCleanupResult(
                "retained", True, "workspace provenance does not match"
            )
        if consolidation is None:
            return WorkspaceCleanupResult(
                "retained", True, "workspace has no verified consolidation"
            )
        if consolidation.status not in {"consolidated", "already_exists"}:
            return WorkspaceCleanupResult(
                "retained", True, "consolidation is not successful"
            )
        if provenance not in consolidation.source_provenance:
            return WorkspaceCleanupResult(
                "retained", True, "consolidation does not include workspace provenance"
            )
        try:
            ordered = self._ordered(consolidation.source_provenance)
            expected_branch = self._integration_branch(ordered)
            if (
                consolidation.integration_branch != expected_branch
                or consolidation.integration_commit is None
            ):
                return WorkspaceCleanupResult(
                    "retained", True, "consolidation identity does not match sources"
                )
            for item in ordered:
                self._verify_provenance_against_git(item)
            derivation = self._derive_integration_commit(ordered)
            if derivation.commit is None:
                return WorkspaceCleanupResult(
                    "retained", True, "source provenance no longer merges cleanly"
                )
            if derivation.commit != consolidation.integration_commit:
                return WorkspaceCleanupResult(
                    "retained", True, "integration commit does not match sources"
                )
            integration_ref = f"refs/heads/{consolidation.integration_branch}"
            if self._resolve_ref(integration_ref) != consolidation.integration_commit:
                return WorkspaceCleanupResult(
                    "retained", True, "integration ref differs from consolidation"
                )
        except WorkspaceLifecycleError:
            return WorkspaceCleanupResult(
                "retained", True, "integration ref or source provenance is stale"
            )
        if not workspace.path.is_dir() or not self._registered(workspace.path):
            return WorkspaceCleanupResult(
                "retained", True, "workspace is not registered"
            )
        status = self._require(("status", "--porcelain"), cwd=workspace.path).stdout
        if status:
            return WorkspaceCleanupResult("retained", True, "workspace is dirty")
        head = self._require(
            ("rev-parse", "--verify", "HEAD^{commit}"), cwd=workspace.path
        ).stdout.strip()
        if head != provenance.head_commit:
            return WorkspaceCleanupResult(
                "retained", True, "workspace HEAD differs from provenance"
            )
        try:
            self._verify_source_ref(provenance)
        except WorkspaceLifecycleError:
            return WorkspaceCleanupResult(
                "retained", True, "source branch differs from provenance"
            )
        self._revalidate_roots()
        try:
            if self._resolve_ref(integration_ref) != consolidation.integration_commit:
                return WorkspaceCleanupResult(
                    "retained", True, "integration ref differs from consolidation"
                )
            self._verify_source_ref(provenance)
        except WorkspaceLifecycleError:
            return WorkspaceCleanupResult(
                "retained", True, "integration ref or source branch is stale"
            )
        self._require(("worktree", "remove", str(workspace.path)))
        return WorkspaceCleanupResult(
            "removed", True, "clean workspace removed; branch retained"
        )
