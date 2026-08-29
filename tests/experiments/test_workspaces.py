"""Local-only behavioral tests for independent experiment worktrees."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from ratchet.experiments import (
    ConsolidationResult,
    ExperimentWorkspaceManager,
    ExperimentWorkspaceSpec,
    WorkspaceLifecycleError,
)
from ratchet.experiments.workspaces import _GitResult

_PROTOCOL_BYTES = b'{"protocol_id":"PROTO-INTEL-0001"}\n'


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def local_repository(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = tmp_path / "repository"
    worktrees = tmp_path / "worktrees"
    repository.mkdir()
    worktrees.mkdir()
    _git(repository, "init")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    protocol = repository / "research" / "protocols" / "PROTO-INTEL-0001.json"
    protocol.parent.mkdir(parents=True)
    protocol.write_bytes(_PROTOCOL_BYTES)
    _git(repository, "add", "research/protocols/PROTO-INTEL-0001.json")
    _git(repository, "commit", "-m", "protocol")
    (repository / "base.txt").write_text("base\n")
    _git(repository, "add", "base.txt")
    _git(repository, "commit", "-m", "base")
    return repository, worktrees, _git(repository, "rev-parse", "HEAD")


def _spec(experiment: str, lane: str, base: str) -> ExperimentWorkspaceSpec:
    return ExperimentWorkspaceSpec(
        experiment,
        "PROTO-INTEL-0001",
        sha256(_PROTOCOL_BYTES).hexdigest(),
        lane,
        base,
    )


def _commit(workspace: Path, name: str, contents: str) -> None:
    (workspace / name).write_text(contents)
    _git(workspace, "add", name)
    _git(workspace, "commit", "-m", f"add {name}")


def _provenance_digest(
    provenance,
    changed_paths: tuple[str, ...],
    branch: str | None = None,
) -> str:
    return sha256(
        json.dumps(
            {
                "base_commit": provenance.base_commit,
                "branch": provenance.branch if branch is None else branch,
                "changed_paths": changed_paths,
                "experiment_id": provenance.experiment_id,
                "head_commit": provenance.head_commit,
                "lane": provenance.lane,
                "protocol_digest": provenance.protocol_digest,
                "protocol_id": provenance.protocol_id,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _single_consolidation(manager, provenance):
    return manager.consolidate((provenance,))


def test_create_finalize_and_cleanup_keep_a_clean_branch(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")

    provenance = manager.finalize(workspace)
    cleanup = manager.cleanup(
        workspace, provenance, _single_consolidation(manager, provenance)
    )

    assert workspace.branch == "ratchet/experiments/exp-0001/proto-intel-0001/attention"
    assert workspace.path == worktrees / "exp-0001--proto-intel-0001--attention"
    assert provenance.branch == workspace.branch
    assert provenance.changed_paths == ("attention.txt",)
    assert cleanup.status == "removed"
    assert cleanup.branch_retained is True
    assert _git(repository, "show-ref", "--verify", f"refs/heads/{workspace.branch}")


def test_finalize_refuses_dirty_workspace(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    (workspace.path / "dirty.txt").write_text("dirty\n")

    with pytest.raises(WorkspaceLifecycleError, match="clean"):
        manager.finalize(workspace)


def test_finalize_rebinds_the_committed_protocol_digest(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    rebound = replace(workspace, protocol_digest="0" * 64)

    with pytest.raises(WorkspaceLifecycleError, match="protocol_digest"):
        manager.finalize(rebound)


def test_create_rejects_a_fabricated_protocol_digest(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)

    with pytest.raises(WorkspaceLifecycleError, match="protocol_digest"):
        manager.create(
            replace(_spec("EXP-0001", "attention", base), protocol_digest="0" * 64)
        )


def test_create_uses_committed_protocol_blob_not_worktree_edit(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    protocol = repository / "research" / "protocols" / "PROTO-INTEL-0001.json"
    protocol.write_text('{"protocol_id":"PROTO-OTHER-0001"}\n')
    manager = ExperimentWorkspaceManager(repository, worktrees)

    workspace = manager.create(_spec("EXP-0001", "attention", base))

    assert workspace.base_commit == base
    assert protocol.read_bytes() != _PROTOCOL_BYTES


def test_finalize_refuses_diff_errors(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "bad.txt", "trailing whitespace \n")

    with pytest.raises(WorkspaceLifecycleError, match="trailing whitespace"):
        manager.finalize(workspace)


def test_consolidates_disjoint_worktrees_deterministically(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    first = manager.create(_spec("EXP-0002", "norm", base))
    second = manager.create(_spec("EXP-0001", "attention", base))
    _commit(first.path, "norm.txt", "norm\n")
    _commit(second.path, "attention.txt", "attention\n")

    result = manager.consolidate((manager.finalize(first), manager.finalize(second)))

    assert result.status == "consolidated"
    assert result.integration_commit is not None
    assert [(item.experiment_id, item.lane) for item in result.source_provenance] == [
        ("EXP-0001", "attention"),
        ("EXP-0002", "norm"),
    ]
    assert _git(
        repository, "ls-tree", "-r", "--name-only", result.integration_commit
    ).splitlines() == [
        "attention.txt",
        "base.txt",
        "norm.txt",
        "research/protocols/PROTO-INTEL-0001.json",
    ]
    assert _git(repository, "rev-parse", "HEAD") == base
    repeated = manager.consolidate((manager.finalize(second), manager.finalize(first)))
    assert repeated.status == "already_exists"
    assert repeated.integration_branch == result.integration_branch
    assert repeated.integration_commit == result.integration_commit


def test_consolidation_rejects_wrong_existing_integration_ref(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    result = _single_consolidation(manager, provenance)

    _git(
        repository,
        "update-ref",
        f"refs/heads/{result.integration_branch}",
        base,
    )

    with pytest.raises(WorkspaceLifecycleError, match="does not match"):
        _single_consolidation(manager, provenance)


def test_consolidation_raises_when_failed_update_ref_has_no_resolvable_ref(
    monkeypatch, local_repository: tuple[Path, Path, str]
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    original = manager._git.run

    def fail_only_update_ref(arguments, *, cwd=None):
        if arguments[0] == "update-ref":
            return _GitResult(1, "", "simulated update failure")
        return original(arguments, cwd=cwd)

    monkeypatch.setattr(manager._git, "run", fail_only_update_ref)

    with pytest.raises(WorkspaceLifecycleError, match="Needed a single revision"):
        _single_consolidation(manager, provenance)


def test_conflict_never_creates_an_integration_ref(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    first = manager.create(_spec("EXP-0001", "attention", base))
    second = manager.create(_spec("EXP-0002", "norm", base))
    _commit(first.path, "same.txt", "first\n")
    _commit(second.path, "same.txt", "second\n")

    result = manager.consolidate((manager.finalize(first), manager.finalize(second)))

    assert result.status == "conflict"
    assert result.integration_branch is None
    assert result.integration_commit is None
    assert "ratchet/experiments/integration" not in _git(
        repository, "show-ref", "--heads"
    )


def test_conflict_paths_are_validated_sorted_and_unique():
    assert ExperimentWorkspaceManager._conflict_paths("z.txt\0a.txt\0z.txt\0") == (
        "a.txt",
        "z.txt",
    )
    with pytest.raises(WorkspaceLifecycleError, match="invalid repository path"):
        ExperimentWorkspaceManager._conflict_paths("../escape\0")


def test_multiple_conflicts_are_reported_as_sorted_unique_repository_paths(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    first = manager.create(_spec("EXP-0001", "attention", base))
    second = manager.create(_spec("EXP-0002", "norm", base))
    _commit(first.path, "z.txt", "first z\n")
    _commit(first.path, "a.txt", "first a\n")
    _commit(second.path, "a.txt", "second a\n")
    _commit(second.path, "z.txt", "second z\n")

    result = manager.consolidate((manager.finalize(second), manager.finalize(first)))

    assert result.status == "conflict"
    assert result.conflicts == ("a.txt", "z.txt")


def test_merge_tree_errors_other_than_conflict_raise(
    monkeypatch, local_repository: tuple[Path, Path, str]
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    original = manager._git.run

    def fail_merge_tree(arguments, *, cwd=None):
        if arguments[0] == "merge-tree":
            return _GitResult(2, "", "simulated merge-tree failure")
        return original(arguments, cwd=cwd)

    monkeypatch.setattr(manager._git, "run", fail_merge_tree)

    with pytest.raises(WorkspaceLifecycleError, match="simulated merge-tree failure"):
        _single_consolidation(manager, provenance)


def test_consolidate_rejects_false_self_consistent_changed_paths_before_ref(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "actual.txt", "actual\n")
    provenance = manager.finalize(workspace)
    false_paths = ("invented.txt",)
    false = replace(
        provenance,
        changed_paths=false_paths,
        provenance_digest=_provenance_digest(provenance, false_paths),
    )

    with pytest.raises(WorkspaceLifecycleError, match="changed_paths"):
        manager.consolidate((false,))

    assert "ratchet/experiments/integration" not in _git(
        repository, "show-ref", "--heads"
    )


def test_cleanup_refuses_dirty_or_mismatched_provenance(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    consolidation = _single_consolidation(manager, provenance)
    (workspace.path / "dirty.txt").write_text("dirty\n")

    assert manager.cleanup(workspace, provenance, consolidation).status == "retained"
    assert workspace.path.exists()
    other = manager.create(_spec("EXP-0002", "norm", base))
    other_provenance = manager.finalize(other)
    assert (
        manager.cleanup(workspace, other_provenance, consolidation).reason
        == "workspace provenance does not match"
    )


def test_cleanup_requires_matching_live_successful_consolidation(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)

    assert manager.cleanup(workspace, provenance, None).reason == (
        "workspace has no verified consolidation"
    )
    conflict = ConsolidationResult(
        "conflict", None, None, (provenance,), ("attention.txt",)
    )
    assert manager.cleanup(workspace, provenance, conflict).reason == (
        "consolidation is not successful"
    )
    result = _single_consolidation(manager, provenance)
    _git(
        repository,
        "update-ref",
        f"refs/heads/{result.integration_branch}",
        base,
    )
    retained = manager.cleanup(workspace, provenance, result)

    assert retained.status == "retained"
    assert retained.reason == "integration ref differs from consolidation"
    assert workspace.path.exists()


def test_cleanup_retains_a_workspace_when_source_branch_moves(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    result = _single_consolidation(manager, provenance)
    _git(repository, "update-ref", f"refs/heads/{workspace.branch}", base)

    retained = manager.cleanup(workspace, provenance, result)

    assert retained.status == "retained"
    assert workspace.path.exists()


def test_cleanup_recomputes_integration_commit_before_removal(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    result = _single_consolidation(manager, provenance)
    _git(
        repository,
        "update-ref",
        f"refs/heads/{result.integration_branch}",
        base,
    )
    forged = replace(result, integration_commit=base)

    retained = manager.cleanup(workspace, provenance, forged)

    assert retained.status == "retained"
    assert retained.reason == "integration commit does not match sources"
    assert workspace.path.exists()


def test_cleanup_retains_a_result_that_does_not_include_the_source(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    other = manager.create(_spec("EXP-0002", "norm", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    other_provenance = manager.finalize(other)
    result = _single_consolidation(manager, provenance)
    mismatch = ConsolidationResult(
        "consolidated",
        result.integration_branch,
        result.integration_commit,
        (other_provenance,),
        (),
    )

    retained = manager.cleanup(workspace, provenance, mismatch)

    assert retained.status == "retained"
    assert retained.reason == "consolidation does not include workspace provenance"


def test_consolidate_rejects_absent_or_moved_source_branch(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    source_ref = f"refs/heads/{workspace.branch}"

    _git(repository, "update-ref", "-d", source_ref)
    with pytest.raises(WorkspaceLifecycleError, match="Needed a single revision"):
        _single_consolidation(manager, provenance)

    _git(repository, "update-ref", source_ref, base)
    with pytest.raises(WorkspaceLifecycleError, match="source branch does not match"):
        _single_consolidation(manager, provenance)


def test_provenance_rejects_mutation_or_noncanonical_digest(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    provenance = manager.finalize(workspace)

    with pytest.raises(ValueError, match="immutable"):
        replace(provenance, changed_paths=["path.txt"])
    with pytest.raises(ValueError, match="canonical"):
        replace(provenance, provenance_digest="0" * 64)
    mismatched_branch = "ratchet/experiments/exp-0001/proto-intel-0001/other"
    with pytest.raises(ValueError, match="generated branch"):
        replace(
            provenance,
            branch=mismatched_branch,
            provenance_digest=_provenance_digest(
                provenance, provenance.changed_paths, mismatched_branch
            ),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda base: ExperimentWorkspaceSpec(
            "exp-0001", "PROTO-INTEL-0001", "a" * 64, "lane", base
        ),
        lambda base: ExperimentWorkspaceSpec(
            "EXP-0001", "proto-intel-0001", "a" * 64, "lane", base
        ),
        lambda base: ExperimentWorkspaceSpec(
            "EXP-0001", "PROTO-INTEL-0001", "A" * 64, "lane", base
        ),
        lambda base: ExperimentWorkspaceSpec(
            "EXP-0001", "PROTO-INTEL-0001", "a" * 64, "Lane", base
        ),
        lambda base: ExperimentWorkspaceSpec(
            "EXP-0001", "PROTO-INTEL-0001", "a" * 64, "lane", "A" * 40
        ),
    ],
)
def test_spec_rejects_malformed_ids_and_digests(
    factory, local_repository: tuple[Path, Path, str]
):
    with pytest.raises(ValueError):
        factory(local_repository[2])


def test_root_must_be_explicit_external_nonsymlink(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, _ = local_repository
    with pytest.raises(ValueError, match="external"):
        ExperimentWorkspaceManager(repository, repository)
    with pytest.raises(ValueError, match="external"):
        ExperimentWorkspaceManager(repository, repository.parent)
    link = worktrees.parent / "worktree-link"
    link.symlink_to(worktrees, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        ExperimentWorkspaceManager(repository, link)


@pytest.mark.parametrize("operation", ("create", "finalize", "cleanup"))
def test_lifecycle_rejects_post_construction_worktree_root_substitution(
    operation: str, local_repository: tuple[Path, Path, str]
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = None
    provenance = None
    consolidation = None
    if operation != "create":
        workspace = manager.create(_spec("EXP-0001", "attention", base))
        _commit(workspace.path, "attention.txt", "isolated\n")
        provenance = manager.finalize(workspace)
        consolidation = _single_consolidation(manager, provenance)
    moved = worktrees.parent / "moved-worktrees"
    worktrees.rename(moved)
    worktrees.symlink_to(moved, target_is_directory=True)

    with pytest.raises(
        WorkspaceLifecycleError, match="worktree_root directory identity changed"
    ):
        if operation == "create":
            manager.create(_spec("EXP-0002", "norm", base))
        elif operation == "finalize":
            assert workspace is not None
            manager.finalize(workspace)
        else:
            assert workspace is not None
            assert provenance is not None
            manager.cleanup(workspace, provenance, consolidation)


def test_manager_rejects_workspace_outside_direct_child_root(
    local_repository: tuple[Path, Path, str],
):
    repository, worktrees, base = local_repository
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    misplaced = replace(workspace, path=worktrees / "nested" / workspace.path.name)

    with pytest.raises(ValueError, match="direct child"):
        manager.finalize(misplaced)


def test_subprocess_adapter_uses_literal_argv_without_shell(
    monkeypatch, local_repository: tuple[Path, Path, str]
):
    repository, worktrees, base = local_repository
    calls: list[tuple[object, object]] = []
    original = subprocess.run

    def spy(*args, **kwargs):
        if kwargs.get("check") is False:
            calls.append((args[0], kwargs.get("shell")))
        return original(*args, **kwargs)

    monkeypatch.setattr("ratchet.experiments.workspaces.subprocess.run", spy)
    manager = ExperimentWorkspaceManager(repository, worktrees)
    workspace = manager.create(_spec("EXP-0001", "attention", base))
    _commit(workspace.path, "attention.txt", "isolated\n")
    provenance = manager.finalize(workspace)
    manager.cleanup(workspace, provenance, _single_consolidation(manager, provenance))

    assert calls
    assert all(
        isinstance(command, tuple) and shell is False for command, shell in calls
    )
    assert all(command[0] == "git" and command[1] == "-C" for command, _ in calls)
    assert {command[3] for command, _ in calls if isinstance(command, tuple)} <= {
        "rev-parse",
        "cat-file",
        "show-ref",
        "status",
        "worktree",
        "merge-base",
        "diff",
        "merge-tree",
        "update-ref",
        "commit-tree",
        "show",
    }
    forbidden = {
        "checkout",
        "reset",
        "clean",
        "restore",
        "rebase",
        "fetch",
        "push",
        "remote",
        "branch",
    }
    assert not any(
        forbidden.intersection(command)
        for command, _ in calls
        if isinstance(command, tuple)
    )
    assert not any(
        "--force" in command for command, _ in calls if isinstance(command, tuple)
    )
