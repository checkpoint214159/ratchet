"""Public contracts for immutable experiment evidence."""

from .archive import FileExperimentArchive
from .schema import (
    EMPIRICAL_EVENT,
    NO_RUN_EVENT,
    SCHEMA_VERSION,
    ArchiveIntegrityError,
    CatalogueProjection,
    EnvironmentArtifact,
    EnvironmentId,
    EventId,
    ExperimentCatalogue,
    ExperimentEvent,
    ExperimentId,
    validate_event_payload,
    validate_provisional_environment,
)
from .workspaces import (
    ConsolidationResult,
    ExperimentWorkspace,
    ExperimentWorkspaceLifecycle,
    ExperimentWorkspaceManager,
    ExperimentWorkspaceSpec,
    WorkspaceCleanupResult,
    WorkspaceLifecycleError,
    WorkspaceProvenance,
)

__all__ = [
    "ArchiveIntegrityError",
    "CatalogueProjection",
    "ConsolidationResult",
    "EMPIRICAL_EVENT",
    "EnvironmentArtifact",
    "EnvironmentId",
    "EventId",
    "ExperimentCatalogue",
    "ExperimentEvent",
    "ExperimentId",
    "ExperimentWorkspace",
    "ExperimentWorkspaceLifecycle",
    "ExperimentWorkspaceManager",
    "ExperimentWorkspaceSpec",
    "FileExperimentArchive",
    "NO_RUN_EVENT",
    "SCHEMA_VERSION",
    "WorkspaceCleanupResult",
    "WorkspaceLifecycleError",
    "WorkspaceProvenance",
    "validate_event_payload",
    "validate_provisional_environment",
]
