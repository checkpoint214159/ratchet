"""Public evaluation contracts for the authoritative transformer workload."""

from __future__ import annotations

from dataclasses import dataclass


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class TransformerConfiguration:
    """Torch-free description of one authoritative transformer invocation."""

    batch_size: int
    sequence_length: int
    model_width: int
    head_count: int
    feed_forward_width: int
    layer_count: int
    causal: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("sequence_length", self.sequence_length),
            ("model_width", self.model_width),
            ("head_count", self.head_count),
            ("feed_forward_width", self.feed_forward_width),
            ("layer_count", self.layer_count),
        ):
            _positive(name, value)
        if self.model_width % self.head_count:
            raise ValueError("model_width must be divisible by head_count")


@dataclass(frozen=True, slots=True)
class CorrectnessPolicy:
    """The evaluator's executable absolute-OR-relative acceptance policy."""

    absolute_tolerance: float
    relative_tolerance: float

    def __post_init__(self) -> None:
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("correctness tolerances must be non-negative")


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    """Evaluator-owned outcome that gates any later measurement stage."""

    passed: bool
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if not self.passed and not self.diagnostic:
            raise ValueError("failed correctness requires a diagnostic")


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """A deterministic correctness or timing case owned by evaluation."""

    case_id: str
    configuration: TransformerConfiguration
    dtype: str
    seed: int
    padding_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must not be empty")
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        if not 0.0 <= self.padding_ratio < 1.0:
            raise ValueError("padding_ratio must be in [0, 1)")
