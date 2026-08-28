"""Public evaluation contracts for the authoritative transformer workload."""

from __future__ import annotations

from dataclasses import dataclass

REFERENCE_BENCHMARK_PATH = "benchmarks/reference/torch_transformer_benchmark.py"
REFERENCE_BENCHMARK_SHA256 = (
    "5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e"
)


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


@dataclass(frozen=True, slots=True)
class CandidateSeamContract:
    """The sole evaluator class a future implementation may replace."""

    evaluator_path: str
    evaluator_sha256: str
    candidate_class: str
    observed_base_class: str
    forward_parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.evaluator_path != REFERENCE_BENCHMARK_PATH:
            raise ValueError("candidate seam must use the authoritative evaluator path")
        if self.evaluator_sha256 != REFERENCE_BENCHMARK_SHA256:
            raise ValueError("candidate seam must use the authoritative evaluator hash")
        if self.candidate_class != "UserOptimizedTransformer":
            raise ValueError("candidate seam class is fixed by the evaluator")
        if not self.observed_base_class:
            raise ValueError("candidate seam must record the current base class")
        if self.forward_parameters != ("x", "valid_token_mask"):
            raise ValueError("candidate seam forward signature must remain unchanged")


@dataclass(frozen=True, slots=True)
class WeightCopyContract:
    """The fair-comparison parameter compatibility required by the evaluator."""

    function_name: str
    source_state: str
    target_load: str
    strict_default: bool
    parameter_name_mismatch_handling: str

    def __post_init__(self) -> None:
        if self.function_name != "copy_model_weights":
            raise ValueError("weight copy function is fixed by the evaluator")
        if self.source_state != "baseline.state_dict":
            raise ValueError("weight copy source must be baseline state_dict")
        if self.target_load != "optimized.load_state_dict":
            raise ValueError("weight copy target must load optimized state")
        if not self.strict_default:
            raise ValueError("weight copy must default to strict compatibility")
        if self.parameter_name_mismatch_handling != "customize copy_model_weights":
            raise ValueError(
                "parameter-name differences must customize copy_model_weights"
            )


@dataclass(frozen=True, slots=True)
class MaskAndOutputContract:
    """The observable behavior a future candidate must preserve exactly."""

    valid_mask_contract: str
    causal_contract: str
    output_contract: str

    def __post_init__(self) -> None:
        if (
            self.valid_mask_contract
            != "mask invalid keys and zero invalid token outputs"
        ):
            raise ValueError("valid-mask contract is fixed by the evaluator")
        if self.causal_contract != "causal=True masks future key positions":
            raise ValueError("causal contract is fixed by the evaluator")
        if self.output_contract != "tensor [batch_size, seq_len, d_model]":
            raise ValueError("output contract is fixed by the evaluator")


@dataclass(frozen=True, slots=True)
class CandidateIntegrationContract:
    """A structural-only future integration specification, not an executable path."""

    seam: CandidateSeamContract
    weight_copy: WeightCopyContract
    mask_and_output: MaskAndOutputContract
    implementation_state: str

    def __post_init__(self) -> None:
        if self.implementation_state != "structural_only":
            raise ValueError("this build permits structural characterization only")


AUTHORITATIVE_CANDIDATE_INTEGRATION_CONTRACT = CandidateIntegrationContract(
    seam=CandidateSeamContract(
        evaluator_path=REFERENCE_BENCHMARK_PATH,
        evaluator_sha256=REFERENCE_BENCHMARK_SHA256,
        candidate_class="UserOptimizedTransformer",
        observed_base_class="BaselineTransformer",
        forward_parameters=("x", "valid_token_mask"),
    ),
    weight_copy=WeightCopyContract(
        function_name="copy_model_weights",
        source_state="baseline.state_dict",
        target_load="optimized.load_state_dict",
        strict_default=True,
        parameter_name_mismatch_handling="customize copy_model_weights",
    ),
    mask_and_output=MaskAndOutputContract(
        valid_mask_contract="mask invalid keys and zero invalid token outputs",
        causal_contract="causal=True masks future key positions",
        output_contract="tensor [batch_size, seq_len, d_model]",
    ),
    implementation_state="structural_only",
)


__all__ = [
    "AUTHORITATIVE_CANDIDATE_INTEGRATION_CONTRACT",
    "CandidateIntegrationContract",
    "CandidateSeamContract",
    "CorrectnessPolicy",
    "CorrectnessResult",
    "EvaluationCase",
    "MaskAndOutputContract",
    "REFERENCE_BENCHMARK_PATH",
    "REFERENCE_BENCHMARK_SHA256",
    "TransformerConfiguration",
    "WeightCopyContract",
]
