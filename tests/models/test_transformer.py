"""Tests for OptimizedTransformer ensuring strict correctness against reference baseline."""

import copy
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from benchmarks.reference.torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    compare_outputs,
    generate_random_case,
)
from ratchet.models.transformer import (
    OptimizedSelfAttention,
    OptimizedTransformer,
    OptimizedTransformerBlock,
)


def _copy_weights(baseline: nn.Module, optimized: nn.Module) -> None:
    state_dict = copy.deepcopy(baseline.state_dict())
    optimized.load_state_dict(state_dict, strict=True)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("padding_ratio", [0.0, 0.25])
def test_optimized_self_attention_matches_baseline(causal: bool, padding_ratio: float):
    torch.manual_seed(42)
    B, S, D, H = 2, 64, 128, 4
    config = TransformerConfig(
        batch_size=B,
        seq_len=S,
        d_model=D,
        num_heads=H,
        ffn_dim=256,
        num_layers=1,
        causal=causal,
    )
    baseline_model = BaselineTransformer(config)
    optimized_model = OptimizedTransformer(
        d_model=D,
        num_heads=H,
        ffn_dim=256,
        num_layers=1,
        causal=causal,
    )
    _copy_weights(baseline_model, optimized_model)

    baseline_model.eval()
    optimized_model.eval()

    device = torch.device("cpu")
    dtype = torch.float32

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=100,
        padding_ratio=padding_ratio,
        input_scale=1.0,
    )

    with torch.inference_mode():
        ref_out = baseline_model(x, valid_mask)
        opt_out = optimized_model(x, valid_mask)

    res = compare_outputs(ref_out, opt_out, rtol=0.02, atol=0.002)
    assert res.passed, (
        f"Failed on causal={causal}, padding_ratio={padding_ratio}: "
        f"max_abs={res.max_abs_error}, max_rel={res.max_relative_error}"
    )


@pytest.mark.parametrize(
    "config",
    [
        TransformerConfig(batch_size=1, seq_len=32, d_model=64, num_heads=2, ffn_dim=128, num_layers=2, causal=False),
        TransformerConfig(batch_size=2, seq_len=127, d_model=128, num_heads=4, ffn_dim=256, num_layers=2, causal=True),
        TransformerConfig(batch_size=4, seq_len=64, d_model=256, num_heads=8, ffn_dim=512, num_layers=3, causal=False),
    ],
)
def test_full_optimized_transformer_matches_baseline_across_shapes(config: TransformerConfig):
    torch.manual_seed(123)
    baseline = BaselineTransformer(config)
    optimized = OptimizedTransformer(
        d_model=config.d_model,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        num_layers=config.num_layers,
        causal=config.causal,
    )
    _copy_weights(baseline, optimized)
    baseline.eval()
    optimized.eval()

    device = torch.device("cpu")
    for trial in range(3):
        x, valid_mask = generate_random_case(
            config=config,
            device=device,
            dtype=torch.float32,
            seed=500 + trial,
            padding_ratio=0.2 if trial > 0 else 0.0,
            input_scale=1.5,
        )
        with torch.inference_mode():
            ref = baseline(x, valid_mask)
            opt = optimized(x, valid_mask)

        result = compare_outputs(ref, opt, rtol=0.02, atol=0.002)
        assert result.passed, (
            f"Failed shape {config} trial {trial}: "
            f"max_abs={result.max_abs_error}, max_rel={result.max_relative_error}"
        )
