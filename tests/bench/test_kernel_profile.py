"""The profile recorded alongside each measurement."""
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from bench.run_matrix import kernel_profile


def _ref():
    import importlib.util, sys
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_kp", p)
    m = importlib.util.module_from_spec(spec); sys.modules["ref_kp"] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.gpu
def test_profile_attributes_time_to_named_kernels():
    m = _ref()
    cfg = m.TransformerConfig(batch_size=8, seq_len=64, d_model=128, num_heads=4,
                              ffn_dim=128, num_layers=2, causal=True)
    torch.manual_seed(0)
    model = m.BaselineTransformer(cfg).cuda().eval()
    prof = kernel_profile(model, torch.randn(8, 64, 128, device="cuda"), None)
    assert "error" not in prof, prof
    assert prof["launches"] > 0 and prof["distinct_kernels"] > 0
    assert prof["kernels"], "no kernels attributed"
    assert abs(sum(k["pct"] for k in prof["kernels"]) - 100.0) < 40.0 or prof["truncated"]
    for k in prof["kernels"]:
        assert k["name"] and k["launches"] > 0 and k["us"] >= 0


@pytest.mark.gpu
def test_profile_is_not_presented_as_a_measurement():
    """profiled_ms exists so a reader can sanity-check attribution, NOT to be compared
    across rows -- the profiler perturbs execution. The field name and the docstring must
    keep saying so, because a plausible-looking millisecond figure invites exactly that
    misuse (L41: a probe may propose, it may never conclude)."""
    src = (Path(__file__).resolve().parents[2] / "bench" / "run_matrix.py").read_text()
    assert "profiled_ms" in src
    assert "NOT the measurement" in src
    i = src.index("def kernel_profile")
    assert "AFTER" in src[i:i + 2500] or "after all timing" in src[i:i + 2500]


def test_profiling_runs_after_timing_not_during():
    """Pinned at the source: if profiling ever moves above the timing block it silently
    corrupts every speedup in the ledger."""
    src = (Path(__file__).resolve().parents[2] / "bench" / "run_matrix.py").read_text()
    timing = src.index("cand_ms = min(median_ms")
    profiling = src.index('out["profile"]')
    assert timing < profiling, "profiling must not precede or interleave with timing"


def test_profile_diff_reads_the_ledger_without_touching_the_gpu():
    """It must be safe to run while agents measure (finding 26)."""
    src = (Path(__file__).resolve().parents[2] / "bench" / "profile_diff.py").read_text()
    assert "cuda" not in src.lower() or "never touches the GPU" in src
    assert "clean_rows" in src
