"""Reference implementations and the baseline family.

ZONE A -- IMMUTABLE. Do not edit as part of an optimization step.

The most consequential thing in this file is `baseline_family`. Roughly half of all
published kernel speedups in 2025-2026 are an artifact of comparing against the wrong
baseline: KernelBench originally used FP32 PyTorch with TF32 DISABLED, and fixing only
that accounted for ~47% of apparent gains and dropped the best frontier model from a
reported 1.43x to 0.88x -- below PyTorch.

So the baseline here is a FAMILY, every member is timed, and the reported baseline is the
BEST of them. Anything else is not a result.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _HAS_SDPA_CTX = True
except Exception:  # pragma: no cover
    _HAS_SDPA_CTX = False


def _expand_kv(k: torch.Tensor, v: torch.Tensor, n_heads_q: int):
    """Expand GQA/MQA kv heads to match query heads. Views, not copies, where possible."""
    n_kv = k.shape[1]
    if n_kv == n_heads_q:
        return k, v
    rep = n_heads_q // n_kv
    return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)


def reference_fp64(q, k, v, causal: bool = False) -> torch.Tensor:
    """The error floor. Slow, and that is fine -- it runs on correctness shapes only.

    Everything here is done in float64 with an explicit safe softmax so that the result
    is as close to exact as the machine allows. Measure your FP32 kernel against this to
    establish the floor before you measure anything else against anything else.
    """
    q64, k64, v64 = (t.to(torch.float64) for t in (q, k, v))
    k64, v64 = _expand_kv(k64, v64, q64.shape[1])

    scale = 1.0 / math.sqrt(q64.shape[-1])
    s = (q64 @ k64.transpose(-2, -1)) * scale

    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        mask = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril(n_k - n_q)
        s = s.masked_fill(~mask, float("-inf"))

    m = s.amax(dim=-1, keepdim=True)
    p = torch.exp(s - m)
    p = p / p.sum(dim=-1, keepdim=True)
    return (p @ v64)


def reference_fp32(q, k, v, causal: bool = False) -> torch.Tensor:
    """The semantic reference the gate compares against. Same algebra, float32."""
    q32, k32, v32 = (t.to(torch.float32) for t in (q, k, v))
    k32, v32 = _expand_kv(k32, v32, q32.shape[1])

    scale = 1.0 / math.sqrt(q32.shape[-1])
    s = (q32 @ k32.transpose(-2, -1)) * scale

    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        mask = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril(n_k - n_q)
        s = s.masked_fill(~mask, float("-inf"))

    p = torch.softmax(s, dim=-1)
    return (p @ v32).to(q.dtype)


# --------------------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------------------

def _sdpa(backend) -> Callable:
    def run(q, k, v, causal=False):
        k2, v2 = _expand_kv(k, v, q.shape[1])
        if _HAS_SDPA_CTX and backend is not None:
            with sdpa_kernel([backend], set_priority=True):
                return F.scaled_dot_product_attention(q, k2, v2, is_causal=causal)
        return F.scaled_dot_product_attention(q, k2, v2, is_causal=causal)
    return run


def baseline_family() -> dict[str, Callable]:
    """Every baseline worth beating. The reported baseline is the fastest that PASSES.

    Note the explicit backend forcing. PyTorch's SDPA dispatch heuristic is known to
    choose badly (pytorch#138907 is open precisely because nobody has produced the
    shape-by-shape table), so relying on the default means quoting a speedup against a
    baseline that was itself mis-dispatched.
    """
    torch.set_float32_matmul_precision("high")  # TF32 ON. This is the important line.

    fam: dict[str, Callable] = {
        "eager_tf32": reference_fp32,
        "sdpa_default": _sdpa(None),
    }

    if _HAS_SDPA_CTX:
        for name, backend in (
            ("sdpa_flash", SDPBackend.FLASH_ATTENTION),
            ("sdpa_mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("sdpa_math", SDPBackend.MATH),
        ):
            fam[name] = _sdpa(backend)
        cudnn = getattr(SDPBackend, "CUDNN_ATTENTION", None)
        if cudnn is not None:
            fam["sdpa_cudnn"] = _sdpa(cudnn)

    try:
        fam["compile_max_autotune"] = torch.compile(
            _sdpa(None), mode="max-autotune", dynamic=False
        )
    except Exception:
        pass  # recorded as absent rather than silently substituted

    return fam


def best_baseline(timings: dict[str, float]) -> tuple[str, float]:
    """Given {name: mean_ns}, return the winner. Record the whole dict in the ledger --
    which backend wins at which shape is itself a publishable result, and nobody has
    produced that table."""
    name = min(timings, key=timings.get)
    return name, timings[name]
