"""Deterministic input generation.

ZONE A -- IMMUTABLE. Do not edit as part of an optimization step.

Two rules encoded here, both of which exist because someone gamed a benchmark:

  1. CORRECTNESS_SHAPES and BENCHMARK_SHAPES are DISJOINT. If you tune against the shapes
     you validate on, you have measured your own tail. Correctness sizes are deliberately
     off-by-one around powers of two (127/128/129) to catch masking and tail bugs.

  2. Four distributions minimum, per correctness case. Uniform [0,1) inputs alone let a
     kernel hardcode "all values are positive" -- the canonical case being a ReLU kernel
     that returns its input unchanged and reported 374x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class Shape:
    B: int          # batch
    N: int          # sequence length
    H: int          # query heads
    D: int          # head dim
    H_kv: int = 0   # kv heads; 0 means == H (MHA). Set for GQA/MQA.
    causal: bool = False
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.H_kv == 0:
            object.__setattr__(self, "H_kv", self.H)

    @property
    def model_dim(self) -> int:
        return self.H * self.D

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per byte for the attention core: 4*N^2*D flops over 4*N*D*s bytes."""
        s = 2 if self.dtype in ("bfloat16", "float16") else 4
        return self.N / s

    def grid_ctas(self, block_m: int = 128) -> int:
        return self.B * self.H * -(-self.N // block_m)

    def key(self) -> str:
        return (f"B{self.B}_N{self.N}_H{self.H}_D{self.D}_"
                f"Hkv{self.H_kv}_{'causal' if self.causal else 'full'}_{self.dtype}")


# --------------------------------------------------------------------------------------
# Shape sets. DISJOINT BY CONSTRUCTION -- there is an assertion at the bottom of the file.
#
# TODO(agent): replace BENCHMARK_SHAPES with the actual announced competition matrix as
# soon as you have it. Keep the off-by-one correctness shapes regardless; they catch a
# class of bug the benchmark matrix never will.
# --------------------------------------------------------------------------------------

CORRECTNESS_SHAPES: tuple[Shape, ...] = (
    # off-by-one around powers of two: masking and tail bugs live here
    Shape(B=2, N=127, H=4, D=64),
    Shape(B=2, N=128, H=4, D=64),
    Shape(B=2, N=129, H=4, D=64),
    Shape(B=1, N=255, H=8, D=64, causal=True),
    Shape(B=1, N=257, H=8, D=64, causal=True),
    # awkward head dims
    Shape(B=2, N=200, H=3, D=32),
    Shape(B=2, N=200, H=5, D=128),
    # GQA: few kv heads is the occupancy-collapse regime
    Shape(B=1, N=513, H=8, D=128, H_kv=2),
    # single-token decode
    Shape(B=1, N=1, H=8, D=128, H_kv=2),
)

BENCHMARK_SHAPES: tuple[Shape, ...] = (
    Shape(B=1, N=128, H=8, D=64),                       # launch-bound
    Shape(B=32, N=512, H=16, D=64),                     # the ridge-straddling one
    Shape(B=8, N=4096, H=32, D=128, causal=True),       # long-context prefill
    Shape(B=1, N=512, H=8, D=128, H_kv=2),              # occupancy-bound decode-ish
    Shape(B=64, N=256, H=12, D=64),                     # wide and short
)

_c = {s.key() for s in CORRECTNESS_SHAPES}
_b = {s.key() for s in BENCHMARK_SHAPES}
assert not (_c & _b), f"correctness and benchmark shapes must be disjoint; overlap: {_c & _b}"


# --------------------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------------------

DISTRIBUTIONS = ("standard", "scaled_up", "scaled_down", "negated")

_ADVERSARIAL = ("denormal", "near_overflow", "noncontiguous", "with_nan", "with_inf")


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def generate(shape: Shape, seed: int, distribution: str = "standard",
             device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (q, k, v). Deterministic in (shape, seed, distribution).

    An explicit Generator is used rather than the global RNG so that generating inputs
    for one case cannot perturb another, and so a rerun reproduces exactly.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    dt = _dtype(shape.dtype)

    def rand(h: int) -> torch.Tensor:
        return torch.randn(shape.B, h, shape.N, shape.D,
                           generator=g, device=device, dtype=torch.float32)

    q, k, v = rand(shape.H), rand(shape.H_kv), rand(shape.H_kv)

    if distribution == "standard":
        pass
    elif distribution == "scaled_up":
        q, k, v = q * 3.0, k * 3.0, v * 3.0
    elif distribution == "scaled_down":
        q, k, v = q * 0.01, k * 0.01, v * 0.01
    elif distribution == "negated":
        # The distribution that catches "return the input unchanged" and every other
        # kernel that assumes non-negative values.
        q, k, v = -q, -k, -v
    elif distribution == "denormal":
        tiny = torch.finfo(dt).tiny
        q, k, v = q * tiny, k * tiny, v * tiny
    elif distribution == "near_overflow":
        big = torch.finfo(dt).max ** 0.5
        q, k, v = q * big * 0.5, k * big * 0.5, v
    elif distribution == "with_nan":
        q = q.clone()
        q[0, 0, 0, 0] = float("nan")
    elif distribution == "with_inf":
        k = k.clone()
        k[0, 0, 0, 0] = float("inf")
    elif distribution == "noncontiguous":
        # A transposed view. Kernels that assume contiguity fail here, and they should:
        # real callers pass views constantly.
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        k = k.transpose(1, 2).contiguous().transpose(1, 2)
    else:
        raise ValueError(f"unknown distribution {distribution!r}")

    return q.to(dt), k.to(dt), v.to(dt)


def correctness_suite(shape: Shape, seed: int = 4242,
                      include_adversarial: bool = True) -> dict[str, tuple]:
    """All distributions for one shape, as the mapping check_all() expects."""
    names = list(DISTRIBUTIONS) + (list(_ADVERSARIAL) if include_adversarial else [])
    return {n: generate(shape, seed + i * 13, n) for i, n in enumerate(names)}


def iter_correctness_cases(seed: int = 4242) -> Iterator[tuple[Shape, dict]]:
    for i, s in enumerate(CORRECTNESS_SHAPES):
        yield s, correctness_suite(s, seed + i * 101)


# --------------------------------------------------------------------------------------
# The adversarial pool (Tier 2). Grows from near-misses; see specs/05-critic.md.
# Append-only, like the ledger: a case that once caught a bug is kept forever.
# --------------------------------------------------------------------------------------

ADVERSARIAL_POOL_PATH = "ledger/adversarial_pool.jsonl"


def append_adversarial_case(shape: Shape, seed: int, distribution: str,
                            reason: str, path: str = ADVERSARIAL_POOL_PATH) -> None:
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "shape": shape.__dict__, "seed": seed,
            "distribution": distribution, "reason": reason,
        }) + "\n")
