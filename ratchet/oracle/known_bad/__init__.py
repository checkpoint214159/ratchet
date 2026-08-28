"""Hand-seeded known-bad kernels -- the anchor's negative examples.

ZONE A -- IMMUTABLE once seeded, for the same reason the tolerances are: these
fixtures exist to be FAILED. They are the correctness gate's own test suite (M2:
if any of them passes, the gate is broken and nothing downstream is trustworthy)
and, later, the critic's anchor (specs/05-critic.md): a lenient critic that says
"pass" to everything must be actively penalised, which requires candidates whose
ground truth is known to be bad. Seeded by hand, before the critic exists, so
the critic can never have had a hand in choosing them.

Each fixture carries exactly one defect, chosen from the classes that
machine-generated kernels actually exhibit, and each records the gate expected
to catch it so a gate regression is detectable by name, not just by count.

The fifth seed from specs/05-critic.md -- a race condition that passes at low
occupancy -- is deferred to M3: it needs a real Triton kernel with an actual
data race to host it, and everything here is deliberately plain torch. Add it
alongside the first hand-authored Triton kernel, where the determinism gate has
something real to bite on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import identity, nan_swallower, shape_special, wrong_scale


@dataclass(frozen=True)
class KnownBadKernel:
    name: str
    kernel: Callable            # (q, k, v, causal=False) -> Tensor
    expected_gate: str          # the gate_failed value the oracle must report
    description: str


KNOWN_BAD: list[KnownBadKernel] = [
    KnownBadKernel(
        name="identity",
        kernel=identity.kernel,
        expected_gate="tolerance",
        description="returns q unchanged; the canonical 374x 'ReLU that returns "
                    "its input'",
    ),
    KnownBadKernel(
        name="wrong_scale",
        kernel=wrong_scale.kernel,
        expected_gate="tolerance",
        description="scores scaled by 1/d instead of 1/sqrt(d); invisible at "
                    "N=1 where softmax of a single score is 1 regardless",
    ),
    KnownBadKernel(
        name="shape_special",
        kernel=shape_special.kernel,
        expected_gate="tolerance",
        description="exact only at the benchmarked (32,16,512,64) q-shape; "
                    "drops the last KV token everywhere else",
    ),
    KnownBadKernel(
        name="nan_swallower",
        kernel=nan_swallower.kernel,
        expected_gate="nonfinite",
        description="correct attention followed by nan_to_num; passes every "
                    "finite distribution, launders NaN/Inf into numbers",
    ),
]
