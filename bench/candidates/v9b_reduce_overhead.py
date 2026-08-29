"""Candidate v9b -- the same idea as v9a, with the cheaper compiler mode.

Generation 9, sibling B. Parent: v8_padfast. Branch: cand/g9/reduce-overhead.

A true sibling of v9a: same parent, same hypothesis family (hand the decomposition to
Inductor), one variable changed. v9a uses `mode="max-autotune"`, which autotunes GEMM
tilings and costs 2-19s of compilation per shape. This uses `mode="reduce-overhead"`,
which skips autotuning and leans on CUDA-graph capture instead.

WHAT THE COMPARISON ANSWERS. The matrix has 13 shapes and a submission pays compilation
for each. If reduce-overhead lands within the 3% noise floor of max-autotune, the
autotuning is buying nothing here and the cheaper mode is strictly better for a graded
run. If max-autotune wins clearly, the compile cost is real work and must be reported as
part of the cost of the approach.

Either answer is useful, which is what makes this worth a sibling rather than a guess.
"""

from __future__ import annotations

import torch

from .v8_padfast import build as build_v8


def build(baseline_cls):
    v8_cls = build_v8(baseline_cls)

    class CandidateV9B(v8_cls):
        use_graph = False          # reduce-overhead applies CUDA graphs itself

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if not hasattr(self, "_compiled"):
                self._compiled = torch.compile(
                    self._core, mode="reduce-overhead", dynamic=False)
            return self._compiled(x, valid_token_mask)

    return CandidateV9B
