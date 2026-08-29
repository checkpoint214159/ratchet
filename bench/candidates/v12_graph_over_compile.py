"""Candidate v12 -- compile for fusion, then capture the result in OUR graph.

Generation 12. Parent: v11_lean. Branch: cand/g12/graph-over-compile.

THE MEASUREMENT THAT MOTIVATED IT
---------------------------------
Profiling config 2 (B=1, the weakest config at 1.36x over the compiled baseline) found it
is **CPU-dispatch-bound, not GPU-bound**: 232 us of CPU against 126 us of GPU per call,
with the CPU side dominated by

    TorchDynamo Cache Lookup    22.5 us   <- guard evaluation, EVERY call
    cudaGraphLaunch             49.8 us

Config 2 does about 1.2 us of actual arithmetic. Everything else is the cost of deciding
what to run.

v9a/v11 disabled our static-buffer CUDA graph on the reasoning that `reduce-overhead`
captures graphs itself and stacking two mechanisms invites silent staleness. That
reasoning was sound but incomplete: Inductor's graph still sits *behind* Dynamo's guard
check, so every call re-evaluates guards before reaching the replay.

THE CHANGE
----------
Compile with the DEFAULT mode -- fusion, no Inductor cudagraphs -- and capture the
compiled callable in our own static-buffer graph. After the first call the steady-state
path is a single `graph.replay()` with no Dynamo participation at all.

This is not re-adding what v9a removed. v9a removed our graph while keeping Inductor's;
this removes Inductor's while keeping ours, so the two mechanisms still never nest.

WHY IT MIGHT NOT PAY
--------------------
Dynamo's guard cost is per-call and roughly constant, so it is only worth reclaiming where
a call is short. On config 13 (3.3 ms) 22 us is 0.7% and invisible; on config 2 (~97 us)
it is over 20%. Expect the win to concentrate entirely in the launch-bound configs and to
vanish elsewhere -- and if it does not appear even there, the guard cost is not really on
the critical path and this whole direction is closed.
"""

from __future__ import annotations

import torch

from .v8_padfast import build as build_v8


def build(baseline_cls):
    v8_cls = build_v8(baseline_cls)

    class CandidateV12(v8_cls):
        use_graph = False          # our own capture is done explicitly below

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)

            if not hasattr(self, "_compiled_core"):
                # Default mode: Inductor fuses but does NOT install its own cudagraphs,
                # so ours is the only graph mechanism in play.
                self._compiled_core = torch.compile(self._core, dynamic=False)
                self._graph = None

            if self._graph is None:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    # Extra warmup: compilation itself must complete on the side stream,
                    # and Dynamo needs to have settled on a single graph before capture.
                    for _ in range(5):
                        self._compiled_core(x, valid_token_mask)
                torch.cuda.current_stream().wait_stream(side)

                self._static_x = x.clone()
                self._static_m = (None if valid_token_mask is None
                                  else valid_token_mask.clone())
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._static_y = self._compiled_core(self._static_x, self._static_m)
                self._graph = g

            self._static_x.copy_(x)
            if self._static_m is not None:
                self._static_m.copy_(valid_token_mask)
            self._graph.replay()
            return self._static_y.clone()   # the harness holds it across replays

    return CandidateV12
