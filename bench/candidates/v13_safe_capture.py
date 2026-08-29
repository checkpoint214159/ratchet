"""Candidate v13 -- v12's graph capture, made fail-safe.

Generation 13. Parent: v12_graph_over_compile. Branch: cand/g13/safe-capture.

THE LATENT BUG IN v12
---------------------
v12 compiles with Dynamo and then captures the compiled callable in a static-buffer CUDA
graph. That is worth +7.9% (finding 16). But under a call pattern where Dynamo has not
fully settled before capture, it fails like this:

    UserWarning: The CUDA Graph is empty. This usually means that the graph was
    attempted to be captured on wrong device or stream.
    InternalTorchDynamoError: RuntimeError: Cannot call
    CUDAGeneratorImpl::current_seed during CUDA graph capture.

Dynamo re-traces *inside* the capture region, touches the RNG state, and the capture is
rejected. Reproduced by calling the candidate directly with a single input tensor rather
than through the harness, which happens to warm Dynamo differently.

**The dangerous half is the first line, not the second.** An exception is loud and
survivable. An *empty* graph is not: `replay()` then executes nothing, `_static_y` keeps
whatever it held from the warmup, and the candidate returns a stale tensor that is the
right shape and dtype and silently wrong. That is precisely the failure this project's own
notes flagged as the most likely source of a silent wrong answer from graph capture.

v12's measured results are sound -- the harness runs correctness before timing, and five
accuracy trials on fresh inputs would not pass against a stale buffer. But "sound because
the harness happens to call it in the right order" is not a property worth shipping.

THE FIX
-------
Capture defensively and verify, in three parts:

  1. Warm up until Dynamo stops re-tracing, by calling the compiled function on the
     DEFAULT stream first -- side-stream warmup alone does not settle it.
  2. Wrap capture in try/except and treat ANY failure as "no graph".
  3. After a nominally successful capture, VERIFY the graph is real: replay it and check
     the output against a freshly computed reference. An empty graph fails this check
     because the static buffer will not have been rewritten from the new input.

If any step fails, `_graph` stays None and every call falls through to the compiled
callable -- v11's behaviour, ~7.9% slower and correct. Degrading to slower-and-correct is
the only acceptable direction for a failure in a component whose failure mode is silence.
"""

from __future__ import annotations

import torch

from .v8_padfast import build as build_v8


def build(baseline_cls):
    v8_cls = build_v8(baseline_cls)

    class CandidateV13(v8_cls):
        use_graph = False
        graph_verified: bool = False        # observable, for tests and the report

        def _try_capture(self, x, mask):
            """Capture and verify. Returns True only if the graph is real and correct."""
            try:
                # Settle Dynamo on the DEFAULT stream. Tracing inside the capture region
                # is what triggers the RNG access that kills the capture.
                for _ in range(3):
                    self._compiled_core(x, mask)
                torch.cuda.synchronize()

                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._compiled_core(x, mask)
                torch.cuda.current_stream().wait_stream(side)

                self._static_x = x.clone()
                self._static_m = None if mask is None else mask.clone()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._static_y = self._compiled_core(self._static_x, self._static_m)
            except Exception:
                return False

            # VERIFY. An empty graph replays as a no-op, so the static output keeps its
            # warmup value; feeding a different input and comparing against a freshly
            # computed reference catches exactly that.
            try:
                probe = torch.randn_like(x)
                expected = self._compiled_core(probe, mask)
                self._static_x.copy_(probe)
                if self._static_m is not None:
                    self._static_m.copy_(mask)
                g.replay()
                if not torch.allclose(self._static_y, expected, rtol=1e-3, atol=1e-4):
                    return False
            except Exception:
                return False

            self._graph = g
            self.graph_verified = True
            return True

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)
            if not hasattr(self, "_compiled_core"):
                self._compiled_core = torch.compile(self._core, dynamic=False)
                self._graph = None
                self._capture_attempted = False

            if self._graph is None and not self._capture_attempted:
                self._capture_attempted = True          # try once, never in a loop
                self._try_capture(x, valid_token_mask)

            if self._graph is None:
                return self._compiled_core(x, valid_token_mask)   # slower, correct

            self._static_x.copy_(x)
            if self._static_m is not None:
                self._static_m.copy_(valid_token_mask)
            self._graph.replay()
            return self._static_y.clone()

    return CandidateV13
