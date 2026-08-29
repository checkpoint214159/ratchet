"""Candidate v18 -- make graph capture depend on our code, not on the caller's habits.

Generation 18. Parent: v17_dispatched_megakernel. Branch: cand/g18/capture-insurance.
Idea from proposal D-06; the trigger was isolated by research agent D and reproduced here.

THE DEFECT
----------
v13's `_try_capture` (inherited by v14-v17) settles Dynamo by calling the compiled core
before capturing. If the input tensor was NOT allocated inside `torch.inference_mode()`,
Dynamo re-traces *inside* the capture region, touches the RNG, and the capture is
rejected. v13 then does exactly what it was designed to do -- degrade to the compiled
callable, slower and correct.

What nobody checked is WHEN it degrades. Reproduced here, one variable held against
everything else:

    x allocated INSIDE  torch.inference_mode()   0.2673 ms   graph_verified=True
    x allocated OUTSIDE torch.inference_mode()   0.6008 ms   graph_verified=False
                                                 ^^^^^^ 2.25x slower

Now the graded harness, `benchmarks/reference/torch_transformer_benchmark.py:529`:
**it generates the timing input OUTSIDE `inference_mode`.** We are fast today only
because `run_accuracy_tests` runs first, allocates inside it, and captures the graph
there -- so by the time the timing loop starts the graph already exists.

That is L24 restated for speed instead of correctness: *fast because of how the harness
happens to call us* is not fast. Reorder the harness, run timing alone, or call the model
from a different driver, and the frontier quietly loses more than 2x with every test still
passing.

THE FIX
-------
Do the warm-up and capture on an input WE allocated under `inference_mode`, so the
capture path stops depending on the caller. Two lines of substance.

AND REPORT IT. `capture_source` records whether the graph came from the caller's tensor,
from our insurance clone, or not at all. A silent degradation is precisely what hid this
for six generations (L36: a guard that cannot be observed cannot be trusted), so the
degradation is now observable and the tests assert on it.

NUMERICS ARE UNCHANGED. The graph replays the same compiled callable over the same static
buffers; the finding-17 verification against a freshly computed reference is untouched and
still runs. This candidate cannot be more or less accurate than its parent.
"""

from __future__ import annotations

import torch

from .v17_dispatched_megakernel import build as build_v17


def build(baseline_cls):
    v17_cls = build_v17(baseline_cls)

    class CandidateV18(v17_cls):
        capture_source: str = "none"

        def _try_capture(self, x, mask):
            """Capture from an input allocated under inference_mode, whatever the caller did.

            Tried in the caller's own context first: if that works there is nothing to
            insure against, and we avoid changing a path that already succeeds.
            """
            if super()._try_capture(x, mask):
                self.capture_source = "caller"
                return True

            # The caller's tensor was not capture-friendly. Build one that is.
            try:
                with torch.inference_mode():
                    x_ins = x.clone()
                    m_ins = None if mask is None else mask.clone()
            except Exception:
                self.capture_source = "none"
                return False

            self._capture_attempted = False        # the parent latches this; unlatch to retry
            with torch.inference_mode():
                ok = super()._try_capture(x_ins, m_ins)
            self.capture_source = "insurance" if ok else "none"
            return ok

    return CandidateV18
