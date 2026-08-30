"""Candidate v38 -- residency is ATTEMPTED, not estimated. Streaming is what happens
after an `OutOfMemoryError`, not what happens after a guess.

Generation 38. Parent: `v37_recombined2`. Branch: `cand/g38/stream-fallback`.

THE DEFECT THIS CLOSES
----------------------
`v33_streamed_long`'s docstring claims `choose()` returns "resident" on all thirteen
runnable configs, so the streaming layer costs the frontier nothing. **On config 6 --
83% of the matrix's wall time -- that claim is false**, and it has been false in every
sweep of every candidate on the streaming line:

    v26_causal_correct    57.437 ms      v34_launch_bound    62.085 ms
    v33_streamed_long     92.178 ms      v36_gemm_gelu       59.076 ms
    v37_recombined2       90.974 ms

The baseline arms of those five rows agree within 1.5%, so it is not drift. The split is
by LINEAGE: everything with v33's dispatch in it is 1.6x slower on config 6.

WHY, MEASURED
(`bench/probes/g38_stream_fallback/probe_decision_moment.py`, from a live run)
-----------------------------------------------------------------------------
`_decide_stream` reads `torch.cuda.mem_get_info` at the FIRST FORWARD and latches the
answer for that shape. The first forward is the harness's correctness check, and
`bench/run_matrix.py` runs correctness with BOTH models resident -- after the baseline
arm has been timed. So the decision is taken at the one moment in the run when memory is
scarcest, and it is then latched into a timed phase where memory is plentiful:

    [after baseline arm timed            ] free  3.69 GiB  reserved 11.00  allocated 0.62
    [both models resident, pre-forward   ] free  3.69 GiB  reserved 11.00  allocated 0.62
    [+ reference output (the real moment)] free  2.75 GiB  reserved 11.61  allocated 1.85
    DECISION AT FIRST FORWARD: streamed
      streamed: working set 3.66 GiB vs 0.35 x 2.75 GiB free; slice=2627
    [baselines freed, timing phase       ] free 13.27 GiB  reserved  1.40  allocated 0.69
    DECISION STILL LATCHED AS: streamed
    candidate 90.956 ms

and `choose()` evaluated on the SAME shape at the SAME moment the timing loop runs
returns ("resident", True). The predicate is not wrong about memory; it is asked at the
wrong time and never asked again.

TWO THINGS ARE WRONG WITH THE PREDICATE ITSELF, AND ONLY ONE IS THE TIMING
--------------------------------------------------------------------------
1. `estimate_working_set_bytes` returns `activation * 6` = 3.66 GiB for config 6. The
   candidate's MEASURED peak on that config is 2656 MB = 2.47 GiB, and that figure
   includes the input, the output and the CUDA graph's static buffers. The estimate is
   ~1.5x pessimistic, and `RESIDENT_BUDGET = 0.35` multiplies the pessimism by another
   2.9x: to run resident, config 6 must see 10.46 GiB free to use 2.47.

2. **`mem_get_info` is the wrong denominator entirely.** It reports memory free ON THE
   DEVICE. At the moment of the decision this process's own caching allocator had
   *reserved* 11.61 GiB and *allocated* only 1.85 of it -- nearly 10 GiB of blocks that
   the resident forward can have for free, without a single `cudaMalloc`, and that
   `mem_get_info` counts as unavailable. The predicate therefore understates the memory
   a resident forward can actually use by roughly the amount the process itself has
   already cached, which on this harness is most of the card.

No choice of `RESIDENT_BUDGET` fixes (2). Tuning that constant until config 6 passes
would be fitting one number to one row of a fourteen-row matrix, and it would still be
asking a question whose denominator does not mean what the caller thinks it means.

THE FIX: ASK THE ALLOCATOR, NOT AN ESTIMATE
-------------------------------------------
The exact test of "does this fit" is to try it. So:

    TRY RESIDENT. On `torch.cuda.OutOfMemoryError`, release the partial state, empty the
    cache, size a slice from the memory that is free AFTER the release, and stream.

This has no coefficient in it and therefore nothing to calibrate. It cannot be
mispredicted in either direction: a shape that fits runs resident because it fit, and a
shape that does not fit streams because it did not. It also self-corrects across the
memory pressure that caused the defect -- residency is judged by the allocator that will
serve it, at the moment it is asked.

AND THE SLICE IS ATTEMPTED TOO, BECAUSE THE FIRST DRAFT OF THIS FILE WAS INERT
-------------------------------------------------------------------------------
The first version of the fallback was watched failing, and it failed in the shape [L36]
describes. `bench/probes/g38_stream_fallback/probe_real_oom.py` caps the caching allocator
with `set_per_process_memory_fraction` so it refuses for real, through its own code path.
The fallback fired -- `stream_fallbacks == 1`, `stream_basis == "oom_fallback"`, every
kernel decision re-settled -- and then **OOMed again**, because the slice was still being
sized from `mem_get_info`, which under the cap reported 13.94 GiB free against a 534 MiB
budget, and `slice = 256` for a batch of 256. A slice equal to the batch is not a smaller
computation; it is the identical one. So the same argument that replaced the path
predicate now applies to the slice: it is bounded below the batch whenever this candidate
commits to streaming, and `_streamed_forward` HALVES and retries on an allocator refusal,
down to one row, at which point the OOM is re-raised with its own traceback because the
shape genuinely does not fit this device.

`torch.cuda.OutOfMemoryError` SPECIFICALLY, AND NOTHING BROADER
---------------------------------------------------------------
The catch is narrow on purpose. `torch.cuda.OutOfMemoryError` is raised by PyTorch's
caching allocator after it has already flushed its cache and retried, and the CUDA
context is intact afterwards -- it is a recoverable, exact signal that the memory was not
there. A bare `except Exception` would convert every bug in the stack below into a silent
switch to a slower path that returns an answer, which is the silent-wrong-answer shape
[L23] and [L25] catalogue. A raw driver `CUDA error: out of memory` (`AcceleratorError`)
is deliberately NOT caught: it poisons the context, so "recovering" from it would produce
numbers from a broken device.

THE ONE PRE-CHECK THAT SURVIVES, AND WHY IT IS NOT AN ESTIMATE
---------------------------------------------------------------
Attempting residency is reckless exactly where it cannot possibly succeed and where
failing costs more than the attempt saves. That is not a working-set estimate; it is the
SIGNATURE FLOOR:

    signature_floor_bytes(B, S, d, elem) = 2 * B * S * d * elem   > total_memory

`forward(x) -> y` holds both tensors at once and no implementation removes either one
(returning a mutated view of the input is the [L25] defect). Config 14's full batch is
24.42 GiB of input and output against a 15.99 GiB card: 1.53x over, on two tensors, with
no coefficient. Config 6 is 1.22 GiB against the same card and is nowhere near it.
`total_memory` is a measured device property, so this is evaluable on a card nobody here
has seen -- no config id appears (CLAUDE.md rule 2), and the same predicate keeps config
14 streaming on this box while letting an 80 GiB card try it resident.

WHAT DOES NOT CHANGE
--------------------
* v37's derived shape-latch reset (`SHAPE_LATCHED`, `shape_latched_over`) is inherited
  untouched, and the fallback uses it -- `_invalidate_shape_state` is exactly the routine
  that has to run between a half-finished resident forward and a streamed one.
* The streamed loop, the slice sizing, `_settle_slice_decisions`, the non-causal guard
  and v26's causal path are v33's and v37's, by inheritance. One copy ([L14]).
* Config 14's capability result is untouched: its per-sequence path runs at B=1, where
  the floor is 0.82 GiB and residency is attempted and succeeds -- which is what v33
  already did there (`stream_path: resident, slice=1` in every config-14 ledger row).
  Only the full-batch attempt streams, and it still does.

HONEST LIMITS
-------------
* An OOM inside `_try_capture` is swallowed by v13's own guard, so a shape that fits
  eagerly but cannot afford graph capture stays resident and un-captured rather than
  falling back. That is v13's designed behaviour and this candidate does not change it.
* The fallback is latched per shape. It is not retried on a later call at the same shape
  even if memory has since been freed, because paying an OOM and an allocator flush on
  every call of a timed loop would cost more than the streamed path it avoids.
* No new kernel, so no new speed argument ([L33]). Against v37 this is a null everywhere
  the resident path was already being taken, and the whole claim is config 6.
"""

from __future__ import annotations

import gc

import torch

from .v14_dispatch import (RESIDENT_BUDGET, estimate_working_set_bytes,
                           signature_floor_bytes)
from .v37_recombined2 import build as build_v37


def _first_line(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:180]}"


def build_on(base_cls):
    """The fallback dispatch, applied to whatever class is handed in.

    Factored the way v33's streaming layer was, so a later recombination can stack this
    on a sibling of v37 without a second copy of the decision existing.
    """

    class CandidateV38(base_cls):
        # Observable, so a test can assert the MECHANISM engaged before it asserts
        # anything about the answer ([L36]). `stream_basis` says WHICH of the three
        # routes produced `stream_path`, which `stream_path` alone cannot distinguish.
        stream_basis: str = "undecided"
        stream_attempted_resident: bool = False
        # CUMULATIVE over the model's life and deliberately not reset on a shape change:
        # "this model has fallen back N times" is diagnostics, not shape-latched state.
        stream_fallbacks: int = 0
        stream_narrowings: int = 0

        def _decide_stream(self, x):
            """Replaces v33's `choose()` call. Two outcomes, no estimate in either.

            The floor branch is a statement about the function signature and the device's
            total memory. The other branch is not a prediction at all -- it is
            "provisionally resident, pending the attempt".
            """
            free, total = torch.cuda.mem_get_info(x.device)
            b, s, d = x.shape
            floor = signature_floor_bytes(b, s, d, x.element_size())
            self._decided_for = (b, s, d)
            # Per SHAPE, unlike the counters above: the question it answers is "was
            # residency attempted for the shape currently latched".
            self.stream_attempted_resident = False

            if floor > total:
                self.stream_path, self.stream_is_tuned = "streamed", False
                self.stream_basis = "signature_floor"
                self._size_slice(x, free, below_batch=True)
                self.stream_reason = (
                    f"streamed without attempting resident: forward(x) -> y holds "
                    f"{floor / 2**30:.2f} GiB of input+output against "
                    f"{total / 2**30:.2f} GiB of device memory, so no implementation "
                    f"fits this shape; slice={self.stream_slice}")
                return

            self.stream_path, self.stream_is_tuned = "resident", True
            self.stream_basis = "attempt"
            self._size_slice(x, free, below_batch=False)
            self.stream_reason = (
                f"resident, attempted: input+output floor {floor / 2**30:.2f} GiB fits "
                f"{total / 2**30:.2f} GiB total; whether the working set fits is decided "
                f"by the allocator, not by an estimate ({free / 2**30:.2f} GiB reported "
                f"free, which excludes this process's own cache)")

        def _size_slice(self, x, free_bytes, below_batch: bool):
            """How many rows a streamed slice carries. A SIZING heuristic, not a path
            predicate: getting it wrong costs a longer or shorter loop and nothing else,
            which is why the conservative `RESIDENT_BUDGET` fraction is still the right
            tool here and was the wrong tool for the decision above.

            `below_batch` is not a refinement of the heuristic, it is a CORRECTNESS
            bound on it. Whenever this candidate commits to streaming it is because the
            whole batch has been shown not to fit, so a slice equal to the batch is not
            a smaller computation -- it is the identical one, and it will fail the
            identical way. Measured: without this bound the first real allocator OOM this
            candidate saw fell back, computed `slice = 256` for a batch of 256, and OOMed
            again with `stream_fallbacks == 1` and a mechanism that had done nothing
            ([L36]). v33's own test states the same invariant from the outside --
            "streaming with a full-batch slice is not streaming".
            """
            b, s, d = x.shape
            per_row = max(1, estimate_working_set_bytes(
                1, s, d, self.layers[0].attention.num_heads, len(self.layers),
                x.element_size()))
            rows = int(free_bytes * RESIDENT_BUDGET // per_row)
            if below_batch:
                rows = min(rows, b // 2)
            self.stream_slice = max(1, min(b, rows))

        def _resident_forward(self, x, valid_token_mask):
            self.stream_attempted_resident = True
            oom = None
            try:
                return super()._resident_forward(x, valid_token_mask)
            except torch.cuda.OutOfMemoryError as exc:
                # Keep the MESSAGE, not the exception. Recovery deliberately happens
                # AFTER this block: while `exc` is bound its traceback holds every frame
                # of the failed forward alive, and with them every intermediate tensor
                # that made it fail -- so freeing here would free nothing.
                oom = _first_line(exc)

            self._fall_back_to_streamed(x, valid_token_mask, oom)
            return self._streamed_forward(x, valid_token_mask)

        def _streamed_forward(self, x, valid_token_mask):
            """v33's loop, with the slice HALVED and retried on an allocator refusal.

            The slice is sized from `mem_get_info`, and finding this candidate's own
            defect established that that number is not what the process can have -- so
            the first slice is a guess like any other and the same argument that replaced
            the path predicate applies to it. Halving is exact in the only sense that
            matters here: each retry asks for strictly less, and the loop terminates
            because a slice of 1 row is the smallest question there is. At a slice of 1
            the OOM is re-raised with its own traceback rather than swallowed -- the shape
            genuinely does not fit this device, and saying so is the honest answer.
            """
            while True:
                oom = None
                try:
                    return super()._streamed_forward(x, valid_token_mask)
                except torch.cuda.OutOfMemoryError as exc:
                    if self.stream_slice <= 1:
                        raise
                    oom = _first_line(exc)
                self._narrow_slice(x, valid_token_mask, oom)

        def _release_shape_state(self, valid_token_mask):
            """Give back what a failed attempt latched.

            v37's derived reset drops the CUDA graph and its static buffers -- the
            largest thing a half-finished resident forward leaves behind -- restores
            every shape-latched attribute v34 and v36 introduced (so the next attempt
            re-plans against the slice that will actually run), and re-primes the mask
            state.

            `gc.collect()` is between the reset and `empty_cache` and is load-bearing.
            A `CUDAGraph` hands its private memory pool back when the Python object is
            DESTROYED, and `_graph = None` only drops one reference: a capture that
            OOMed part-way leaves the object in a reference cycle through its own
            traceback and frames, so refcounting alone does not collect it and
            `empty_cache` then has nothing it is allowed to release. Measured on a
            capped allocator: 180 MiB of a 188 MiB recovery budget was still held in
            private pools after the reset, leaving too little to allocate the output
            tensor -- which no slice size can shrink.
            """
            self._invalidate_shape_state(valid_token_mask)
            gc.collect()
            torch.cuda.empty_cache()

        def _fall_back_to_streamed(self, x, valid_token_mask, oom: str):
            self.stream_path, self.stream_is_tuned = "streamed", False
            self.stream_basis = "oom_fallback"
            self.stream_fallbacks += 1
            self._release_shape_state(valid_token_mask)
            free, _total = torch.cuda.mem_get_info(x.device)
            self._size_slice(x, free, below_batch=True)
            self.stream_reason = (
                f"streamed after the resident attempt raised {oom}; "
                f"{free / 2**30:.2f} GiB free after releasing it; "
                f"slice={self.stream_slice}")

        def _narrow_slice(self, x, valid_token_mask, oom: str):
            target = max(1, self.stream_slice // 2)
            self.stream_narrowings += 1
            # AFTER the reset, never before: the reset restores class-body defaults and
            # would put `stream_slice` back to whatever this layer declares.
            self._release_shape_state(valid_token_mask)
            self.stream_slice = target
            self.stream_reason += f" | narrowed to slice={target} after {oom}"

    return CandidateV38


def build(baseline_cls):
    return build_on(build_v37(baseline_cls))
