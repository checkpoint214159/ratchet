"""Candidate v21 -- stop paying for the graph's output copy on every call.

Generation 21. Parent: v18_capture_insurance. Branch: cand/g21/double-buffered.
From proposal B-05, narrowed: this takes ONLY the output-copy component. The proposal's
cost-benefit "should we use the graph at all" gate is a separate variable and is not
changed here.

THE DEFECT
----------
The frontier's steady state is three operations, not one::

    self._static_x.copy_(x)        # DtoD copy of the INPUT
    self._graph.replay()
    return self._static_y.clone()  # DtoD copy of the OUTPUT

A clean profile of the frontier at config 6's shape puts `Memcpy DtoD` at **6.6% of
total forward time**, split roughly evenly between those two copies. So the output copy
alone is **~3% of config 6**, ~2.5% of config 13 and ~1.3% of config 8 (proposal B-05's
per-config `Memcpy DtoD` table, halved). On the 11 small configs it is close to nothing.
**That is the whole size of the prize, and it sits inside the +/-7% noise floor (L29);
this candidate is only resolvable per-config, on the two largest shapes, and only with
replicates.** Stated here before measuring, as L29 requires.

The input copy is NOT removable: a captured graph reads from a fixed address and the
caller's tensor is somewhere else every call. Only the output copy has an alternative.

WHY THE CLONE IS THERE, AND WHY DELETING IT IS A BUG
----------------------------------------------------
`_static_y` is overwritten by the next `replay()`. Finding 24 records four candidates
(v9a, v9b, v11, v15) that returned a static buffer and had the caller's tensor rewritten
underneath them -- a silent wrong answer, in the archive for a week, with the test that
was written to catch it reporting green. The clone is the only thing standing between
this lineage and that bug. Deleting it is not an optimization, it is finding 24 again.

WHAT THIS CANDIDATE DOES INSTEAD
--------------------------------
Two mechanisms. The second is the one that makes the first honest, and the second is also
the one that does the work.

1. **Double-buffered graph exec.** The same compiled callable is captured TWICE, into two
   `graphExec`s with two distinct static output buffers, and replays alternate. The
   tensor returned at call k is then not touched by call k+1 -- it is touched by call
   **k+N**. Gated on measured free memory, and disabled unless the two buffers are proven
   independent (below).

2. **A liveness check before every clobber, because (1) alone is only safe by accident.**
   Double buffering does not make returning a static buffer correct. It moves the safety
   margin from 0 calls to N-1 calls, and N-1 is still a contract the caller never agreed
   to. It happens to cover the graded harness -- the timing loop discards the result
   immediately (reference benchmark lines 494 and 504) and the accuracy loop holds it
   exactly one call deep (line 391) -- and "fine because of how the harness calls it" is
   exactly what L24 forbids. Shipping N=2 on its own would be finding 24 with a longer
   fuse.

   So before replaying into buffer i, this candidate asks whether anything outside itself
   still refers to the tensor it handed out from buffer i:

     * nothing does  -> replay. **No copy at all.** This is the fast path, and it is what
       both timing loops (`model(x, mask)` as a bare statement, result discarded before
       the next call) and the accuracy loops actually hit.
     * the caller still holds the tensor, un-aliased -> copy the buffer's contents into
       fresh memory and rebind the caller's tensor onto it (`Tensor.set_`), so its value
       is preserved exactly. Cost: one clone -- **identical to the parent**, never worse.
     * the caller holds an alias we cannot rebind (a view, a slice) -> we cannot preserve
       it, so we do not destroy it: that buffer is **retired** from the rotation, zero-copy
       is switched off permanently, and this call is served from the compiled callable
       (v13's documented fallback, ~7.9% slower and correct). Slower-and-correct is the
       only acceptable direction for a failure in a component whose failure mode is
       silence.

   The result is a candidate that is faster when nobody is holding its output, exactly as
   expensive as its parent when somebody is, and never silently wrong.

   **Mechanism 2 is doing almost all of the work, and mechanism 1 almost none.** Both
   timing loops discard the result as a bare expression statement, so the previous handout
   is already dead when the next call starts and N=1 is enough to get the whole win. N=2
   only adds the `out = model(x)` idiom, where the previous result is still bound while
   the next call runs. That is a real caller pattern and worth having, but nothing in
   either harness exercises it. Said plainly so the ledger number is not read as evidence
   for double buffering: **the sweep measures mechanism 2.**

WHAT IS VERIFIED BEFORE THE SECOND BUFFER IS TRUSTED
----------------------------------------------------
A second capture could quietly make this candidate a no-op or a corruption, so three
things are checked at build time (L23/L36: assert the mechanism actually ran):

  * the two output buffers must have **different addresses** -- if the allocator reused
    the first one, double buffering is a costume and the safety argument is void;
  * each graph must replay **correct** values against a freshly computed reference;
  * replaying either graph must leave the OTHER graph's output buffer **bit-identical**.

That last check is not paranoia, it is a bug this candidate actually had. The first
implementation captured graph 1 into `self._graph.pool()`, so the two graphs would share
one memory pool and only the output buffer would be duplicated -- config 6 would have
cost +655 MiB instead of a second working set. MEASURED: the allocator handed capture 1's
OUTPUT an address that capture 0 uses for an intermediate, and replaying graph 0
overwrote graph 1's result. The check caught it; the pool is no longer shared. The price
is that a second buffer now costs a whole second working set, which is why the memory
gate is a ratio of measured free memory to measured reserved bytes, and why it will
decline on the largest shapes -- the ones where the copy is worth the most. That is not
a loss: mechanism 2 still runs there.

Any failure, any exception, or too little free memory leaves `buffering == "single"` and
the candidate keeps mechanism 2 alone. The reason is recorded in `buffering_reason` so
the degradation is observable rather than silent.

NUMERICS ARE UNCHANGED. The same compiled callable is replayed over the same shapes; no
arithmetic, dtype, or algorithm is touched. This candidate cannot be more or less
accurate than its parent.
"""

from __future__ import annotations

import contextlib
import sys

import torch

from .v18_capture_insurance import build as build_v18


def _storage_use_count(t: torch.Tensor) -> int:
    """Number of TensorImpls sharing `t`'s storage. 1 == nobody but `t`."""
    return torch._C._storage_Use_Count(t.untyped_storage()._cdata)


def build(baseline_cls):
    v18_cls = build_v18(baseline_cls)

    class CandidateV21(v18_cls):
        # How many graphExecs to rotate over. 2 is the smallest number that makes the
        # ubiquitous `out = model(x)` loop copy-free: the previous result is still bound
        # to the caller's name while the next call runs, so N=1 would have to preserve it
        # every time. Neither benchmark harness uses that idiom -- both discard the result
        # as a bare statement -- so N=2 buys them nothing over N=1.
        N_BUFFERS = 2
        # Free memory required before a second capture is attempted, as a multiple of what
        # this process already has reserved. Two measured byte counts -- no config id, no
        # shape literal (rule 2).
        MEM_HEADROOM = 1.5
        MEM_RESERVE_BYTES = 1 << 30

        buffering: str = "none"
        buffering_reason: str = "not built"
        # Observable counters. These are what the tests assert on, and what keeps the
        # mechanism from rotting into dead code: if the clone comes back, output_copies
        # stops being zero on the discard pattern.
        output_copies: int = 0
        zero_copy_returns: int = 0
        preserve_rebinds: int = 0
        retired_buffers: int = 0

        # ---------------------------------------------------------------- construction

        def _build_buffers(self) -> None:
            """Called once, after the parent has captured and verified graph 0."""
            self._buffers_built = True
            self._bufs = [(self._graph, self._static_y)]
            self._retired: list = []
            self._live: dict = {}
            self._rr = 0
            self._zero_copy = True
            self.buffering = "single"
            self.output_copies = 0
            self.zero_copy_returns = 0

            y0 = self._static_y
            try:
                free, _total = torch.cuda.mem_get_info(y0.device)
                held = torch.cuda.memory_reserved(y0.device)
            except Exception as exc:                          # pragma: no cover
                self.buffering_reason = f"single: mem_get_info failed ({exc!r})"
                return
            # A second capture duplicates the graph's ENTIRE working set, not just its
            # output, because a capture allocates into its own pool (see the shared-pool
            # note on `_add_buffer`). So the predicate is: could this process reserve
            # 1.5x what it already holds, and still keep a gigabyte spare? Two measured
            # byte counts, no shape literal and no config id (rule 2); on a bigger card
            # the crossover moves on its own.
            need = self.MEM_HEADROOM * held + self.MEM_RESERVE_BYTES
            if free < need:
                self.buffering_reason = (
                    f"single: {free / 2**20:.0f} MiB free < "
                    f"{self.MEM_HEADROOM:g}x{held / 2**20:.0f} MiB already reserved + "
                    f"{self.MEM_RESERVE_BYTES / 2**20:.0f} MiB reserve -- a second graph "
                    f"pool would risk the co-residency spill of finding 05")
                return

            for _ in range(self.N_BUFFERS - 1):
                ok, why = self._add_buffer()
                if not ok:
                    self.buffering_reason = f"single: {why}"
                    return
            self.buffering = "double" if len(self._bufs) == 2 else f"x{len(self._bufs)}"
            self.buffering_reason = (
                f"{len(self._bufs)} graphExecs, distinct output buffers, "
                f"mutual non-interference verified")

        def _add_buffer(self):
            """Capture the same callable again, into its OWN pool, and prove it is real.

            NOT into `self._graph.pool()`. Sharing the pool would cost only one extra
            output buffer instead of a whole working set, and it was tried first --
            MEASURED here and rejected: the allocator handed capture 1's OUTPUT an address
            that capture 0 uses for an intermediate, so replaying graph 0 overwrote
            graph 1's result. `_verify_pair` is what caught it, which is the only reason
            this is a paragraph of documentation instead of a silent wrong answer.
            """
            # The parent tells us which context the capture that worked ran in.
            ctx = (torch.inference_mode() if getattr(self, "capture_source", "") == "insurance"
                   else contextlib.nullcontext())
            try:
                with ctx:
                    # v13's recipe: settle Dynamo on the default stream, then a side
                    # stream, before entering the capture region.
                    for _ in range(3):
                        self._compiled_core(self._static_x, self._static_m)
                    torch.cuda.synchronize()
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            self._compiled_core(self._static_x, self._static_m)
                    torch.cuda.current_stream().wait_stream(side)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        y = self._compiled_core(self._static_x, self._static_m)
            except Exception as exc:
                return False, f"second capture raised {type(exc).__name__}: {exc}"

            try:
                ok, why = self._verify_pair(g, y)
            except Exception as exc:
                return False, f"verification raised {type(exc).__name__}: {exc}"
            if not ok:
                return False, why
            self._bufs.append((g, y))
            return True, ""

        def _verify_pair(self, g_new, y_new):
            """The whole safety argument for sharing one graph pool, executed.

            An empty or aliasing capture is a silent-wrong-answer mode (finding 17,
            finding 24). None of these checks costs anything at steady state.
            """
            g_old, y_old = self._bufs[0]
            if y_new.data_ptr() == y_old.data_ptr():
                return False, ("second capture reused the first output address -- "
                               "double buffering would be a no-op")
            if y_new.shape != y_old.shape or y_new.dtype != y_old.dtype:
                return False, "second capture produced a different output shape/dtype"

            pa = torch.randn_like(self._static_x)
            pb = torch.randn_like(self._static_x)
            ea = self._compiled_core(pa, self._static_m)
            eb = self._compiled_core(pb, self._static_m)
            tol = dict(rtol=1e-3, atol=1e-4)

            # new graph writes its own buffer, correctly
            self._static_x.copy_(pb)
            g_new.replay()
            if not torch.allclose(y_new, eb, **tol):
                return False, "second graph replayed wrong values (empty or stale capture)"
            snap_new = y_new.clone()

            # replaying the OLD graph must not touch the NEW buffer
            self._static_x.copy_(pa)
            g_old.replay()
            if not torch.allclose(y_old, ea, **tol):
                return False, "first graph stopped replaying correct values"
            if not torch.equal(y_new, snap_new):
                return False, ("replaying graph 0 mutated graph 1's output buffer -- "
                               "the shared pool overlapped them")
            snap_old = y_old.clone()

            # ... and symmetrically
            self._static_x.copy_(pb)
            g_new.replay()
            if not torch.equal(y_old, snap_old):
                return False, ("replaying graph 1 mutated graph 0's output buffer -- "
                               "the shared pool overlapped them")
            return True, ""

        # ------------------------------------------------------------------- liveness

        def _clobber_verdict(self, idx: int) -> str:
            """May we overwrite buffer `idx`? 'free' | 'rebind' | 'aliased'."""
            if idx not in self._live:
                return "free"
            # NB: never bind the handout to a local here. `sys.getrefcount` counts every
            # live reference, so a convenience local makes the count read 3 and the check
            # reports "the caller is holding it" on every single call -- which measured
            # as a clone per call, i.e. exactly the parent, silently.
            _g, buf = self._bufs[idx]
            try:
                # buf's own TensorImpl + the handout's == 2 when nothing else aliases it.
                if _storage_use_count(buf) > 2:
                    return "aliased"
            except Exception:                                  # pragma: no cover
                return "aliased"                               # conservative
            # Exactly two Python references (the dict slot and getrefcount's argument)
            # means the caller has dropped it.
            if sys.getrefcount(self._live[idx]) <= 2:
                return "free"
            return "rebind"

        def _retire(self, idx: int) -> None:
            """Take a buffer out of rotation forever and stop handing buffers out.

            Its memory is kept alive (the graph and the tensor are held) so anything the
            caller still aliases stays valid, and it is never replayed into again.
            """
            self._retired.append(self._bufs[idx])
            del self._bufs[idx]
            self.retired_buffers += 1
            self._zero_copy = False
            self._live.clear()
            self._rr = 0
            self.buffering_reason = (
                f"{self.buffering} -> clone-on-return: a caller aliased a returned tensor, "
                f"which cannot be rebound; {self.retired_buffers} buffer(s) retired")

        # -------------------------------------------------------------------- forward

        def forward(self, x, valid_token_mask=None):
            if getattr(self, "_graph", None) is None:
                # Priming, compilation, capture and the no-graph fallback all live in the
                # parent chain and are unchanged. Whatever it returns is already safe (it
                # clones, or it is a fresh allocation from the compiled callable).
                out = super().forward(x, valid_token_mask)
                if getattr(self, "_graph", None) is not None and not getattr(
                        self, "_buffers_built", False):
                    try:
                        self._build_buffers()
                    except Exception as exc:                   # pragma: no cover
                        self._buffers_built = True
                        self._bufs = [(self._graph, self._static_y)]
                        self._retired, self._live, self._rr = [], {}, 0
                        self._zero_copy = False
                        self.buffering = "single"
                        self.buffering_reason = f"single: build raised {type(exc).__name__}"
                return out

            if not self._bufs:                       # everything retired
                return self._compiled_core(x, valid_token_mask)

            idx = self._rr % len(self._bufs)
            if self._zero_copy:
                verdict = self._clobber_verdict(idx)
                if verdict == "aliased":
                    self._retire(idx)
                    return self._compiled_core(x, valid_token_mask)
                if verdict == "rebind":
                    handout = self._live.pop(idx)
                    try:
                        handout.set_(handout.clone())
                    except Exception:
                        # Cannot preserve it; do not destroy it either.
                        self._retire(idx)
                        return self._compiled_core(x, valid_token_mask)
                    self.preserve_rebinds += 1
                    self.output_copies += 1
                else:
                    self._live.pop(idx, None)

            self._rr += 1
            graph, buf = self._bufs[idx]
            self._static_x.copy_(x)
            if self._static_m is not None:
                self._static_m.copy_(valid_token_mask)
            graph.replay()

            if not self._zero_copy:
                self.output_copies += 1
                return buf.clone()

            handout = buf.detach()      # a distinct TensorImpl, so aliases are countable
            self._live[idx] = handout
            self.zero_copy_returns += 1
            return handout

    return CandidateV21
