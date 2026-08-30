"""Candidate v40 -- a SECOND attention tile shape, chosen by a symmetric sweep.

Generation 40. Parent: `v38_stream_fallback` (`7cee27c`). Branch: `cand/g40/attn-loop-census`.
Picks up the proposal finding 48 opened and declined to build.

WHAT IT ADDS
------------
`bench/kernels/attn_looped.py`: causal attention with the K/V axis back in a loop, so K
and V tiles stage through shared memory and Triton's pipeliner has something to overlap
memory latency with. `bench/kernels/attn_choice.py` then sweeps BOTH forms plus
`sdpa+repack` over their complete legal grids on the measured device, with one timer and
one repeat count, and may return exactly one answer: "use the looped form with this tile",
under v23's existing `DECISIVE` margin. **Every other outcome falls back to v23's
`autotune_tile` run unchanged**, so a shape the looped kernel does not win is byte-identical
to v38 by construction rather than by coincidence.

THE CENSUS, STATED BEFORE THE KERNEL WAS WIRED IN
--------------------------------------------------
`bench/probes/g40_attn_loop/probe_census.py`, config 10 on `v38_stream_fallback`, inside
the replayed graph, 20 forwards, after 200 settling calls:

    wall (median of 100)                     251.90 us
    device time in the graph                 222.50 us   (88.3% of wall)
      projection GEMM     x16   124.59 us    56.0% of device   49.5% of wall
      layernorm           x 9    46.15       20.7             18.3
      ATTENTION           x 4    44.14       19.8             17.5
      copy (DtoD)         x 2     7.62        3.4              3.0

So attention is **17.5% of config 10's wall**, confirming the 17.6% finding 48 assumed but
had never measured on the shipping candidate. Per call that is **11.04 us**.

AND THE REGIME CORRECTION FINDING 48 DID NOT MAKE
--------------------------------------------------
Finding 48 priced this at +0.0048 by multiplying that in-graph time by an op-level ratio
measured under `do_bench`, which flushes L2 and pays a launch. Its incumbent read 24.757
us per call against the 11.04 us the model pays -- a **2.24x regime gap** [L33]. Since the
whole mechanism is *latency hiding*, and latency is exactly what changes when the operands
are already in this card's 48 MB L2, the ratio had to be re-measured in the model's own
regime before it could be believed.

`bench/probes/g40_attn_loop/probe_regime.py` times both forms in both regimes, each swept
over its full legal grid, `sdpa+repack` included, two independent runs:

    cfg 10   flushed  incumbent 22.528 us   best looped 18.432   1.222x / 1.375x (run1)
             HOT      incumbent 11.189      best looped  9.109   1.228x / 1.226x (run1)

The hot-regime ratio **replicates to 0.2% across runs** and the incumbent's hot time
(11.189 us) reproduces the census's 11.04 us/call independently. The advantage survives.

    DILUTED CEILING, at the measured hot ratio of 1.228x:
        saved   44.14 * (1 - 1/1.228)  =  8.20 us/fwd  =  3.25% of config 10's wall
        config 10   2.33x -> 2.408x
        Delta weighted_score            =  +0.0056

WHAT IT ACTUALLY MEASURED
-------------------------
`bench/abba.py --rounds 5 --warmup 200`, both arms resident, cold round discarded, TWO
independent runs, with configs 1, 3 and 9 carrying byte-identical code in both arms as the
in-run control:

    cfg 10   231.42 us -> 223.23    1.0367x  and  1.0365x     replicates to 0.02%
    cfg  4    87.04    ->  87.04    1.0000x  /  0.9933x       engages, no effect
    cfg 12    75.78    ->  76.80    0.9867x  /  1.0000x       engages, no effect
    cfg 1/3/9  CONTROL, identical code       0.9808x-1.0000x

Every control difference is zero or exactly ONE 1.024 us event-timer quantum -- the raw
medians are all multiples of it. Config 10's delta is 8.19 us = EIGHT quanta, twice.

And the saving is attributable. `probe_census_pair.py` profiles both arms in ONE process,
ABBA-interleaved:

    bucket             v38 us/fwd   v40 us/fwd    delta
    attention              40.84        33.26     -7.58     <- the whole difference
    layernorm              42.96        43.00     +0.04
    projection GEMM       115.38       115.32     -0.05
    copy                    7.17         7.00     -0.17
    per call:  _attn_single_tile 10.211 us -> _attn_looped 8.315 us  =  1.228x

**Delta weighted_score = +0.0061** (config 10 only; 2.33 -> 2.4153). Configs 4 and 12
score zero, and the other ten configs are byte-identical to the parent.

    NOTE FOR WHOEVER SWEEPS THIS. The prime-time sweep compiles and times 5-48 looped
    tiles plus the single-tile grid plus sdpa, so first-forward time is 14-67 s per
    config. That is exactly the construction-time work finding 45 showed `run_matrix`'s
    ISOLATED arm misreports by 2-4x. RANK THIS ON THE INTERLEAVED ARM.

TWO RESULTS THAT WERE NOT ASKED FOR, AND ONE OVERTURNS FINDING 48
------------------------------------------------------------------
* **Config 9 is CLOSED, and finding 48's +0.0021 for it is withdrawn.** Its looped winner
  was `BM=128`, which at `B*H = 64` is one CTA per SM -- one wave, the thing this file's
  predicate declines. With that arm excluded the best looped tile loses to SDPA:
  **0.955x flushed, 0.826x hot**, both replicated. Finding 48 measured config 9 only in
  the flushed regime and only against a baseline it had already shown to be unreliable.

* **`sdpa+repack` beats the INCUMBENT single-tile kernel on config 10 in the hot regime**
  (9.987 us against 11.189). `attn_single_tile`'s docstring pre-registered exactly this
  ("the screen measured config 10 at -7.1% end to end -- the marginal case, sitting at
  exactly MIN_RESIDENT_BLOCKS") and deliberately left it unresolved pending a sweep. This
  is that sweep, and the pre-registered suspicion was right. **This candidate deliberately
  does not act on it.** Switching config 10 to the vendor is a different change with its
  own evidence needs, and bundling it here would make this candidate's A/B uninterpretable
  -- a config-level delta could no longer be attributed to the looped kernel. It is
  written up as a proposal instead. What sdpa does here is serve as a hard guard: a Triton
  form slower than the vendor call it replaces is never selected, whatever its margin over
  our own incumbent (finding 48's config-8 case, 1.006x).

  Note the arithmetic that follows from this, because it decided the shape of the code:
  the looped form is 1.228x the incumbent but only 1.096x sdpa. A first draft applied
  `DECISIVE` against `min(incumbent, sdpa)` and therefore declined config 10 -- the one
  config the candidate exists for. `DECISIVE` asks whether the gain over the STATUS QUO
  beats the noise, so it belongs against the incumbent; sdpa is a floor, not a margin.

THE EXTENSION POINT, AND WHY IT IS A REFACTOR OF THE ANCESTORS
---------------------------------------------------------------
The `if self.attn_used: single_tile_attention(...) else: <sdpa + repack>` block existed
inline in **four** places -- v23's `_core`, v34's `_core`, and both branches of v36's.
Overriding attention meant copying three long `_core` bodies and keeping them in sync with
their originals forever, which is the [L14] shape exactly. So that block moved, unchanged,
into `CandidateV23._attention`, and the four call sites became `self._attention(qkv, a, b,
s)`. **That refactor is behaviour-preserving and is asserted to be**, not assumed:
`tests/bench/test_attn_extension_point.py` checks the method reproduces the inline
expression on both branches, and the census was re-run afterwards and reproduces.

WHAT DOES NOT CHANGE
--------------------
* Every v38 correctness fix is inherited untouched: v33's shape latch, v35's
  combination-only mask fix, v33's streaming with the config-14 protocol, v38's
  attempt-then-fall-back residency.
* `attn_tile` keeps its meaning for the single-tile form (a 3-tuple), so every ancestor
  reading it is unaffected. The looped form is intercepted before that read.
* Config 14 is untouched: `attn_choice`'s probe-memory budget declines seq_len
  100000 (a 9.8 GiB probe tensor) without allocating, and the shape falls back to SDPA --
  which is exactly what v38 already does there.
* `attn_form` is reset on a shape change alongside v37's derived latch set, so a model
  warmed at one shape and called at another re-decides rather than running a plan sized
  for a batch that is no longer there.
"""

from __future__ import annotations

import torch

from .v38_stream_fallback import build as build_v38
from ..kernels import attn_choice, attn_looped
from ..kernels.attn_single_tile import applies as single_tile_applies


def build(baseline_cls):
    v38_cls = build_v38(baseline_cls)

    class CandidateV40(v38_cls):
        # Which form `_attention` dispatches. "single_tile" is the inherited behaviour,
        # so an instance that never reaches `_decide_attn` behaves exactly like v38.
        attn_form: str = "single_tile"

        def _decide_attn(self, x):
            """Decided ONCE per shape, before compilation and graph capture, so the form
            and the tile are Python constants by the time anything traces them.

            Overrides v23's version. The difference is only WHICH sweep runs: v23 sweeps
            the single-tile form's legal grid, this sweeps both forms' plus SDPA. The
            failure handling is v23's verbatim -- never fail closed on a tuner.
            """
            a = self.layers[0].attention
            b, s, _ = x.shape
            props = torch.cuda.get_device_properties(x.device)

            ok_looped, why_looped = attn_looped.applies(b, a.num_heads, s, a.head_dim,
                                                        props)
            if ok_looped:
                try:
                    tile, how = attn_choice.autotune_looped(
                        s, a.head_dim, a.num_heads, b, x.device)
                    self.attn_form = "looped"
                    self.attn_tile = tile
                    self.attn_used = True
                    self.attn_reason = f"looped; {how}"
                    return
                except Exception as exc:           # never fail closed on a tuner
                    why_looped = str(exc)

            # THE FALLBACK IS THE PARENT'S OWN DECISION, RUN UNCHANGED. Not "something
            # equivalent to it" -- literally v23's `_decide_attn`, so a shape the looped
            # kernel does not win is byte-identical to v38 by construction. An earlier
            # draft reimplemented the fallback and drifted from it immediately: its
            # different (hot) timer re-tuned the SINGLE-TILE form too, returning
            # `(16, 2, 1)` on config 2 where v38 runs `(64, 4, 1)`. That is a separate
            # change riding into this candidate's measurement, and it would have
            # destroyed the byte-identical control the measurement depends on.
            super()._decide_attn(x)
            self.attn_form = "single_tile" if self.attn_used else "sdpa"
            self.attn_reason = f"{self.attn_reason}; looped declined: {why_looped}"

        def _attention(self, qkv, a, b, s):
            """The looped form, or whatever v23 would have done.

            `attn_tile` is a 4-tuple only while `attn_form == "looped"`, and that case
            never reaches the superclass -- so every ancestor's 3-tuple unpack stays
            valid.
            """
            if self.attn_used and self.attn_form == "looped":
                return attn_choice.dispatch("looped", self.attn_tile, qkv,
                                            a.num_heads, a.head_dim, a.scale)
            return super()._attention(qkv, a, b, s)

        def _invalidate_shape_state(self, mask=None):
            """Reset the form with everything else latched to an input shape.

            `attn_form` is introduced by THIS class, so v37's `SHAPE_LATCHED` -- derived
            at v37's build time over the classes below it -- cannot name it. v33 already
            reopens the decision by setting `attn_reason = "undecided"`, and
            `_decide_attn` always assigns `attn_form` on every path including its two
            decline paths, so this is belt and braces. It is here because [L50] is the
            defect this lineage keeps rediscovering: a fix that makes a second, dormant
            defect reachable for the first time.
            """
            super()._invalidate_shape_state(mask)
            self.attn_form = type(self).attn_form

    return CandidateV40
