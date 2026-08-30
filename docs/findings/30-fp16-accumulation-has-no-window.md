# Finding 30 — fp16 MMA accumulation: the reading is right, the window is empty

Recorded 2026-08-30. Candidate: `bench/candidates/v25_fp16_accum.py`, branch
`cand/g25/fp16-accum`, parent `v18_capture_insurance`. Kernel:
`bench/kernels/ffn_accum.py`. Tests: `tests/bench/test_v25_fp16_accum.py` (24 passed).

Three research agents independently proposed this mechanism (C-03, A-05, D-05). Per L34
that is corroboration of the READING, not of the value — all three were reading the same
sm_89 documentation and none had run the model. This is the run.

*(Method note: the three proposal files named in the assignment do not exist anywhere in
the repository or any sibling worktree. The mechanism was reconstructed from the
assignment's own statement of it.)*

## The reading is correct, and it is worth more than a datasheet citation

Consumer Ada runs tensor-core FP16-with-FP32-accumulate at half rate. Verified from
generated PTX rather than assumed:

| `tl.dot(out_dtype=...)` | emitted instruction |
|---|---|
| `tl.float32` | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` |
| `tl.float16` | `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` |

In a loop of 1024 `mma` instructions with everything else held constant, the f16 form
measures **1.62x** faster (38.2 us → 23.6 us). The premise every proposal rested on is
true, and a test pins it — if it ever stops holding, this finding is closing the wrong
thing.

## Reason 1: our kernels are not standing anywhere near that instruction

The fused FFN block moves `d_model*(2+4+4)` bytes per token and does `4*d_model*ffn_dim`
FLOP. At `d_model == ffn_dim == 128` that is an arithmetic intensity of **51.2 FLOP/B**
against this device's measured ridge point of **143.7 FLOP/B** (88.2 TFLOP/s ÷ 613.7
GB/s). Memory-bound by 2.8x. Measured directly:

| tokens | achieved | % of measured BW | % of peak FLOP/s |
|---|---|---|---|
| 8,192 | 327 GB/s | 53.3% | 18.9% |
| 65,536 | 510 GB/s | 83.1% | 29.6% |
| 1,280,000 | 609–614 GB/s | **99.2–100.1%** | 35.3% |

At config 6's token count the kernel is **bandwidth-saturated**. The tensor cores are
already idling on HBM; a 1.6x instruction shortens something that is not on the critical
path. And so:

    config 6's shape, 1.28M tokens, four accumulator arms
      acc(0,0)  2667.8 us   1.000x
      acc(1,0)  2664.0 us   1.001x
      acc(0,1)  2674.0 us   0.998x
      acc(1,1)  2666.9 us   1.000x

**Not a win inside the noise floor. No win.** This is L33 in its purest form: the
mechanism measured 1.62x in isolation because the isolation was an MMA-saturated loop
with no memory traffic, which is the one condition our kernel never satisfies.

## Reason 2: the error is over budget at every depth the hardware can issue

Finding 08 established the distinction that made this worth probing — the residual
ACCUMULATES across layers, an elementwise op does not, so an fp16 accumulator inside one
GEMM over K=128 is a different risk from an fp16 residual across four layers. The
distinction is real. It is not enough.

One GEMM, unit-magnitude output, against an fp32 reference:

| K | fp32 accumulator | fp16 accumulator | % of the 2e-3 budget |
|---|---|---|---|
| **16** | 9.54e-07 | **2.800e-03** | **140.0%** |
| 32 | 1.43e-06 | 2.941e-03 | 147.0% |
| 64 | 2.15e-06 | 4.560e-03 | 228.0% |
| 128 | 4.29e-06 | 6.227e-03 | 311.4% |
| 256 | 4.53e-06 | 7.485e-03 | 374.3% |

**K=16 is the shallowest contraction sm_89's `m16n8k16` MMA can perform, and it already
misses by 40%.** There is no legal depth at which this instruction's accumulator fits the
locked tolerance. The region is not narrow. It is empty.

## The scissors, which is the part that generalizes

Two conditions must both hold, and they are monotone in *opposite* directions in the
contraction depth K — and in this architecture `K == d_model == ffn_dim`, so one shape
parameter drives both:

* **Fast** requires being above the ridge point. Intensity is linear in `d_model`, so
  `mma_bound` needs **d_model ≥ 359** on this device.
* **Accurate** requires `eps_fp16 * sqrt(K) ≤ atol`, which at the locked 2e-3 and unit
  output magnitude needs **K ≤ 16.8**.

A gap of **21.4x**, and it widens on better hardware: a higher peak-FLOPs-to-bandwidth
ratio pushes the ridge point up while fp16's mantissa stays at 11 bits. Both bounds are
computed from measured device properties in `ffn_accum.no_shape_satisfies_both()` rather
than asserted, so the claim is checkable on a card nobody here has seen.

Note the two arguments are independent. Even if a shape were compute-bound, its K would
be far past the accuracy ceiling; even if K were small enough, that shape would be deeply
memory-bound. Neither half rescues the other.

## Per-config margins — the deliverable

End-to-end against the fp32 reference at the locked tolerance (atol 2e-3, rtol 2e-2, OR),
with `fused=1` confirming the kernel actually ran. Batch clamped to ≤256 (config 6:
10000→256); `1,0` = fp16 accumulator on the first GEMM only, `0,1` the second, `1,1` both.

| cfg | shipped (fp32) | 1,0 | 0,1 | 1,1 | fails |
|---|---|---|---|---|---|
| 1 | 70.6% | 86.1% | 93.6% | **139.6%** | 1 |
| 2 | 60.9% | 68.9% | 75.6% | 81.1% | — |
| 3 | 60.9% | 74.9% | 83.1% | 110.5% | — |
| 4 | 65.0% | 81.3% | 88.4% | **117.5%** | 1 |
| 6 | 68.6% | 96.7% | 106.3% | **123.9%** | 5 |
| 5 | 76.9% | 88.1% | 106.2% | **125.1%** | 3 |
| 7 | 61.3% | 76.9% | 76.1% | 84.6% | — |
| 9 | 73.3% | 87.9% | 103.9% | **117.1%** | 2 |
| 10 | 70.1% | 91.4% | 106.0% | 122.9% | — |
| 11 | 66.2% | 84.8% | **105.1%** (1) | **133.9%** | 2 |
| 12 | 72.8% | 106.8% | 88.4% | 126.5% | — |
| 13 | 59.2% | 100.4% | 107.7% | **131.4%** | 14 |

**The shipped fp32 arm passes every config with 59–77% of budget spent.** The `1,1` arm
fails 6 of 12. The single-site arms pass — and that is the result worth reading carefully:

* `1,0` costs **15 to 34 percentage points of tolerance budget** for a measured 1.001x.
  Config 12 lands at 106.8% of the absolute budget with zero failures, surviving purely
  on the relative leg — finding 08's exact point that `max_abs` alone does not predict
  pass/fail, and L4's warning that the number to judge by is `failed_elements`.
* Config 6 is clamped 10000→256 here, so its error is **understated**. The op-level probe
  shows site-A error growing with token count (3.24e-3 at 8k → 3.88e-3 at 1.28M): more
  tokens, more draws from the tail.
* L26 measured that a routine, benchmark-exposed change in input distribution multiplies
  our error by ~2.5x, and that at `input_scale=0.01` every candidate already fails. A
  candidate sitting at 90–107% of budget does not survive that. Spending a third of the
  remaining margin to buy 1.001x is not a trade with a good side.

## Verdict

**Closed.** Not "needs tuning" — the same shape of result as finding 08: a mechanism whose
premise is correct and whose operating window does not exist on this hardware. It should
not be given GPU time, and `bench/run_matrix.py` was deliberately not run.

The stage-1 screen was run once, on an idle GPU: **PROMOTE, 2.297x vs the parent's
2.292x (+0.2%)**. That verdict must not be read as support for the mechanism. The screen
promotes on "not clearly worse than the parent", and this candidate *is* its parent on
the shipped path, so a tie was the only possible outcome. What the screen actually
confirms is narrow and worth having: subclassing v18 did not disturb its compile and
capture path. `bench/run_matrix.py` was deliberately not run.

`v25_fp16_accum` ships with the accumulator predicate declining on every shape, which
makes it **numerically identical to v18** (pinned by `torch.equal` against the parent).
Its value is the recorded boundary and the falsifier that keeps the boundary measured:
`RATCHET_FORCE_ACCUM=a,b` forces either site so the refused path can be re-measured
rather than re-argued.

## Method note — the first version of this test was green and vacuous

The margin report initially showed all four arms producing *byte-identical* numbers on
every config. That looked like a clean null. It was L36: v17's amortization predicate
declines the fused kernel at small token counts, so `_core` fell through to the parent and
**the forced accumulator never ran**. The table was a page of numbers about nothing.

Fixed two ways: forcing an arm now also bypasses the token-count gate (the shared-memory
gate is physical and still binds), and the test asserts both that `fused_ffn_used` is true
and that each forced arm's `max_abs` *differs* from the fp32 arm's. A falsifier that
cannot distinguish "the mechanism is harmless" from "the mechanism did not run" is not a
falsifier. L36 was written two days ago and it caught me anyway — the assertion that the
subject was actually built has to be written into the test, not remembered.
