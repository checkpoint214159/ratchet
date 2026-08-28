# 00 — Mission, constraints, tiers

## Target hardware

Measured and queried 2026-08-28 (M0/M1 calibration, cached in `ledger/device.json`).

```
GPU:                    NVIDIA GeForce RTX 4070 Ti SUPER  (AD103, WSL2)
Compute capability:     sm_89  (Ada)
SM count:               66
Shared mem / block:     99 KB  (optin; 100 KB per SM)
L2 cache:               48 MB
Memory bandwidth:       672 GB/s theoretical  /  613.7 GB/s measured  (91%)
Peak BF16 dense:        88.2 TFLOP/s  (table: 66 SM x 2610 MHz x 512 FLOP/SM/clk,
                                       FP32-accumulate; = 2x the 44.1 FP32 shader figure)
Ridge point:            144 FLOP/B  (derived from measured bandwidth)
MMA family:             mma.sync  (no wgmma, no tcgen05, no TMA)
Clocks lockable:        NO (WSL: nvidia-smi -lgc fails "Unknown Error"; persistence
                        mode works). Use minimum-of-N, interleave candidate/baseline.
Launch overhead:        2.22 us measured  (do_bench - do_bench_cudagraph)
torch / triton / py:    2.8.0+cu128 / 3.4.0 / 3.10.12
                        (NOTE: py 3.10 < the 3.11+ CLAUDE.md asks for; seed code runs
                        fine on 3.10 -- avoid 3.11-only syntax until upgraded)
Beryl control plane:    pinned @ 44353116bc328d66e0581161ff4f05ee97effcdf
```

Registers/SM 65536; total memory 16 GB. Consumer-part caveats from `device.py` apply:
FP32-accumulate tensor rate is HALF the FP16-accumulate rate, and FP16-accumulate will
not survive the abs<0.002 tolerance. The 99 KB shared-memory budget is the binding
constraint for tile selection (spec 04): BLOCK_M=BLOCK_N=128, d=128, 3 stages needs
224 KB and does NOT fit -- solve for feasible tiles, do not copy H100 configs.

## What we are building

An agentic harness that continuously proposes, measures and improves Triton attention
kernels for one specific GPU, and keeps a permanent, honest record of every measurement
it has ever taken.

Three things distinguish it from "an agent that writes kernels":

1. **The measurement apparatus is a first-class, protected artifact**, separated from the
   thing being optimized by a hard boundary. Most published agentic kernel results are
   partly measurement artifacts; this is the design response.
2. **The dispatch table is calibrated from device properties**, not hardcoded shape
   thresholds. The same arithmetic intensity is compute-bound on one GPU and
   memory-bound on the next, so a constant that is right here is wrong there.
3. **The evaluator improves too**, under a promotion rule anchored to real measured
   outcomes. This is the Red Queen idea, restructured so that hardware measurements are
   never invalidated (see `docs/01-architecture.md`, "Where we depart from RQGM").

## Competition constraints (TikTok TechJam 2026, GPU kernel track)

Restated from the problem statement, with the operative reading.

| Clause | Operative reading |
|---|---|
| "Submit one or several GPU kernels that implement the layers" | The fusion boundary is ours to choose. At small shapes the win is not in attention at all. |
| "All the combinations of input shapes will be told to the participants" + "can decide different implementations for different shapes by adding shape checks" | This is explicitly a **dispatch problem**. One kernel does not win across the matrix. |
| "relative error < 0.02, abs error < 0.002" | The absolute bound is the binding one on order-1 outputs — effectively 0.2%. FP32 accumulation is mandatory; naive FP8 is out. |
| "Optimize & test your codes on your own machine" | No fixed hardware, so raw speedups are not comparable across teams. Methodology is the scored artifact. |
| "The use of AI tools is encouraged… bonus points for a tech report on AI skills/tools used" | The loop is a graded deliverable, not a footnote. |
| Judging: Technical Execution 35 / Innovation 20 / Impact 20 / Feasibility 15 / Presentation 10 | Speed lives inside 35%. 55% rewards a system somebody else could pick up and use. |

### The strategic consequence

Do not submit fast kernels. Submit **a shape-aware kernel dispatch system with an
AI-in-the-loop autotuner and an honest measurement harness**, which happens to contain
fast kernels. The kernels score under Technical Execution. The system scores under
Innovation, Impact and Feasibility, and it is the only part that survives the objection
every judge will raise: *you all benchmarked on different GPUs, so how do I compare you?*

### Where this problem sits relative to the field

Worth being clear-eyed. A dense transformer layer with static shapes announced in
advance is a 2023-shaped problem. Dense GEMM, dense attention forward and backward,
elementwise fusion and paged attention are all closed — cuBLAS, FlashAttention-4, cuDNN
and `torch.compile` own them, and of 24 operators in one 2026 study only 1 of 9
vendor-backed ops was beaten. So "we made attention fast" is not available as a
differentiator. What is open, and what this harness targets, is the meta-level: the
dispatch, the measurement discipline, and the search loop.

## Tiers

Build in this order. Each tier is worthless on top of a broken predecessor.

### Tier 0 — trustworthy measurement (target: one working day)

The oracle, the ledger, and one hand-written baseline kernel that passes correctness on
the full shape matrix. No search, no agent loop, no cleverness.

**Gate:** you can produce a speedup number and defend, line by line, how it was obtained.
Re-running the same measurement twice gives you overlapping confidence intervals.

### Tier 1 — the search loop (the competition deliverable)

Automated propose → gate → compile → verify → time → record → select. The dispatch table
populated from measurement. A report generator. Parametric search using dual annealing at
low budget and first-improvement iterated local search above it, with explicit handling of
infeasible configurations.

**Gate:** the loop runs unattended for an hour and the best-known table improves without
any measurement in the ledger having been deleted or overwritten.

### Tier 2 — co-evolution (the novel part)

A learned critic that predicts compile failure, correctness failure and rough performance
from source, promoted at epoch boundaries only when it beats the incumbent at predicting
*held-out real measurements*. An adversarial input pool that grows from near-misses. A
research scout that reads other open implementations and proposes architectural moves.

**Gate:** the critic demonstrably saves GPU time — measure candidates-pruned × mean
evaluation cost against critic overhead — without its false-negative rate on the held-out
slice exceeding a stated bound.

## Timeline reality

The competition window is **Aug 29 – Sep 1, 2026**, 72 hours, submission at 12:00 on
Sep 1. Today is Aug 27.

That means: Tier 0 today and tomorrow. Tier 1 in the first 12 hours of the window. Tier 2
only as far as it gets, with the design documented regardless — a documented design with
an honest "not yet implemented" is worth more under Feasibility than a half-working one
that inflates the numbers.

Budget the last 18 hours for the tech report, README and video. That is 30% of the rubric
and the only part a judge is guaranteed to consume.
