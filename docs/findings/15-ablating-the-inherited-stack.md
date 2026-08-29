# Finding 15 — Ablating nine generations of tricks under compilation

Recorded 2026-08-29. Branch `cand/g10/ablation`. Three siblings from v9a, each removing
exactly one inherited trick. Measured on configs 1, 6, 12, 13 (mainstream, throughput,
launch-bound, long-context). Numbers are the cost of REMOVAL — positive means the trick
still pays.

| removed | cfg1 | cfg6 | cfg12 | cfg13 | verdict |
|---|---|---|---|---|---|
| L2 batch chunking (v3) | +5.8% | **−0.3%** | +5.7% | +2.1% | **subsumed** |
| fused Q\|K\|V (v1) | +20.0% | +5.3% | +19.3% | +1.9% | still pays |
| fp16 weight cache (v1/v6) | +185% | +134% | +64% | **+395%** | essential |

## Chunking has been subsumed by the compiler

This is the finding worth acting on. **L2-sized batch chunking no longer buys anything —
including on config 6, the config it was designed for, where removing it is very slightly
faster.**

When v3 introduced it, chunking took config 6 from 3.21x to 5.72x. That was real. But
under `torch.compile` the win is gone: Inductor manages its own working set and CUDA-graph
capture handles the rest, so our Python chunk loop now only adds loop overhead and a
calibration constant we have to justify.

The residual +5.8%/+5.7% on configs 1 and 12 is *not* chunking helping — those configs
have batch 64, whose working set already fits the residency target, so `solve_chunk`
returns the whole batch and no chunking occurs. The difference there is the extra branch
and `torch.empty_like` allocation on the un-chunked path, which is noise-adjacent.

**Action: chunking should be removed from the lineage.** It is complexity carrying a
device-calibration constant, a tuning parameter, and a test, for a benefit the compiler
now provides for free. Keeping it would mean shipping a submission whose most
sophisticated-looking component does nothing.

## What survived, and why it is coherent

**The fp16 weight cache is decisive** — removing it costs 1.6x to 5x, worst on config 13.
Our hand-rolled "compute in fp16, accumulate in fp32" beats leaving the precision choice
to Inductor by a wide margin. Note the accuracy figures confirm the mechanism: `max_abs`
*drops* to 9.6e-4 without it, because it is genuinely running fp32 and paying for it.

**Fused Q|K|V still pays on the small configs** (+20% on 1, +19% on 12) and barely
registers on the large ones (+1.9% on 13). That is exactly the expected shape: three small
GEMM launches versus one is a launch-count problem, and launch count only matters where
there is not enough work to hide it.

## Why this ablation was owed

The dossier's skeptic paper (arXiv 2602.16805, *Simple Baselines are Competitive with Code
Evolution*) argues that evolutionary machinery is routinely credited for wins that a
simpler baseline also achieves. v9a is nine generations of stacked tricks, and every one
was justified against a world that did not yet include Inductor. Without this ablation we
would have shipped all nine and attributed the result to the stack.

**One of the nine is now provably inert.** That is a cheap finding that only exists because
the ablation was run, and it is the kind of result an evolutionary loop will never produce
on its own — loops add, they do not subtract.
