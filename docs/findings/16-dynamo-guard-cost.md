# Finding 16 — Reclaiming Dynamo's per-call guard cost, and what it says about our regimes

Recorded 2026-08-29. Candidate `v12_graph_over_compile`, branch `cand/g12/graph-over-compile`.

## The measurement that started it

Config 2 (B=1) was the frontier's weakest config at 1.36x over the compiled baseline.
Profiling it found something that does not show up in a speedup number: **it is
CPU-dispatch-bound, not GPU-bound.**

```
per call:  CPU 232 us   GPU 126 us
  TorchDynamo Cache Lookup   22.5 us   <- guard evaluation, every single call
  cudaGraphLaunch            49.8 us
```

Against roughly **1.2 us of actual arithmetic**. Essentially the entire cost of config 2
is deciding what to run.

## The change, and why it is not a reversal

v9a and v11 deliberately disabled our static-buffer CUDA graph, reasoning that
`reduce-overhead` installs its own and stacking two graph mechanisms invites silent
staleness. That was sound but incomplete: **Inductor's graph still sits behind Dynamo's
guard check**, so every call re-evaluates guards before it can reach the replay.

v12 inverts the arrangement rather than undoing it — compile in the *default* mode
(fusion, no Inductor cudagraphs) and capture the compiled callable in our own graph. The
two mechanisms still never nest; ownership just moves.

## Result: +7.9%, entirely where predicted

| | geomean vs compiled | losses |
|---|---|---|
| v11_lean | 2.514x | 0 |
| **v12_graph_over_compile** | **2.712x** | **0** |

Per config, the gain is almost perfectly concentrated:

| config | v11 | v12 | delta |
|---|---|---|---|
| 3 | 1.63x | **3.05x** | **+87.3%** |
| 2 | 1.36x | **1.91x** | **+40.6%** |
| 1, 4, 5, 7, 11, 12 | — | — | +0.4% to +5.1% |
| 6, 8, 9, 10, 13 | — | — | −1.0% to −3.3% |

Everything outside configs 2 and 3 is inside the noise floor. That is the signature of a
**fixed per-call cost**: 22 us is 20%+ of a 97 us call and 0.7% of a 3.3 ms one.

## The finding that outlives the optimization

**Our regime labels are too coarse.** `bench/matrix.py` classifies configs 2, 3, 4 and 12
together as `launch_bound`. v12 helped 2 and 3 enormously and did nothing for 4 and 12
(+0.8%, +0.7%).

The label groups by *shape* — small batch, short sequence — but the thing that actually
predicts the win is **call duration relative to a fixed dispatch cost**, and configs 4 and
12 do enough total work (B=16 and B=64) that 22 us is no longer a meaningful fraction.

This matters beyond one candidate. The regime table is used for reporting and reasoning,
and it just made a wrong prediction. Regimes derived from shape parameters are a
convenience; the honest predicate is measured time, and where the two disagree the
measurement wins. The labels are kept, with this caveat attached, rather than silently
adjusted to fit — they were wrong for a reason worth remembering.
