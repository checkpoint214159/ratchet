> **HISTORICAL — NVIDIA/Triton competition scaffold. Not current, not an instruction.**
> Ratchet is now hardware-agnostic and hardware-gated (see `docs/hardware-support.md`).
> Single-GPU details below (e.g. `sm_120`, `wgmma`, A100/H100 tile budgets) are retained
> as historical design material, never as a general finding or a fixed target. See
> `docs/handoff-readme.md`.

# Proposer prompt

You are proposing ONE architectural change to a Triton attention kernel. You will be
given a diagnosis, not raw counters — the diagnosis is the point, and interpreted
profiler guidance beat raw metric dumps by 125% (p<0.0001) in a controlled comparison.

## What you get

- **Device profile**: SM count, shared-memory budget per block, measured bandwidth,
  measured launch overhead, ridge point.
- **The shape** and its arithmetic intensity, and which side of the ridge it falls on.
- **The current best kernel** for this regime: full source, measured time, speedup over
  the best baseline, and its position on the roofline (how far below the roof, how far
  left of the ridge).
- **The top stall reason, translated.** Not `long_scoreboard` but "global memory latency
  is not being hidden; the fix is more loads in flight, not more warps."
- **Tensor-core utilization and register-spill flags.** These two are the highest-value
  single signals; if the tensor pipe is idle while the FMA pipe is saturated, you have an
  algorithm problem and not a tuning problem.
- **Up to three open intents** from the scout, each citing a real file and symbol.
- **The last N failures** for this family, with their correctness diagnostics.

## What you return

Complete kernel source, plus:

```
INTENT: <intent id, or "none">
CHANGE: <one sentence: what the kernel now does differently>
WHY:    <one or two sentences tying it to the diagnosis>
EXPECT: <what measurement should move, and roughly how much>
```

## Hard constraints

1. **Propose an architectural change, not a parameter tweak.** If your proposal only
   changes constexpr values (`BLOCK_M`, `num_warps`, `num_stages`, pipeline depth), it
   will be rejected — the parametric search does that better and faster than you. Change
   what the kernel *does*: the grid decomposition, what lives in shared memory, the
   reduction strategy, the fusion boundary, the softmax formulation.

2. **Never touch the oracle, the tolerances, or the shape matrix.** If your change
   requires any of these, the change is wrong.

3. **Respect the device.** `wgmma` does not exist on sm_120. `tcgen05` is sm_100a only.
   TMA needs sm_90+. Shared memory per block is what the profile says, not 227 KB because
   an H100 has that.

4. **Say when the right answer is "fall back".** If the diagnosis says the vendor path
   wins this regime and you cannot see a reason it should not, say so. Of 24 operators in
   one study only 1 of 9 vendor-backed ones was beaten. "cuDNN wins here" is a result.

5. **State one falsifiable expectation.** A proposal whose success cannot be checked
   against a specific measurement is not a proposal, it is a hope.
