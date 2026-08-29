> **HISTORICAL — NVIDIA/Triton competition scaffold. Not current, not an instruction.**
> Ratchet is now hardware-agnostic and hardware-gated (see `docs/hardware-support.md`).
> Single-GPU details below (e.g. `sm_120`, `wgmma`, A100/H100 tile budgets) are retained
> as historical design material, never as a general finding or a fixed target. See
> `docs/handoff-readme.md`.

# 01 — Architecture

## The shape of the thing

```
                    ZONE B — WORKSPACE (evolvable)
      ┌──────────────────────────────────────────────────────┐
      │  proposer ──► kernels/ ──► dispatch/ ──► critic/      │
      └───────┬──────────────────────────────────────┬───────┘
              │ candidate                            │ prediction
              ▼                                      │
      ┌───────────────────────────────────────┐      │
      │  ZONE A — ORACLE (immutable)          │      │
      │  reference · inputs · correctness     │      │
      │  timing · device · harness            │      │
      │  runs in a SEPARATE PROCESS           │      │
      └───────┬───────────────────────────────┘      │
              │ measurement (fact)                   │
              ▼                                      │
      ┌───────────────────────────────────────┐      │
      │  ZONE C — LEDGER (append-only)        │──────┘
      │  measurements.jsonl · artifacts/      │  held-out slice
      └───────────────────────────────────────┘  anchors critic promotion
```

Three zones, three different sets of rules, one direction of trust: the workspace may
read the ledger, the ledger only accepts writes from the oracle, and the oracle reads
nothing from the workspace except the candidate under test.

## Zone A — the oracle

`ratchet/oracle/`. Immutable during optimization. Six modules:

| Module | Responsibility |
|---|---|
| `device.py` | Introspect and calibrate: SM count, shared memory, L2, measured bandwidth, measured launch overhead, derived ridge point. Cached. |
| `inputs.py` | Deterministic input generation. Correctness distributions (including adversarial) and benchmark distributions, from disjoint shape sets. |
| `reference.py` | PyTorch reference attention, and the baseline family (SDPA across all backends, `torch.compile` max-autotune). |
| `correctness.py` | The gate. Locked tolerances, non-finite propagation, determinism, shape generalization. Returns rich diagnostics, not a boolean. |
| `timing.py` | Multiple timing backends behind one interface, each recording its own method metadata. |
| `harness.py` | Runs one candidate end to end in a subprocess: correctness first, then timing, then optional profiling. Returns a measurement record. |

**Why a separate process.** Three reasons, all load-bearing. A candidate kernel can hang,
illegal-memory-access, or OOM, and must not take the loop down with it. Triton JIT state
and CUDA context leak across candidates in-process and contaminate timing. And an
in-process candidate can monkeypatch the oracle, which defeats the entire boundary.

**Why a checksum manifest.** `scripts/check-oracle.sh` hashes every file under `oracle/`
against `oracle/.manifest.sha256`. This is detection, not prevention — a determined
process can rewrite both — but it converts a silent catastrophe into a loud one, which
is the realistic goal.

## Zone B — the workspace

`ratchet/kernels/` holds Triton kernel variants, one file per architectural family, each
exposing a uniform entry point so the harness can call any of them identically.

`ratchet/dispatch/` holds the shape × device decision tree. **Its predicates are
functions of queried device properties, never hardcoded constants.** See
`specs/04-dispatch.md`. This is the single most important design decision in the project
and the one that answers the "different GPUs" objection.

`ratchet/search/` holds the two-level search. `ratchet/critic/` holds the Tier 2
surrogate.

## Zone C — the ledger

`ledger/measurements.jsonl`, one JSON object per line, append-only. Every row is a fact
about hardware at a moment in time and is never edited or deleted. Schema in
`specs/02-ledger.md`.

Everything else — best-known tables, rankings, critic training sets, report tables — is a
*derived view*, rebuilt from the ledger by a pure function, and may be deleted freely.
The distinction is not bureaucratic: it is what lets the critic evolve without ever
invalidating a GPU measurement.

## The loop

```
  intents/  ──►  PROPOSE  ──►  critic GATE  ──►  COMPILE  ──►  CORRECTNESS
   (scout)         (LLM)        (cheap,           (Triton)      (oracle, gate)
                                 Tier 2)              │              │
                                                      │ fail         │ fail
                                                      ▼              ▼
                                                  record         record + add to
                                                  failure        adversarial pool
                                                                     │
                              SELECT ◄── RECORD ◄── PROFILE ◄── TIME ┘
                                │                                (oracle)
                                └──► dispatch table update (only on non-overlapping CI)
```

Every arrow that leaves the workspace and enters the oracle crosses a process boundary.
Every arrow that leaves the oracle writes to the ledger.

### Two levels of search

They have different costs and different operators, and conflating them is why naive
loops plateau.

**Architectural** — what the kernel *does*. Split-K over the KV axis, materialize S in
shared memory with two-pass softmax, fuse the norm into the epilogue, change the pipeline
depth strategy. Proposed by an LLM, seeded by the scout from real reference
implementations. Rare, expensive, high variance, and where the large wins live: in one
published case three rounds of parameter tuning took 9.52 → 4.03 ms and the 2× came from
a fourth round that changed the design to one-row-per-program.

**Parametric** — `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`, pipeline depth, swizzle
choice. Classical black-box optimization over a discrete, non-convex space. Do not use an
LLM for this; it is worse and slower than a proper optimizer.

The empirical guidance for the parametric level comes from Schoonhoven, van Werkhoven and
Batenburg's survey over 26 kernel spaces on 9 GPUs (`docs/03-research-dossier.md`):

- **Dual annealing** wins at low evaluation budgets (≤ 200 evaluations).
- **First-improvement iterated local search (FirstILS)** wins above that.
- **Treat tuning as deterministic**, using mean runtime as fitness, rather than modelling
  runtime as a random variable. Stochastic-aware methods (irace, SMAC) did not beat
  conventional black-box algorithms and needed higher budgets.
- **SMAC and Bayesian methods underperform here**, plausibly because of the failure rate.

That failure rate is the fact to design around: in their spaces **68% of convolution
configurations and 78% of GEMM configurations failed to compile**. An infeasible point
must be cheap, must be recorded, and must be assigned a large finite fitness — never an
exception that aborts the run and never a silent skip.

### Selection

Not "keep the fastest." Two rules:

1. **Promotion to the dispatch table requires non-overlapping confidence intervals**
   against the incumbent on the same shape, same device, same toolchain. A 3% win with
   overlapping error bars is noise.
2. **Parent selection for the next proposal round uses clade metaproductivity**, not the
   node's own score: sample a parent by the pooled success rate of its entire descendant
   subtree, via Thompson sampling. A mediocre kernel that spawns good children is a good
   parent, and scoring nodes by their own performance systematically loses those
   stepping stones. This is the Huxley-Gödel Machine's correction to the Darwin Gödel
   Machine and it is cheap to implement.

## Tier 2 — the critic, and where we depart from RQGM

The Red Queen Gödel Machine co-evolves an agent with the *evaluator that scores it*,
freezing evaluators within an epoch, swapping them only at checkpoints, and promoting a
challenger only when it beats the incumbent's ε-best-belief on a **fixed, held-out,
evaluator-independent anchor**. The demonstration that matters is its ablation: with a
frozen critic, the agent saturated its validation set at 100% while collapsing to 78% on
the anchor — a textbook Goodhart, fixed by making the evaluator move.

**What we take.** The cheap learned surrogate, the anchored promotion rule, the epoch
freeze, and the adversarial pool.

**What we deliberately do not take, and why.**

*Selective erasure of measurements.* In RQGM, erasing a utility record costs an LLM call.
Here it would cost a GPU benchmark run, which is the dominant cost of the whole system,
and its O(B) amortization argument does not survive that. **So the erasable evaluator is
the critic's *predictions*, and hardware measurements are never erased.** This
restructuring is mandatory, not optional.

*Co-evolving the task distribution.* RQGM does not do this — it holds task pools fixed and
evolves the scorer. Neither should we for shapes: the shape matrix is given by the
competition and by real workloads, and a shape nothing runs is wasted search. If you ever
do want evolving tasks, that is the POET / PAIRED literature, not this one, and it needs
a difficulty band or a regret objective that RQGM has no analogue for.

### What the critic actually is

Input: kernel source, target shape, device properties. Output: `P(compiles)`,
`P(passes correctness)`, and a predicted speedup band.

Purpose: skip GPU evaluation on predicted-hopeless candidates. With a 68–78% compile
failure rate in comparable spaces, even a mediocre compile-failure predictor pays for
itself immediately.

Promotion: at epoch boundaries only, and only when a challenger beats the incumbent's
ε-quantile lower bound on a **held-out slice of the ledger, held out by candidate rather
than by row**, so that a candidate's own measurements cannot appear on both sides.

### The anchor-independence constraint

This is the one that would sink the project if you get it wrong, and the kernel domain
makes it *easier to get right and easier to walk into*.

Easier: anchor labels are free. Every candidate ever benchmarked yields a ground-truth
tuple, so the anchor set is large, domain-matched, and continuously refreshed. RQGM's
stated first limitation — "evaluator quality is only as good as its anchor" — binds much
less here than in paper review.

Easier to walk into: precisely *because* the anchor is auto-generated by our own harness.
If the loop can edit the harness, the anchor is circular and provides nothing. Hence:

- The oracle lives outside the evolvable workspace and is checksummed.
- It runs in a separate process.
- The anchor must include **known-bad kernels with known failure modes** — subtly wrong
  results, a race condition, a kernel fast only on benchmarked shapes, one that returns
  its input unchanged. A lenient critic must be actively penalised, not merely
  unrewarded. Seed these by hand in `oracle/known_bad/` before the critic exists.

### The adversarial input pool

After each epoch, collect inputs where the current best kernel **passed the harness but
disagreed with the reference under a wider sweep**: denormals, NaN and Inf propagation,
non-contiguous strides, ragged tail batch sizes, extreme dynamic ranges that expose
accumulation order, negated and scaled distributions. Those become a permanent addition to
the correctness suite.

This converts "the tests were too weak" from a static problem into a search problem, and
it is the highest-value co-evolution axis available here. Note that tolerance is **not**
on this axis and must never be: it is a locked requirement, not a search parameter.

## Reporting

`ratchet/report/` generates, from the ledger alone:

- The dispatch table with the winning implementation and measured margin per regime.
- A roofline position per shape: measured achieved FLOP/s against the calibrated ceiling.
- The full search trace: candidates proposed, pruned, failed to compile, failed
  correctness, timed, promoted.
- Ablations: what the critic saved, what the scout contributed, what the dispatch is worth
  against a single-kernel baseline.
- Explicit statements of what was **not** covered, which shapes fell back, and where the
  vendor library still wins. The negative results are worth more than they look.
