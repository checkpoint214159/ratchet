> **HISTORICAL — NVIDIA/Triton competition scaffold. Not current, not an instruction.**
> Ratchet is now hardware-agnostic and hardware-gated (see `docs/hardware-support.md`).
> Single-GPU details below (e.g. `sm_120`, `wgmma`, A100/H100 tile budgets) are retained
> as historical design material, never as a general finding or a fixed target. See
> `docs/handoff-readme.md`.

# Spec 03 — Search loop

**Zone B.** Two levels with different costs, different operators, and different owners.
Conflating them is why naive agentic loops plateau.

## Level 1 — parametric (classical optimizer, no LLM)

Space: `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`, pipeline depth, swizzle choice,
and any kernel-specific constexpr. Discrete, non-convex, with a large infeasible region.

### Algorithm selection, from arXiv 2210.01465

| Evaluation budget | Use |
|---|---|
| ≤ 200 | **Dual annealing** (`scipy.optimize.dual_annealing` over the mapped continuous cube) |
| > 200 | **First-improvement iterated local search** (FirstILS) |

Both must beat random sampling over the same space; if they do not, report that — it is a
legitimate finding and the skeptic's paper (arXiv 2602.16805) exists for exactly this
reason. Do **not** reach for SMAC or Bayesian optimization: they underperformed here,
plausibly because the surrogate cannot fit a space where most points fail.

**Treat fitness as deterministic.** Mean runtime over N repetitions is the fitness.
Measured runtime variation in comparable spaces was ~1%; stochastic-aware methods needed
larger budgets and did not win.

### Infeasibility is the design constraint

In the surveyed spaces **68% of convolution and 78% of GEMM configurations failed to
compile**. Therefore:

- An infeasible point costs a compile attempt, gets `status="compile_error"`, a ledger
  row, and a **large finite fitness** (`1e10`). Never an exception that aborts the run;
  never a silent skip that hides the failure rate.
- Cache visited points. A repeat visit returns the cached fitness and does **not** consume
  budget — this is how the reference implementation counts evaluations.
- Report the failure fraction in every search summary. It is a property of the space and
  a headline result.

### Neighbourhood

Use the *adjacent* definition: a neighbour differs in one dimension, moving to an
adjacent value in that dimension's ordered list (`128`'s neighbours in
`{16,32,64,128}` are `{64}`). Closely-related parameter values are related in performance;
full Hamming neighbourhoods waste budget.

### Config-space construction

Follow the production pattern: **the config list itself is device-dependent, evaluated at
import**, not filtered at runtime.

```python
BK_LIST = [32, 64] if smem_at_least("ampere") else [16, 32]
```

Then prune in three layers, as the Triton tutorial does:
1. A static `keep(cfg)` filter on invalid combinations.
2. A runtime `early_config_prune(configs, named_args, **kwargs)` using launch arguments
   — e.g. drop `BLOCK_M > N_CTX`, and `BLOCK_M < BLOCK_N` when causal. **Must return at
   least one config.**
3. A single-config collapse under a test env var, for reproducible tests.

## Level 2 — architectural (LLM proposer)

What the kernel *does*. Rare, expensive, high variance, and where the large wins are: in
one documented case three rounds of parameter tuning took 9.52 → 4.03 ms and the 2× came
from a fourth round that changed the design to one-row-per-program.

### The proposer's input

Not a raw `ncu` dump. A **diagnosis**, because interpreted profiler guidance beat raw
counters by 125% (p < 0.0001) in a controlled comparison. Assemble:

- The device calibration (SM count, shared memory budget, ridge point, launch overhead).
- The shape and its arithmetic intensity, and which side of the ridge it falls on.
- The current best kernel's source and its measured position: which ceiling, how far
  below the roof, how far left of the ridge.
- The top stall reason, translated (`long_scoreboard` → "global memory latency is not
  hidden; add in-flight loads, not more warps").
- Tensor-core utilization and register-spill flags — the two highest-value single signals.
- Up to three relevant `intents/` entries from the scout, with their source citations.
- The last N failures for this family with their diagnostics.

### The proposer's output

A **candidate**: complete kernel source plus a one-line rationale plus the intent it is
testing. Not a diff, not a config change — those belong to level 1.

Constrain it explicitly: *propose an architectural change, not a parameter tweak. If your
proposal only changes constexpr values, it is rejected and level 1 will do it better.*

### Parent selection — clade metaproductivity

Do not sample the best-performing node. Sample by the pooled success rate of a node's
entire descendant subtree, via Thompson sampling over a Beta posterior:

```python
def sample_parent(ledger):
    scores = {cid: beta_sample(1 + succ, 1 + fail)
              for cid, (succ, fail) in clade_stats(ledger).items()}
    return max(scores, key=scores.get)
```

A mediocre kernel that spawns good children is a good parent. Scoring nodes by their own
performance systematically loses those stepping stones.

## Objective

```
score = 0                                          if not correct
score = Σ_regimes  w_r · speedup_r                 if correct
```

where `w_r` is the regime's share of the announced shape matrix. The weighting is not
cosmetic: one published system's best speedups landed on operators worth 0.12–5.93% of
runtime while the operator worth 90% stayed below baseline.

Clip `speedup_r` at some cap (3× is the published choice) so one outlier regime cannot
dominate the objective.

**Correctness is a gate, not a term.** A failing candidate scores nothing, not "a bit
less". And a round that fixed a compile error without moving the timing is recorded as a
failure to improve — otherwise the loop drifts toward safe, slow, compiling code, which is
a measured phenomenon (compile rate 52.3% → 68.8% while mean speedup fell 1.58× → 1.44×).

## Termination

Loop-until-dry: stop when K consecutive rounds produce no candidate that beats the
best-known with non-overlapping confidence intervals. Start at K = 3. A fixed round count
misses the tail; a fixed improvement threshold stops too early on a plateau that a single
architectural jump would clear.

## Acceptance

1. 200 evaluations on the baseline kernel's space find a config at least as good as
   `@triton.autotune` over the same space, with every compile failure recorded.
2. The reported compile-failure fraction is non-trivial and plausible.
3. Ablate against random sampling over the same space and report the gap honestly.
4. A proposer round that returns only a constexpr change is rejected by the loop, not
   silently accepted.
5. Parent selection demonstrably samples a non-best node at least sometimes.
