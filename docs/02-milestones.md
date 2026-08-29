> **HISTORICAL — NVIDIA/Triton competition scaffold. Not current, not an instruction.**
> Ratchet is now hardware-agnostic and hardware-gated (see `docs/hardware-support.md`).
> Single-GPU details below (e.g. `sm_120`, `wgmma`, A100/H100 tile budgets) are retained
> as historical design material, never as a general finding or a fixed target. See
> `docs/handoff-readme.md`.

# 02 — Milestones and acceptance gates

Each milestone has a gate. Run the gate before claiming the milestone. Do not start
milestone N+1 with milestone N's gate red.

Rough sizing assumes one agent working with a human available for decisions. `[H]` marks
a step that needs a human answer.

---

## Tier 0 — trustworthy measurement

### M0 — Environment and hardware truth `[H]` · ~1 h

- Fill in **Target hardware** in `docs/00-mission.md`.
- Confirm `torch.cuda.is_available()`, record torch / triton / CUDA versions.
- Verify Triton compiles and runs a trivial kernel.
- Attempt `sudo nvidia-smi -pm 1` and `-lgc`. Record whether clock locking works. In WSL
  it often does not; that is fine, but the report must say so.
- `[H]` If the GPU is not what the plan assumed — different MMA family, much smaller
  shared memory — flag it before proceeding.

**Gate:** the hardware table is filled in and every field is a measured or queried value,
not an assumption.

### M1 — Oracle: device calibration · ~1 h

Copy in `seed/ratchet/oracle/device.py`. Complete and run the calibration.

- Query SM count, shared memory optin, L2 size, warp size, compute capability.
- **Measure** achievable bandwidth with a large streaming copy (working set ≥ 4× L2,
  minimum-of-N). Do not trust the clock-rate formula; `MEMORY_CLOCK_RATE` is unreliable
  on HBM3 parts.
- **Measure** kernel launch overhead as `do_bench(trivial) − do_bench_cudagraph(trivial)`.
- Derive the ridge point from a peak-FLOPs table plus measured bandwidth.
- Cache to `ledger/device.json` keyed by device name and driver version.

**Gate:** measured bandwidth is within 20% of the datasheet figure, and launch overhead
lands in the 1–10 µs range. Numbers wildly outside these mean the measurement is wrong,
not the hardware.

### M2 — Oracle: correctness and inputs · ~2 h

Copy in `seed/ratchet/oracle/{inputs,reference,correctness}.py`.

- Shape matrix from the competition statement, split into **disjoint** correctness and
  benchmark sets. Correctness sizes off-by-one around powers of two.
- Four input distributions per correctness case, plus the adversarial cases.
- Reference attention in FP64 for the error floor, plus the baseline family.
- The gate: locked tolerances, non-finite propagation, determinism, rich diagnostics.
- Seed three known-bad kernels in `oracle/known_bad/`: one returning its input, one with a
  subtly wrong scale factor, one correct only at a single benchmarked shape.

**Gate:** the reference passes against itself with error at the FP64 floor, **and all
three known-bad kernels are rejected** with a diagnostic naming why. If a known-bad
kernel passes, the gate is broken and nothing downstream is trustworthy.

### M3 — Oracle: timing and harness, plus one baseline kernel · ~3 h

Copy in `seed/ratchet/oracle/{timing,harness}.py`.

- Timing backends: `cuda_event`, `do_bench`, `do_bench_impl` (controllable repeats),
  `cudagraph`, `host`. Each records its own method descriptor.
- Adaptive stopping on relative standard error (stop at 0.1% SEM, or a wallclock cap).
- Subprocess isolation per candidate.
- Write one hand-authored Triton attention kernel — start from the Triton tutorial
  `06-fused-attention.py`, do not invent — and take it through the whole pipeline.

**Gate — the Tier 0 gate.** All of:
1. The baseline kernel passes correctness on every shape in the correctness matrix.
2. Timing the *same* kernel twice gives overlapping confidence intervals.
3. Timing the reference *against itself* gives a speedup of 1.00 ± noise. If this is not
   1.0, the harness has a bias and everything after it is contaminated.
4. Two different timing backends agree to within 10% on a kernel longer than 50 µs.
5. `scripts/check-oracle.sh` is green.

---

## Tier 1 — the search loop

### M4 — Ledger · ~1 h

Copy in `seed/ratchet/ledger.py`. Append-only JSONL, the schema in
`specs/02-ledger.md`, plus derived-view builders (best-known table, per-regime rankings)
that are pure functions of the ledger.

**Gate:** kill the process mid-write and the ledger is still parseable. Rebuild every
derived view from scratch and get identical output.

### M5 — Parametric search · ~3 h

`ratchet/search/`. Dual annealing at ≤ 200 evaluations, FirstILS above. Infeasible points
get a large finite fitness and a ledger row, never an exception.

**Gate:** run 200 evaluations on the baseline kernel's config space. The search finds a
config at least as good as `@triton.autotune` over the same space, having recorded every
compile failure. Report the failure fraction — expect it to be large.

### M6 — Self-calibrating dispatch · ~3 h

`ratchet/dispatch/`. Predicates as functions of the M1 calibration, per
`specs/04-dispatch.md`. Four branches: launch-bound, occupancy-bound, bandwidth-bound,
compute-bound.

**Gate:** artificially halve the SM count and the measured bandwidth in the calibration
cache and the dispatch decisions change *in the direction the roofline predicts*. A
dispatch that does not respond to device properties is a hardcoded table wearing a
costume.

### M7 — The propose–measure loop · ~4 h

Wire it together. Proposer draws from `intents/`, emits candidates; harness measures;
ledger records; selection promotes only on non-overlapping confidence intervals; parent
sampling by clade metaproductivity.

**Gate — the Tier 1 gate.** The loop runs unattended for one hour. At the end: the
best-known table has improved on at least one regime, no ledger row was deleted or
modified, and `scripts/check-oracle.sh` is still green.

### M8 — Report generator · ~2 h

`ratchet/report/`. Everything from the ledger, nothing hand-entered. Dispatch table with
margins, roofline positions, search trace, weighted end-to-end number, and an explicit
list of regimes where the vendor path still wins.

**Gate:** delete the report and regenerate it from the ledger alone; it is identical.

---

## Tier 2 — co-evolution

### M9 — Adversarial input pool · ~2 h

After each epoch, harvest inputs where the current best passed but disagreed with the
reference under a wider sweep. Append permanently to the correctness suite.

**Gate:** the pool catches at least one real bug in a candidate that the original suite
passed. If it never fires, either the suite was already adequate or the sweep is too
narrow — determine which and say so.

### M10 — The critic · ~6 h

Per `specs/05-critic.md`. Predicts compile failure, correctness failure and a speedup
band from source plus shape plus device. Epoch-frozen. Promoted only on a held-out slice
of the ledger, held out **by candidate**, with known-bad kernels present.

**Gate:** on held-out data the critic's false-negative rate on *good* kernels is below a
stated bound (start at 5%), and `candidates_pruned × mean_eval_cost > critic_overhead`.
If the critic cannot pay for itself, report that as a finding and turn it off — that is a
legitimate and publishable negative result.

### M11 — The scout · ~3 h

Per `specs/06-scout.md`. Periodically reads reference implementations from the dossier and
emits *architectural intents*, not configs, into `intents/`.

**Gate:** at least one scout-originated intent produces a candidate that beats the
best-known on some regime, and the ledger can trace that candidate back to the specific
implementation that inspired it.

---

## Deliverable milestones (do not skip)

### M12 — Tech report · ~4 h

Environment, methodology, what was measured and how, the dispatch table, the search
statistics, the ablations, the AI tooling used, and an honest limitations section. The
methodology section is the highest-value part: no one else will have one.

### M13 — README and reproduction · ~2 h

Setup, exact reproduction steps, limitations, what you would do with more time.

### M14 — Demo video · ~2 h

End to end: the loop running, the ledger growing, the dispatch responding to a changed
device profile, the report generating itself.

---

## If you are running out of time

Cut in this order, and say in the report what was cut:

1. M11 scout — document the design, do not build it.
2. M10 critic — same. A documented, unbuilt Tier 2 with an honest note beats a
   half-working one that inflates the numbers.
3. M5 search sophistication — fall back to `@triton.autotune` over a pruned space.
4. M7 unattended running — run the loop by hand.

Never cut: M2, M3, M4, M8. The oracle, the ledger and the report *are* the submission.
