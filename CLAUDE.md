# Ratchet — durable agent rules

This file is in context on every turn. It is the contract, not a summary.
Orientation lives in `HANDOFF.md`; the reasoning behind these rules lives in
`docs/01-architecture.md` and `docs/04-failure-modes.md`.

## What this project is

An agentic harness that continuously proposes, measures, and improves Triton attention
kernels for a specific GPU, keeping a permanent record of every measurement ever taken.
Built for the TikTok TechJam 2026 GPU kernel track and intended to outlive it.

## Zones

The repository is divided into three zones with different rules. Know which zone you are
editing before you edit.

**Zone A — `ratchet/oracle/` — IMMUTABLE.**
Reference implementations, input generation, the correctness gate, the timers, device
introspection. You may read and call these. You may not modify them during optimization
work. Changes require the user to explicitly ask, on a separate branch, with the checksum
manifest regenerated deliberately. `scripts/check-oracle.sh` fails the build otherwise.

Rationale: this is the only thing standing between the search loop and reward hacking.
If the thing being optimized can edit the thing doing the measuring, every number the
system produces is meaningless. This is not a style preference.

**Zone B — `ratchet/kernels/`, `ratchet/dispatch/`, `ratchet/search/`, `ratchet/critic/` — EVOLVABLE.**
Write freely. This is what the loop optimizes.

**Zone C — `ledger/` — APPEND-ONLY.**
`ledger/measurements.jsonl` is never rewritten, never sorted in place, never pruned.
Derived views (rankings, best-known tables, critic training sets) are rebuilt from it and
may be deleted at will.

## Hard rules

1. **Never widen a tolerance to make a kernel pass.** `REL_TOL = 0.02`, `ABS_TOL = 0.002`
   are constants in `oracle/correctness.py`. If a candidate fails, the candidate is wrong.
   Report it; do not negotiate with it.

2. **Never special-case a benchmarked shape.** Shape-conditional dispatch is the point of
   this project and is legitimate. Detecting the benchmark harness, hardcoding an output,
   or branching on a shape purely because it appears in the timed set is fraud. The line:
   a branch must be justifiable from device properties and arithmetic intensity, and must
   generalize to a shape not in the test matrix.

3. **Correctness before timing, always, in that order, in the same process as the timing.**
   A candidate that has not passed correctness on this exact input is not timed.

4. **Every timed number carries its method.** Record which timer was used, whether L2 was
   flushed, whether clocks were locked, the clock they were locked to, the number of
   repeats, and the standard error. A speedup without this metadata is not a result and
   must not be written to the ledger.

5. **The baseline is `torch.compile(mode="max-autotune")` with TF32 enabled**, plus the
   best available vendor path (`F.scaled_dot_product_attention` across all four backends).
   Never eager FP32 with TF32 off. Roughly half of all published kernel speedups are an
   artifact of getting this wrong.

6. **Report peak memory alongside every speedup.** `torch.cuda.max_memory_allocated()`.
   A faster kernel that raises peak memory is often not an improvement.

7. **No `try/except` around a correctness failure that lets the run continue as a pass.**
   Failures are data. Record them; they are the critic's training signal and the most
   valuable thing in the ledger after the wins.

8. **When in doubt about whether something is a legitimate optimization or a benchmark
   exploit, stop and ask the user.** The heuristic: would this still be a win if the
   grader changed the input distribution and the shape list without telling you?

## Working style

- Prefer editing one component and running its acceptance test to broad refactors.
- Every milestone in `docs/02-milestones.md` has an acceptance gate. Run it before
  claiming the milestone.
- Write the *why* in comments for anything measurement-related. Six weeks from now the
  reason a `torch.cuda.synchronize()` is on a particular line will not be obvious.
- Keep run artifacts under `ledger/artifacts/<candidate_id>/`. Never in the source tree.
- Long autotuning runs go through `scripts/run-loop.sh` with output to `ledger/logs/`,
  not interactively.

## Environment

- WSL2, Ubuntu, NVIDIA GPU. Native PowerShell is not supported by any of this.
- Python 3.11+, PyTorch, Triton. Pin versions in `pyproject.toml` once confirmed and
  record them in every ledger row — a measurement is only meaningful against a toolchain.
- Lock clocks before any benchmarking session: `sudo nvidia-smi -pm 1` then
  `sudo nvidia-smi -lgc <clock>`. Record the clock. If you cannot lock clocks (common in
  WSL), say so explicitly in the report and use minimum-of-N rather than mean.

## Things that are true and easy to forget

- `triton.testing.do_bench(fn, warmup=25, rep=100)` — those are **milliseconds of budget,
  not iteration counts**. It flushes L2 between reps. `do_bench_cudagraph` does **not**
  flush L2 and amortizes launch overhead into the graph; the two are not comparable.
- `torch.cuda.get_device_properties(0).multi_processor_count` (torch) vs
  `triton.runtime.driver.active.utils.get_device_properties(0)['multiprocessor_count']`
  (triton) — different spellings, same number.
- Kernel launch overhead is 1–5 µs. If the kernel runs in less than about 20 µs, you are
  measuring the launch.
- `shared_memory_per_block_optin` is the real budget and requires
  `cudaFuncSetAttribute` equivalent; the default 48 KB limit is not what you have.
