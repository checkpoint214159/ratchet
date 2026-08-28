# Spec 01 — Measurement core (the oracle)

**Zone A. Immutable during optimization.** Seed code in `seed/ratchet/oracle/`.

## Contract

```python
run_candidate(candidate: CandidateRef, shape: Shape, cfg: MeasureConfig) -> Measurement
```

Runs in a **subprocess**. Correctness first, timing second, profiling optionally third.
Returns a fully-populated `Measurement` (schema in `specs/02-ledger.md`) or a
`Measurement` with `status != "ok"` and a diagnostic. It never raises into the caller and
never returns a timing for a candidate that failed correctness.

## Modules

### `device.py`

Queries and **calibrates**. Cached to `ledger/device.json`, keyed by
`(device_name, driver_version, torch_version, triton_version)`.

Queried: `multi_processor_count`, `shared_memory_per_block_optin`, `L2_cache_size`,
`warp_size`, `major`/`minor`, `total_memory`, `regs_per_multiprocessor`.

Measured, not assumed:
- **Achievable bandwidth** — large streaming op with working set ≥ 4× L2, swept over
  sizes, `min` of N (not mean: throttling only makes samples worse). Expect 80–90% of
  datasheet.
- **Launch overhead** — `do_bench(trivial) − do_bench_cudagraph(trivial)`, in µs.
- **Peak dense BF16 TFLOP/s** — from a small table keyed on compute capability, since
  there is no reliable runtime query. Flag `peak_source: "table"` in the record so it is
  never mistaken for a measurement. On GeForce parts note the FP32-accumulate halving.

Derived: `ridge_point = peak_flops / measured_bandwidth`, in FLOP/B.

Every query is `@cache`d and wrapped with a fallback. Follow `fla/utils/_device.py`:
express capability as *"at least as much shared memory as an A100"* rather than a
compute-capability comparison, so it degrades correctly on unenumerated hardware.

### `inputs.py`

Deterministic. `generate(shape, seed, distribution) -> tuple[Tensor, ...]` using an
explicit `torch.Generator(device="cuda")` seeded per case.

**Two disjoint shape sets**, declared as constants:
- `CORRECTNESS_SHAPES` — off-by-one around powers of two (…, 127, 128, 129, …) to catch
  masking and tail bugs.
- `BENCHMARK_SHAPES` — the competition matrix, no overlap with the above.

**Four distributions per correctness case**, minimum: `standard` (N(0,1)), `scaled_up`
(×3), `scaled_down` (×0.01), `negated` (×−1). Plus adversarial cases:
denormals, values near the dtype maximum, non-contiguous strides, a ragged tail batch,
and one case with a single Inf and one with a single NaN to check propagation.

The output buffer is allocated by the harness and **passed in**, so allocation is outside
the measurement.

### `reference.py`

- `reference_fp64(...)` — the error floor. Slow, exact enough.
- `reference_fp32(...)` — the semantic reference.
- `baseline_family()` — returns a dict of named callables, each timed independently:
  - `eager_tf32` — `torch.set_float32_matmul_precision("high")`
  - `compile_max_autotune` — `torch.compile(mode="max-autotune")`
  - `sdpa_flash`, `sdpa_mem_efficient`, `sdpa_cudnn`, `sdpa_math` — each forced explicitly
    with `torch.nn.attention.sdpa_kernel(..., set_priority=True)`, because the default
    dispatch heuristic is known to choose badly

The reported baseline for any shape is the **best** member. Record which one won; that
table is itself a result worth publishing.

### `correctness.py`

```python
REL_TOL = 0.02      # LOCKED. From the competition spec. Never a parameter.
ABS_TOL = 0.002     # LOCKED. Binding constraint on order-1 outputs.
```

`check(got, expected, *, ctx) -> CorrectnessResult` — not a boolean. Gates, all of which
must pass:

1. **Tolerance** — elementwise `|got − exp| <= ABS_TOL` **and**
   `|got − exp| <= REL_TOL * |exp|`. Report the first N mismatches with indices and
   values, in the style of GPU MODE's `verbose_allclose`; a bare boolean makes the agent
   feedback loop far weaker.
2. **Non-finite propagation** — where the reference is NaN or Inf, the candidate must be
   too. Silently replacing non-finites with ordinary numbers is the single most common
   defect in machine-generated kernels.
3. **Determinism** — run twice on identical input; bitwise identical, or if the kernel
   legitimately uses atomics, within a stated and recorded bound.
4. **Distribution coverage** — all four distributions must pass, reported separately so
   a distribution-specific failure is visible.
5. **Shape generalization** — passes on at least one shape outside `BENCHMARK_SHAPES`.

Also here: `DeterministicContext` managing `cudnn.allow_tf32`, `cudnn.deterministic`,
`torch.use_deterministic_algorithms`, and `CUBLAS_WORKSPACE_CONFIG`, restoring on exit.

### `timing.py`

Five backends behind `get_timer(method)`, each returning `(TimingStats, MethodDescriptor)`:

| method | flushes L2 | includes launch | notes |
|---|---|---|---|
| `cuda_event` | yes | yes | explicit warmup/trials/discard_first — the default |
| `do_bench` | yes | yes | Triton's; `warmup`/`rep` are **ms of budget** |
| `do_bench_impl` | yes | yes | vendored copy with auto-sizing removed, repeats are yours |
| `cudagraph` | **no** | **no** | amortizes launch into the graph; not comparable to the others |
| `host` | yes | yes | wall clock, for cross-checking |

`TimingStats`: `runs, mean_ns, std_ns, sem_ns, min_ns, max_ns, p50, p20, p80`.
`MethodDescriptor`: method name, warmup, repeats, L2-flush flag, flush buffer bytes,
clock-locked flag, locked clock MHz, torch/triton versions.

**Adaptive stopping.** After ≥ 3 runs, stop when `sem/mean < 1e-3`, or `mean × runs`
exceeds a per-case budget, or 120 s wallclock — whichever first. This gives tight bars on
fast kernels without burning an hour on slow ones.

**L2 flush** sized from `props.L2_cache_size`, minimum 256 MB, written (not just
allocated) before every timed repetition.

**Short-kernel warning.** If `mean_ns < 5 × launch_overhead_ns`, set
`warning: "launch_dominated"` on the measurement. Do not silently report it as a kernel
time.

### `harness.py`

Orchestrates one candidate. In order:

1. Fork a subprocess (`spawn`, fresh CUDA context).
2. Import the candidate. A failure here is `status="compile_error"` with the traceback —
   recorded, not raised. Expect this to be the majority outcome; comparable spaces show
   68–78% compile failure.
3. Run correctness on all distributions and correctness shapes. Any failure →
   `status="incorrect"` with the diagnostic, **no timing**.
4. Time on `BENCHMARK_SHAPES` with the configured backend, plus one cross-check with a
   second backend on the longest shape.
5. Record `torch.cuda.max_memory_allocated()`.
6. Optionally profile (`ncu` if available) and attach interpreted metrics, not raw
   counters — interpreted guidance beat raw counters by 125% in one controlled study.
7. Emit the `Measurement`. The parent appends it to the ledger.

Timeout every stage. A hung candidate is `status="timeout"`, which is data.

## Acceptance

1. Reference against itself: speedup 1.00 ± noise. **If this is not 1.0 the harness has a
   bias and every downstream number is contaminated.**
2. All three seeded known-bad kernels in `oracle/known_bad/` are rejected, each with a
   diagnostic naming the reason.
3. Two timing backends agree within 10% on a kernel longer than 50 µs.
4. Repeat measurement of one kernel gives overlapping confidence intervals.
5. A deliberately hanging kernel yields `status="timeout"` without killing the loop.
6. `scripts/check-oracle.sh` green.
