> **HISTORICAL — NVIDIA/Triton competition scaffold. Not current, not an instruction.**
> Ratchet is now hardware-agnostic and hardware-gated (see `docs/hardware-support.md`).
> Single-GPU details below (e.g. `sm_120`, `wgmma`, A100/H100 tile budgets) are retained
> as historical design material, never as a general finding or a fixed target. See
> `docs/handoff-readme.md`.

# 04 — How kernel benchmarks lie

Read this before writing any measurement code. Every item is a documented failure from
the 2025–2026 literature, not a hypothetical. The design of `ratchet/oracle/` is a
point-by-point response to this list.

## The weak baseline

**What happened.** The original KernelBench compared generated kernels against FP32
PyTorch with TF32 tensor cores *disabled* — an optimization every practitioner has on by
default. Fixing only this accounted for roughly **47% of apparent gains**, and more than
1.5× inflation on the 24% of tasks that were compute-bound. Under a corrected protocol
the best frontier model went from a reported **1.43× to 0.88×** — below PyTorch.

**Response.** `oracle/reference.py` establishes a baseline *family* and always reports
against the strongest member:

- eager FP32, TF32 **on** (`torch.set_float32_matmul_precision("high")`)
- `torch.compile(mode="max-autotune")`
- `F.scaled_dot_product_attention` benchmarked across **all four** backends explicitly,
  because the default dispatch heuristic is known to pick wrong (`pytorch#138907` is open
  precisely because nobody has produced the shape-by-shape table)

Any speedup quoted against anything weaker is not a result.

## The warm cache

**What happened.** Back-to-back iterations on the same tensors serve from a 40–126 MB L2
and show 2× that vanishes in situ.

**Response.** Flush L2 before **every** timed repetition, with a buffer sized from
`torch.cuda.get_device_properties(0).L2_cache_size` rather than a hardcoded 256 MB.
Note that `triton.testing.do_bench` flushes and `do_bench_cudagraph` does **not** — the
two are not comparable and both must record which they were.

## The loose tolerance

**What happened.** Hand-picked `allclose` tolerances typically run one to three orders of
magnitude looser than the kernel's actual error envelope. In one study a fixed-shape
`allclose` oracle certified **9 of 9 deliberately seeded buggy kernels as correct**.
A separate audit of 2,638 machine-generated kernels *already accepted by their source
system* found a majority carried at least one contract violation under adversarial gates,
the most common being **silently replacing NaN/Inf with ordinary numbers**.

**Response.** Tolerances are locked constants matching the competition spec, and the gate
checks more than `allclose`: non-finite propagation, determinism across repeats, and
generalization to shapes outside the tuned set.

## The friendly input distribution

**What happened.** Test inputs drawn only from uniform `[0, 1)` let a model hardcode
"all values are positive." The canonical example: a ReLU kernel that returns its input
unchanged, reported at **374×**.

**Response.** `oracle/inputs.py` ships four distributions for every correctness case —
standard, scaled ×3, scaled ×0.01, and negated — plus adversarial cases with denormals,
values near overflow, and non-contiguous strides. The distribution set is part of the
locked oracle.

## The shape you tuned on

**What happened.** Tuning against the shapes you validate on measures your own tail.

**Response.** Following the GPU MODE reference-kernels design: correctness sizes are
deliberately off-by-one around powers of two (127 / 128 / 129) to catch masking and tail
bugs, and **benchmark sizes are a disjoint set**. Both are declared in the oracle and are
not visible to the proposer as tunable.

## The memory blowup

**What happened.** 28% of the best model's kernels in one corrected evaluation *increased*
peak GPU memory, making them undeployable regardless of speed.

**Response.** `torch.cuda.max_memory_allocated()` is recorded on every measurement and
appears next to every speedup in the report. A win that costs peak memory is flagged.

## The launch you forgot to measure

**What happened.** A kernel benchmarked inside a CUDA graph and one benchmarked standalone
measure different things; launch overhead is 1–5 µs and vanishes in the former. Also:
without `torch.cuda.synchronize()` in the right place, an async launch means the timer
measures nothing at all.

**Response.** Every timing backend in `oracle/timing.py` records its own method
descriptor, and the harness records the difference between graph and non-graph timing as
the measured launch overhead — which then feeds the dispatch decision, since the
launch-bound regime is a real branch.

## The clock that moved

**What happened.** An unlocked GPU boosts on short benchmarks and throttles on long ones.
A "1.4× speedup" measured against an unlocked baseline is not a result.

**Response.** Lock clocks and record the value. WSL frequently cannot lock clocks — if
so, say it explicitly in the report, use minimum-of-N rather than mean, and interleave
candidate and baseline timings in the same process so thermal drift affects both.

## The refinement that made things worse

**What happened.** Iterative refinement repairs but does not optimize. One benchmark
measured compile rates rising 52.3% → 68.8% across refinement rounds while **average
speedup fell 1.58× → 1.44×**. Without an explicit speed reward the loop drifts toward
safe, slow, compiling code.

**Response.** The objective is explicit and two-term: correctness is a gate, and among
passing candidates the score is speedup weighted by the shape's share of the workload.
A "fixed the compile error" round that does not move the timing is recorded as a failure
to improve, not as progress.

## The op that did not matter

**What happened.** One system's best speedups landed on operators that were 0.12%–5.93%
of total runtime, while `linear` at **90.13% of runtime stayed below baseline**.
Operator-level microbenchmark wins are not end-to-end wins.

**Response.** The dispatch report weights every regime by its share of the announced shape
matrix, and the headline number is the weighted end-to-end figure, never the best
individual kernel.

## The vendor library you cannot beat

**What happened.** Of 24 open-source-implemented operators, 13 beat eager — but only
**1 of 9 vendor-backed operators** did.

**Response.** The dispatch table is allowed, and expected, to select the vendor path for
regimes where it wins. Recording "cuDNN wins here by 12%, we fall back" is a *result*, and
a report that claims a win in every regime is a report nobody will believe.

## The circular anchor

**What happened.** Not yet, to us. This is the failure mode specific to Tier 2: if the
learned critic is promoted against labels generated by a harness the loop can edit, the
promotion rule is circular and provides no protection at all.

**Response.** The oracle is checksummed, immutable, and runs in a separate process, and
the anchor includes hand-seeded known-bad kernels so a lenient critic is actively
penalised rather than merely unrewarded. See `docs/01-architecture.md`.
