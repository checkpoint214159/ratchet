> **HISTORICAL — NVIDIA/Triton competition scaffold. Not current, not an instruction.**
> Ratchet is now hardware-agnostic and hardware-gated (see `docs/hardware-support.md`).
> Single-GPU details below (e.g. `sm_120`, `wgmma`, A100/H100 tile budgets) are retained
> as historical design material, never as a general finding or a fixed target. See
> `docs/handoff-readme.md`.

# 03 — Research dossier

Papers and open implementations, with what to take from each and where the claim is soft.
Do not read these in order; read the row you need when you need it.

---

## A. Search strategy

### Schoonhoven, van Werkhoven, Batenburg — *Benchmarking optimization algorithms for auto-tuning GPU kernels* — arXiv 2210.01465, IEEE TEC

The empirical basis for `ratchet/search/`. 16 black-box optimizers, 9 GPUs, 3 kernels,
26 kernel spaces, with the full spaces brute-forced so optimality is known.

**Take:**
- **Dual annealing** at low budgets (≤ 200 evaluations); **FirstILS** (first-improvement
  iterated local search) above that. Both beat random sampling; not all algorithms do.
- **Treat GPU tuning as a deterministic problem** with mean runtime as fitness. Stochastic
  handling (irace, SMAC) needed higher budgets and did not consistently win; measured
  runtime variation was tiny (normalized 1.000 ± 0.011).
- **SMAC underperformed**, which the authors attribute to the high failure rate confusing
  the surrogate. Directly relevant to our Tier 2 critic: a surrogate over this space must
  model *failure* explicitly, not treat it as a bad score.
- **The failure rate is the design constraint.** Table II: convolution **68%** and GEMM
  **78%** of configurations fail to compile. Infeasible points must be cheap, recorded,
  and assigned a large finite fitness.
- Tuning difficulty varies enormously **across GPUs for the same kernel**, and does not
  track release date. Another argument for calibrated rather than transferred configs.
- **Fitness flow graphs** and PageRank centrality of local optima as a difficulty metric.
  Interesting; a stretch goal at best. Skip unless you have brute-forced a space.

**Soft:** the kernel spaces are 5–10 parameters and fully enumerable. Ours is not, and the
architectural level is not a black-box search space at all.

### Huxley-Gödel Machine — arXiv 2510.21614

**Take:** clade metaproductivity. Score a node by the pooled success of its entire
descendant subtree, sample parents by Thompson sampling over that. A node's own score is a
biased estimator of its value as an ancestor. Cheap to implement; use it in M7.

### Darwin Gödel Machine — arXiv 2505.22954

**Take:** the archive-of-all-agents idea (stepping-stone preservation) and — more usefully
— its documented failure: agents hallucinating tool use and **fabricating test logs**.
This is why the oracle is checksummed.

### Red Queen Gödel Machine — arXiv 2606.26294

The Tier 2 inspiration. See `docs/01-architecture.md` for what we take and what we drop.

**Take:** epoch-frozen evaluators, promotion only on ε-best-belief against a fixed
held-out anchor, ties to the incumbent, and the adversarial pool for a known evaluator
bias.

**The result that matters** is the ablation, not the headline: with a fixed critic the
writer saturated its validation set at 100% while collapsing to 78.2% on the anchor.
That is the Goodhart demonstration and the reason to make the evaluator move at all.

**Be skeptical.** The coding-domain result is **+3 tasks on n=166**, well inside binomial
noise at that n; the defensible claim there is token efficiency, not pass rate. The
component ablations (erasure, adversarial pool) have ±10% error bars and do not separate
statistically — only *evaluator replacement versus fixed* is unambiguous. The adversarial
term was hand-inserted in response to an observed bias, so it demonstrates that the epoch
mechanism gives you a place to put such a correction, not that co-evolution discovers one.

**Do not** transfer selective erasure of measurements. See the architecture doc.

### If you want evolving *tasks* rather than evolving *evaluators*

That is a different literature and RQGM is not it: POET (arXiv 1901.01753), Enhanced POET
(2003.08536), PAIRED (2012.02096), DéjàQ (2601.01931). They solve the collapse/explosion
problem with a difficulty band or a regret objective, which RQGM has no analogue for.
Out of scope here — the shape matrix is given.

---

## B. Agentic kernel generation — what actually works

### *Harness Engineering for LLM-Driven GPU Kernel Generation* — arXiv 2607.17979

**The most important paper in this section, and it is a negative result.** Agent-*assisted*
kernels beat fully-autonomous agent artifacts: "expert-provided optimization directions,
high-quality references, and workload context remain critical." It also argues the
**harness** — compile / verify / time, held separate from the optimization controller — is
the real engineering artifact. That is this project's thesis, and it is citable.

### KernelBench-Verified — arXiv 2607.16241

The catalogue of how kernel benchmarks lie. Every item in `docs/04-failure-modes.md`
traces here or to its companions. Read before writing measurement code.

### KernelPro — arXiv 2606.26453

**Take:** interpreting profiler counters into structured natural-language guidance beat
handing the model raw metrics by **125%** (p < 0.0001). Tensor-core utilization was the
highest-impact single signal; register-spill detection the most reliable. This shapes the
proposer's prompt: give it a *diagnosis*, not an `ncu` dump.

### KernelAgent — PyTorch blog, `meta-pytorch/KernelAgent`

**Take:** the artifact layout (`.optimize/<run_id>/output/best_kernel.py`), parallel seeds
per kernel (`NUM_KERNEL_SEEDS=4`), bounded refinement rounds. And the case study: three
rounds of tuning took 9.52 → 4.03 ms; the 2× came from a **design change** in round four.
Architectural moves, not parameter tweaks.

### CUDA Agent — arXiv 2602.24286 · Dr. Kernel — arXiv 2602.05885 · KernelBrain — arXiv 2608.02611

**Take, in order of usefulness:**
- Dr. Kernel's reward: correctness × clipped speedup × **the kernel's share of total
  runtime**, explicitly to stop the loop optimizing ops that do not matter.
- CUDA Agent's ablation: removing the agent loop collapsed faster-rate from 96.8% to
  14.1%. The gain is search-and-measure, not model knowledge. Also its anti-hack measures:
  eval scripts file-permissioned, fallbacks banned via context managers, forced sync.
- KernelBrain's multi-fidelity successive-halving rungs — cheap screening, then
  concentrated budget. 1.4× over KernelAgent at 48% less time. This is the Tier 2 critic's
  job in a different guise.

**Discount the absolute numbers.** CUDA Agent is measured on original KernelBench; read
it through the Verified correction.

### K-Search — arXiv 2602.19128

Co-evolves an LLM world model over a search tree of optimization *intents*. Closest single
paper to this design. Beat OpenEvolve (2.10× avg) and ShinkaEvolve (2.21×) on
FlashInfer-Bench. Read for the intent representation.

### Survey — arXiv 2601.15727

Best single entry point. Defines `fast_p`. Flags reward hacking explicitly. Living list at
`github.com/flagos-ai/awesome-LLM-driven-kernel-generation`.

### The skeptic's paper — arXiv 2602.16805

*Simple Baselines are Competitive with Code Evolution.* Read it before claiming the
evolutionary machinery is what produced a win. Ablate against random sampling over the
same space; if random matches, say so.

---

## C. Reference implementations to read and benchmark against

Ordered by reading value for this project.

### 1. Triton `06-fused-attention.py` — the vocabulary

`triton-lang/triton`, `python/tutorials/06-fused-attention.py`, ~776 lines. FA-2 forward
and backward.

**Read for:** the three-layer autotune pruning pattern — a static `keep()` filter, a
runtime `early_config_prune(configs, named_args, **kwargs)` using launch arguments, and
a `PYTEST_VERSION` collapse to a single config for reproducible tests. Steal all three.
Also `Config(pre_hook=...)` used to patch `TensorDescriptor.block_shape` per config, which
is the pattern when a config choice changes an *argument* rather than a constexpr.

**Know its limits before you copy it:** backward is not autotuned (hardcoded 32/128/128/32,
`num_warps=4`, `num_stages=5`); `assert N_CTX % 128 == 0` so no ragged sequences; dense
4-D only — no varlen, GQA/MQA, paged KV, sliding window, ALiBi, softcap or bias; FP8 is
forward-only `e5m2` and Blackwell-only for the non-transposed V path; **there is no
persistent variant** despite what several summaries claim.

### 2. vLLM Triton attention backend — dispatch without autotune

`vllm-project/vllm`. **Paths moved; most write-ups cite stale ones.** Current:
- kernel: `vllm/v1/attention/ops/triton_unified_attention.py` (~1189 lines)
- backend: `vllm/v1/attention/backends/triton_attn.py` (~837 lines)
- (`vllm/attention/ops/triton_unified_attention.py` now 404s)

**Read for:** the single most relevant design in open source to what we are building.
There is **no `@triton.autotune` on the kernel at all** — all shape dependence is a Python
decision tree in the launcher, derived from offline microbenchmarks. Study
`unified_attention(...)`: `BLOCK_M` from `num_queries_per_kv`, the `tuned_large_head`
special case gated on `is_device_capability_family(100)`, the 2D/3D (split-K) selection
with `seq_threshold_3D` **derived from the launch grid** via
`MIN_LAUNCH_GRID_SIZE_2D = 128 // num_heads_kv`. That last one is exactly our
occupancy-branch predicate, already written by someone else.

Vendor branching is via `vllm.platforms.current_platform` (`.is_cuda()`,
`.get_device_capability()`, `.is_device_capability_family(100)`), **not** the
`is_nvidia_gpu()` / `is_amd_gpu()` helpers quoted in the paper — those are a
research-branch formulation that was reshaped before merge.

Paper: *The Anatomy of a Triton Attention Kernel*, arXiv 2511.11581 — naive port at 19.7%
of FA-3, same source at 98.6–105.9% after tuning, 5.9× on MI300. The evidence that the
source ports and the tuning does not.

Companion microbenchmark framework: `foundation-model-stack/vllm-triton-backend`.
Autotune caching: `IBM/triton-dejavu` — drop-in `@triton_dejavu.autotune` with JSON
persistence via `TRITON_DEJAVU_STORAGE`, a `ConfigSpaces` object, and Bayesian search over
it. Read before building our own cache.

### 3. MSLK split-K attention — the analytic parallelism heuristic

`meta-pytorch/mslk` (xformers' `triton_splitk.py` is now a 13-line re-export shim into it).
- `mslk/attention/fmha/_triton/splitk_kernels.py` (~1219 lines)
- `mslk/attention/fmha/triton_splitk.py` (~1377 lines)

**Read for:** `FwOp.get_split_k(B, G, H, Mk, Mq, page_size, is_paged)`, a fully worked
split-K heuristic with divergent NVIDIA and HIP branches. The idea to port: *choose splits
so total CTAs ≈ a parallelism target, then halve until each split's chunk is large enough
to amortize.* Also `if Mq > 1 and B*G*H > 64: return 1` — do not split when there is
already enough parallelism. That is our occupancy branch, quantified.

Also notable: the kernel is not decorated; it is post-processed by
`unroll_varargs(..., N=num_groups)` then wrapped with `heuristics` and `autotune`
programmatically, with the autotuner cache exposed via `get_autotuner_cache` /
`set_autotuner_cache` so tuning results can be serialized and shipped.

### 4. flash-linear-attention — autotune hygiene and device introspection

`fla-org/flash-linear-attention`.

**Read `fla/utils/_device.py` first.** It is the best open-source answer to
self-calibrating dispatch: every query `@cache`d, every one wrapped in try/except with a
fallback, and capability expressed as *"do I have at least as much shared memory as an
A100"* (`check_shared_mem('ampere')`) rather than a compute-capability comparison — which
degrades correctly onto hardware nobody has enumerated. Copy this file's approach.

Then `fla/ops/gla/chunk.py` for autotune patterns: the config *list itself* is
device-dependent (`BK_LIST = [32,64] if check_shared_mem() else [16,32]`, evaluated at
import); `@triton.heuristics({'IS_VARLEN': ...})` sits **above** `@triton.autotune` so
each branch gets its own tuned config; `@triton.jit(do_not_specialize=['T'])` stops
recompilation per sequence length; and a custom prune rule that is correctness-adjacent,
not just performance ("avoids a software-pipelined block load prefetching past the
tensor").

Also note the version-churn defence, which we should copy verbatim:
```python
SUPPORTS_AUTOTUNE_CACHE = "cache_results" in inspect.signature(triton.autotune).parameters
```

### 5. GPU MODE reference-kernels — the harness to imitate

`gpu-mode/reference-kernels`, especially `problems/pmpp_v2/eval.py` (~375 lines) and
`utils.py`.

**Take wholesale:**
- **Adaptive stopping on relative standard error** — stop when `std/sqrt(runs)/mean < 0.001`,
  or a wallclock cap. Tight error bars on fast kernels, no hour wasted on slow ones.
- **Process isolation per case** (`pool.apply`) so a crash does not poison the run.
- `clear_l2_cache()` before *every* repetition.
- Correctness sizes 127/128/129; benchmark sizes disjoint.
- `verbose_allclose` reporting the first N mismatches with indices, not a bare boolean —
  makes agent feedback dramatically more useful.
- `DeterministicContext` managing `cudnn.allow_tf32`, `cudnn.deterministic`,
  `use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`, restoring on exit.
- The output buffer is passed *in*, so allocation is outside the measurement.

**Licence caveat:** June 9 Researcher Reciprocity License, with an explicit restriction on
using the repo to improve an AI model or service. Read it before vendoring code.

### 6. KernelBench timing — multi-backend agreement

`ScalingIntelligence/KernelBench`, `src/kernelbench/timing.py` (~626 lines).

**Take:** five timing backends behind `get_timing_function(method)` — `cuda_event`,
`do_bench`, `do_bench_impl` (a vendored `do_bench` with auto-sizing commented out so
repeat counts are yours), `host_time`, `nsight_python_time`. Having both `do_bench` and a
controllable variant in one place is exactly how you show a result is not an artifact of
the timing method. Also their L2 sizing note: A100 40 MB, H100 50 MB, H200 90 MB,
RTX 4090 72 MB, L40S 48 MB, Blackwell ≈192 MB — so flush with > 200 MB, or size from
`props.L2_cache_size`.

Companion: `simonguozirui/simple-torchroofline` for analytical speed-of-light bounds with
no hardware — good for sanity-checking whether a claimed speedup is physically possible.

### 7. Fused-op references — where the non-attention wins are

- **Liger Kernel** `linkedin/Liger-Kernel`, `src/liger_kernel/ops/`: `rms_norm.py`,
  `rope.py`, `swiglu.py`, `fused_linear_cross_entropy.py`. Note `casting_mode` as a
  `tl.constexpr` so the upcast branches compile out — clean constexpr-driven variant
  selection instead of separate kernels.
- The de-facto standard row-kernel launch heuristic, shared by Liger and Unsloth
  (`src/liger_kernel/ops/utils.py::calculate_settings`): `BLOCK_SIZE = next_power_of_2(n)`,
  then a warp ladder 4 → 8 → 16 → 32. No autotune. **Benchmark our dispatch against this
  as the cheap baseline** — if a warp ladder matches our calibrated dispatch, say so.
- **Unsloth** `unsloth/kernels/` — same territory, finetuning-shaped, plus LoRA fused into
  the projections and NF4 dequant fused into the matmul.

### 8. Also worth knowing

- **FlashInfer** — *not* a Triton attention library; its attention is CUDA/CUTLASS/cuDNN,
  JIT-compiled. But `flashinfer/triton/kernels/cascade.py` is the log-sum-exp state-merge
  math for split-KV, and is **the most reusable single file** if you write your own
  split-K reduction. Also study its plan/run split — validate and compile in `plan()`,
  keep `run()` cheap and CUDA-graph-capturable.
- **ROCm/aiter** `aiter/ops/triton/` — offline-tuned JSON keyed by shape bucket
  (`configs/<arch>/<backend>/<op>/<dtype>/`), resolved by a function returning
  `(config, is_tuned)` so callers can *detect shapes running on untuned defaults*. That
  return signature is worth copying even if we never touch AMD.
- **`gpu-mode/triton-index`** — curated index of Triton kernels; `kernel_overview.md`.
- **`BobMcDear/attorch`** — readable pure-Triton reimplementation of a `torch.nn` subset,
  forward and backward. Reading resource only; README pins torch 2.4 / triton 3.0, so
  assume it is stale.

---

## D. API facts that churn and are easy to get wrong

```python
# Triton, python/triton/testing.py
do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean")
do_bench_cudagraph(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="mean")
do_bench_proton(fn, warmup=25, rep=100, ...)            # better for short kernels
do_bench_cudagraph_proton(fn, rep=20, ...)
```

- `warmup` and `rep` are **milliseconds of budget, not iteration counts**. Iteration counts
  are derived from a 5-iteration estimate.
- `do_bench` flushes L2 between reps via `driver.active.clear_cache(cache)` (a 256 MB
  `cache.zero_()`); **`do_bench_cudagraph` does not**, and it captures `n_repeat` unrolled
  calls into one graph so launch overhead is amortized away. The two measure different
  things. Never compare them directly; do subtract them to get launch overhead.
- `quantiles` **overrides** `return_mode`.
- Graph capture is wrapped in `cuda_graph_without_gc()` — a Triton `CompiledKernel`
  finalized by Python's cyclic GC unloads its CUDA module, which is illegal during stream
  capture and silently invalidates the graph. Replicate this if you capture your own.

```python
# Triton autotune, python/triton/runtime/autotuner.py
autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None,
         pre_hook=None, post_hook=None, warmup=None, rep=None,
         use_cuda_graph=False, do_bench=None, cache_results=False)
Config(kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None,
       pre_hook=None, ir_override=None)
```
`warmup`/`rep` here are deprecated — pass a custom `do_bench`. `early_config_prune` has
signature `(configs, named_args, **kwargs) -> List[Config]` and **must return at least
one**. `TRITON_PRINT_AUTOTUNING=1` shows the winner and the tuning cost.

```python
# Device properties — note the different spellings
triton.runtime.driver.active.utils.get_device_properties(0)
#   -> max_shared_mem, max_num_regs, multiprocessor_count, warpSize,
#      sm_clock_rate, mem_clock_rate, mem_bus_width          (exactly 7 keys)
torch.cuda.get_device_properties(0)
#   -> ..., multi_processor_count, L2_cache_size, shared_memory_per_block_optin, ...
```
`L2_cache_size` exists only on the torch side and is the one you want for sizing the
flush buffer. Theoretical bandwidth from `mem_clock_rate` is unreliable on HBM3 parts —
**always** cross-check against a measured number.
