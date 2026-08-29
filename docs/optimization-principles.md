# Optimization principles

These are the standing rules for any optimization work in Ratchet, automated or human.
They are hardware-agnostic: they hold whichever GPU is qualified at runtime. Per-GPU
tuning results and measured device data may be stored, but the *principles* below are not
specialized to any one device.

## The benchmark is the authority, not intuition

Every optimization effort must be justified by profiling or benchmark evidence, never by
"this kernel should be faster." Before spending an experiment, ask:

1. What is currently limiting performance?
2. Is this component actually on the critical path?
3. Is there meaningful headroom?
4. Is the proposed change likely to move end-to-end performance?
5. Did the change actually improve the benchmark under the documented methodology?

A faster microbenchmark does not imply a faster transformer. A first improvement is not
trusted until it survives variance, warm-up, synchronization, and repeated measurement.

## Do not blindly optimize expensive-looking operations

Some operations are already heavily optimized by vendor libraries and mature GPU stacks.
Manually reimplementing a well-tuned matmul commonly yields no gain, can regress, and
wastes research time. Replace a generic implementation only when evidence shows headroom
on the critical path.

## Optimization classes to consider

This list is indicative, not exhaustive; the right technique is discovered empirically.

- **Kernel fusion** — combine operations when it reduces memory traffic, intermediate
  tensors, or launch overhead. Do not assume manual fusion beats existing compiler
  fusion; benchmark it.
- **Kernel selection** — substitute specialized kernels, hardware primitives, or vendor
  libraries for generic implementations.
- **Algorithmic change** — change tiling, reduction strategy, or the computation itself to
  reduce memory movement or unnecessary work.

## What "better" means

Judged by measurable evidence, prioritizing in order: correctness, meaningful performance
improvement, stability/reproducibility, end-to-end relevance, hardware appropriateness,
portability where required, and reasonable engineering complexity. Never trade correctness
for speed. Never trade large engineering complexity for negligible, undocumented gains.

## Hardware is a search dimension, not an assumption

The objective is not "write the fastest CUDA kernel." It is: given this workload and this
qualified hardware, discover the fastest practical implementation subject to correctness
and engineering constraints. The target GPU is detected/configured at runtime (see
[`hardware-support.md`](hardware-support.md)); hardware-specific optimizations are
encouraged but must stay behind the backend abstraction and must never leak into unrelated
infrastructure. Single-GPU findings are recorded as per-GPU data, never generalized.

## Anti-patterns

Do not: modify the benchmark to improve reported numbers; optimize a function because it
looks expensive; assume CUDA is the only backend; trust the first or a noisy measurement;
optimize without a baseline; repeat a failed experiment without new evidence; discard
experiment history; converge on one strategy without considering alternatives; declare
success without correctness testing; overfit to one input shape; or allow the paper to
hide negative results.

## Methodology and integrity

The full experimental loop (understand → baseline → profile → literature → hypothesis →
implement → validate correctness → benchmark → compare → record → iterate) is documented
in [`research-process.md`](research-process.md); the measurement contract, including the
steady-state metric definition, is in [`benchmarking.md`](benchmarking.md). Every
experiment — accepted, rejected, or inconclusive — is preserved in the append-only archive
(`docs/experiments.md`), and the research paper may summarize selectively but may never
erase an adverse result.
