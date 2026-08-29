# Finding 02 — What the rules permit, decided against measurement

Recorded 2026-08-29. The tolerance figures here are **measured on this machine**, not
reasoned about; see `docs/findings/03-baseline-measurements.md` for the raw runs.

The binding constraint is the competition's accuracy gate: **relative error < 0.02 and
absolute error < 0.002** against the reference implementation.

## Flash attention — permitted, and effectively mandatory

The problem statement invites it in as many words: *"participants can decide which parts
of the layers should be fused into 1 kernel."* That is a description of FlashAttention.

It is also **mathematically exact**. Online-softmax rescaling reorders the accumulation
without changing the function computed, so it differs from the reference only in
last-decimal floating-point rounding — the same class of difference the tolerance exists
to permit.

Two reinforcing reasons here specifically: every config is causal, so a fused kernel skips
the masked triangle *exactly*; and config 14's score matrix cannot be materialized at any
batch size, so a fused path is the only way it runs at all.

## Sparse attention — permitted in principle, a trap in practice

Sparsity **drops entries that would carry nonzero weight**. That is approximation, not
reordering, and it is measured against a 0.002 absolute budget.

The decisive detail is the test data: **inputs are random**. Softmax over random scores
produces near-uniform attention weights, so there is no sparsity structure to exploit and
every dropped weight contributes error of the same order as the ones retained. Sparse
patterns that work on natural language — where attention genuinely concentrates — will
not transfer to this benchmark.

Keep the distinction sharp: **skipping the causal triangle is exact** (weight is exactly
zero). **Skipping "small" weights is approximate** and will likely fail the gate.

## Quantization — exactly one step is available

Measured on the reference config, GEMMs cast to low precision with fp32 accumulation:

| precision | max_abs error | verdict |
|---|---|---|
| **fp16, fp32 accumulate** | **0.0011** | **passes** (budget 0.002) |
| bf16, fp32 accumulate | 0.0096 | fails by ~5× |
| fp8 / int8 | — | far outside the budget |

The reason is mechanical rather than empirical. bf16 carries 8 mantissa bits, so its
representable spacing near 1.0 is ≈ 0.008 — **over budget from representation alone**,
before any arithmetic error. fp16 carries 10 bits, spacing ≈ 0.001.

**The tolerance therefore mandates fp16 over bf16**, which is the opposite of the usual
training-workload advice. Worth ~2× on this card: measured 88 TFLOP/s for
16-bit-with-fp32-accumulate against 44.5 TFLOP/s for TF32.

Caveat to watch on the wide configs (8, 14): fp16's exponent range is much narrower than
bf16's, so activation overflow is a live risk at `d_model = 1024`. Guard it.

## Hardware-adaptive dispatch — explicitly invited

The statement invites this twice: shape checks are permitted and all shapes are
disclosed; and *"different methods may be used to optimize the codes depending on the
machine (GPU cards) you use."*

This is the strongest differentiator available, and the answer to the objection every
judge will raise — *everyone benchmarked on different hardware, so how do I compare you?*
Predicates must be functions of **measured device properties**, never hardcoded
constants: the same arithmetic intensity is compute-bound on one card and memory-bound on
the next. `ratchet/oracle/device.py` already performs that calibration
(measured: 66 SMs, 99 KB shared memory, 613.7 GB/s, 2.22 µs launch overhead, ridge point
144 FLOP/B).

Ship the tuner alongside the cached table, so the submission re-tunes on hardware it has
never seen rather than carrying our constants onto a grader's GPU.

## Launch overhead — likely the largest single lever on this matrix

The reference model launches ~178 kernels per forward pass; each launch costs ~2.2 µs
measured on this machine. Against config 1's ~7.5 GFLOP (≈ 85 µs at the fp16 ceiling),
roughly 120 launches over 4 layers is ≈ 264 µs of gap — **launch overhead exceeds the
arithmetic**. Config 2 runs the same launch count over 1/64 the work.

Four of fourteen configs (2, 3, 4, 12) total under 2 GFLOP. For those, CUDA graph capture
is worth multiples rather than percentages, and no kernel-level optimization can
substitute for it. Measured on the reference model, graph capture alone — with the model
otherwise untouched — was worth **1.53×**.
