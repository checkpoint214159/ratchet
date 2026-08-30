# 36 — head_dim=8 is a real 1.4x and an end-to-end nothing, and the cost is the layout

**Date:** 2026-08-30. **Branch:** `cand/g22/headdim8-attn`, parent `v18_capture_insurance`.
**Candidate:** `v22_headdim8_attn`. **Kernel:** `bench/kernels/attn_smallhead.py`.
**Status of the numbers below:** op-level, isolated `do_bench`, median of 5, under the GPU
lock. **INDICATIVE (L41)** — a probe may propose, it may never conclude. No sweep row.

## What was built

A Triton causal flash-attention kernel for head dimensions **strictly below the `tl.dot`
contraction floor** that Triton's own NVIDIA backend reports for this device
(`min_dot_size(...)[2]` = 16 at fp16 on sm_89, from `mma.sync.m16n8k16`). The floor is
queried, not written down; the dispatch predicate is `head_dim < floor` plus divisibility
and power-of-two checks, and it selects exactly the below-floor rows of the matrix
(configs 7 and 11) without naming either.

Finding 23 had already killed the original reason to care ("vendor backends may refuse
head_dim=8" — they do not, all four accept it). The surviving reason: PyTorch's bundled
FlashAttention-2 has no head_dim=8 kernel. `HEADDIM_SWITCH` in
`torch/include/ATen/native/transformers/cuda/flash_attn/static_switch.h` rounds anything
`<= 32` up to `kHeadDim = 32`, so the vendor contracts over 32 lanes where 8 carry data.
Triton's floor is 16, so we pad half as far, inside the kernel, where padding is free.

## Measured: the mechanism is real, and worth much less than the source predicts

| shape | SDPA + repack | ours | ratio |
|---|---|---|---|
| cfg 7 `(64, 4, 128, 8)` | 23.4 us | 16.8 us | **1.40x** |
| cfg 11 `(64, 16, 128, 8)` | 59.0 us | 42.2 us | **1.40x** |

The `HEADDIM_SWITCH` argument predicts ~4x and proposals A-01 / B-04 predicted 1.79x /
1.49x end-to-end. The truth is 1.40x at the *op* level, because **these shapes are nowhere
near mma-bound**: at DP=16 the score and context matmuls account for ~7 us of config 11's
42 us. The pad was never most of the cost. A source-code argument correctly identified a
defect and badly mis-sized it, which is [L34] again — the reading was right, the value was
not carried by it.

**The tiling was swept, not guessed.** 18 tiles on both shapes, then the top 8 re-run at
median-of-5. The spread across tiles on the *same kernel and the same mechanism* is
**1.9x** (best 16.8 us, worst 30.8 us on cfg 7). The top three are tied within 2%.

## L42 — A hand-written kernel's LAUNCH WRAPPER can silently cost you the compiler

**This is the largest thing this generation learned, and it was found by the screen, not
by reasoning.** The first version of v22 screened at **-18.9%**, with config 7 at
**2.18x slower** (0.115 -> 0.250 ms). Every correctness test was green. `graph_verified`
was `True`. `capture_source` reported a successful capture. The profiler, on config 7:

| | v18 | v22 (first version) |
|---|---|---|
| attention | `flash_fwd_kernel` 45.2 us | `_attn_fwd_smallhead` **27.8 us** |
| LayerNorm | `triton_per_fused_*` ~13 us | ATen `vectorized_layer_norm_kernel` x9, **151.1 us** |
| total device time / call | **99.7 us** | **242.8 us** |

**Attention was 1.63x faster. What was lost was Inductor.** The launch wrapper resolved
its tile plan at the call site, so `min_dot_k()` — an import inside a `try/except`, a
locally defined class, and `torch.cuda.get_device_capability()` — executed inside Dynamo's
traced region. Dynamo could not trace it, dropped the whole frame to eager, and our CUDA
graph then faithfully captured an *eager* op sequence. Every guard this project has built
reported success, because every one of them was watching correctness or capture, and the
thing that broke was fusion.

The fix is a split: `plan_for()` queries the device and the compiler and is called once in
`_prime`; `smallhead_attention()` is traced and sees only plain ints and a `torch.empty`.
Same profile after the fix: **27.4 us attention, 80.6 us total — 1.65x on attention,
1.24x on config 7.**

**The rule: a hand-written kernel dropped into a compiled region must be audited for what
its LAUNCH WRAPPER does, not only for what the kernel does.** `bench/kernels/ffn_fused.py`
was safe by accident — its wrapper is pure arithmetic — so nothing in the repo had ever
exercised this. Pinned by
`tests/bench/test_v22_headdim8_attn.py::test_inductor_still_fuses_around_the_hand_written_kernel`,
which asserts a `triton_*` LayerNorm kernel is present and an ATen one is not.

Two corollaries:

* **The mechanism assertion has to point at the right mechanism.** [L36] said assert that
  the subject was actually built. Here the subject (our kernel) *was* built and ran, faster
  than the vendor's; the thing that silently did not run was the *compiler around it*. A
  mechanism check that only watches your own component cannot see this.
* **Do not instrument a compiled region to test it.** The first two tests counted calls by
  monkeypatching the wrapper. A patched wrapper is a new closure Dynamo guards on, which
  forced a recompile every call, blew the cache limit, and produced a spurious failure.
  Ask the device what ran (profiler kernel names) instead of wrapping the call site.

## Correction 1 — the output repack costs nothing, and I was sure it did

The lineage writes `ctx.transpose(1, 2).reshape(B, S, D)` after every SDPA call. A
transposed 4-D view reshaped to 3-D must materialise a copy, so this looked like a free
extra win worth a full read and write of the context tensor per layer. Measured:

    cfg 7    SDPA alone 24.3 us    SDPA + repack 23.0 us
    cfg 11   SDPA alone 58.5 us    SDPA + repack 57.4 us

Zero, twice, and slightly negative — i.e. noise around zero. **PyTorch's flash backend
works internally in `[B, S, H, hd]` and returns a transposed view of that buffer**, so
`.transpose(1, 2)` restores contiguity and the `reshape` is a no-op. Our kernel's
token-major epilogue is kept because it is free for us too, but it must not be credited
with any part of the 1.40x.

## Correction 2 — the remaining headroom is the LAYOUT, not the mma pad

Same kernel, same tile, on **contiguous** `[Z, H, S, hd]` inputs instead of the strided
views this lineage actually produces:

| | strided (what we get) | contiguous |
|---|---|---|
| cfg 7 | 16.8 us = 1.40x | 13.5–14.0 us = **1.67–1.73x** |
| cfg 11 | 42.2 us = 1.40x | 32.1–32.5 us = **1.82–1.84x** |

q/k/v here are views of one `[Z, S, 3D]` GEMM output, so the head-dim axis is 16
contiguous bytes and then jumps `3*D*2` bytes. **At head_dim=8 every load is 16 useful
bytes inside a 32-byte sector: half of attention's DRAM traffic is thrown away by the
layout.** That is a larger effect than the mma pad it was proposed to fix.

It is **not** collectable by `.contiguous()` — repacking three 2.1 MB tensors costs ~20 us
to save ~10 us, which is v20's own finding restated. It is collectable by a QKV projection
that owns its epilogue and scatters head-major, which is exactly
`bench/kernels/qkv_headmajor.py` (v20) — whose `worth_it()` currently returns False for
`head_dim < 16`. **v20 x v22 is a measured, specific recombination lead.**

## The honest end-to-end arithmetic, stated before any sweep ([L33])

In the model the kernel does *better* than the isolated probe on attention — **1.65x**,
because in-graph the vendor's flash kernel is the same 45.2 us it always was while ours
benefits from the warm L2 — and attention is ~45% of config 7's device time, so:

| | | |
|---|---|---|
| config 7 (profiled) | 99.7 -> 80.6 us device time | **~1.24x** |
| config 11 (projected from the same share) | | **~1.25x** |
| 13-config geomean | | **~ +3%** — inside the ±7% noise floor ([L29]) |
| `matrix.weighted_score` (cap 3.0) | | **~ +1%** — config 11 already sits at 6.24x, above the cap, so its entire gain scores **zero** |

**Stage-1 screen, after the fix** (configs 2, 7, 8, 10; one pass; advisory, never a ledger
row). This is a harness measurement rather than a probe, and it agrees with the profile:

| cfg | v18 | v22 | | |
|---|---|---|---|---|
| 2 | 0.0707 ms | 0.0707 ms | +0.1% | head_dim 32 — declines, untouched |
| **7** | **0.1147 ms** | **0.0952 ms** | **-17.0% (1.204x)** | the only config the kernel fires on |
| 8 | 6.5495 ms | 6.5475 ms | -0.0% | head_dim 256 — declines, untouched |
| 10 | 0.2417 ms | 0.2570 ms | +6.4% | head_dim 64 — declines; inside the noise floor |

Screen geomean **2.364x vs the parent's 2.292x, +3.2%** — verdict PROMOTE. Two screens were
spent, and the first one earned its cost: it is what caught the graph break above.

Configs 7 and 11 are also two of the *cheapest* rows in the matrix (~4–6 s of a 112 s
sweep). **A perfect result here is not resolvable by one sweep**, and under the project's
own objective it is nearly invisible. This is a report artefact of the same kind as v17
(finding 25) — "we beat the vendor kernel in the one regime where the hardware says we
should" — not a frontier move, and it should be judged as one.

## L43 — A live silent-wrong-answer bug in v8..v18: `causal` is ignored

Found while writing the fallback tests, not by looking for it.

`v8_padfast._core` and every descendant call
`F.scaled_dot_product_attention(q, k, v, is_causal=True)` **unconditionally**. Nothing
downstream of v8 reads `self.config.causal` in the attention call — v8 only consults it
when deciding whether the right-padding proof applies. So a non-causal model gets causal
attention. Measured on a `(4, 128, 128, heads=16, L=1, causal=False)` shape against the
fp32 baseline, identically for v8 / v13 / v18 / v22:

    max_abs 9.866e-01     39345 failed elements against a 2e-3 budget

Every announced row is causal (`matrix.py`), so **no ledger number is affected**. But the
reference benchmark's own default is `causal=False`, and `--causal` is a CLI flag a grader
chooses. This is [L24] in its purest form: *correct only because of how the harness happens
to call it*. It is also the fifth member of the [L36]/[L38]/[L40] family — an assurance
nobody arranged to be capable of failing, found by someone looking rather than by the
system noticing.

**Not fixed here, deliberately.** Fixing it inside v22 would put two variables in one
candidate. `tests/bench/test_v22_headdim8_attn.py` pins it instead: v22 must produce
*exactly* the parent's failed-element count on a non-causal shape, and the test fails
loudly the moment someone fixes it upstream. The fix is a one-line, one-variable candidate
of its own and should be queued as such.

## What would falsify the value claim rather than the mechanism

The mechanism is measured. What is unmeasured is whether 1.40x on the op survives inside
the compiled + graph-captured core, where Inductor's scheduling and the CUDA graph change
what the neighbours cost. Graph capture is confirmed to still succeed with the kernel in
the region (`graph_verified=True`, `capture_source=caller`, asserted in the tests), which
is the failure this most plausibly had — but capture succeeding is not the same as the
gain surviving, and [L33] says an isolated measurement measures the isolation. Only the
harness answers it, on configs 7 and 11 only, and the answer will be ~1.1x on two of the
matrix's smallest rows.
