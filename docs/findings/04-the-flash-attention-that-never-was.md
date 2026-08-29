# Finding 04 — The FlashAttention that never ran

Recorded 2026-08-29. Measured on RTX 4070 Ti SUPER (sm_89), torch 2.8.0+cu128.
Rows: `bench/results.jsonl`. Code: `bench/candidates/v1_fused_graph.py` (the defect),
`bench/candidates/v2_fp16_flash.py` (the fix).

## The finding

Candidate v1 called `F.scaled_dot_product_attention` and we assumed it was getting
FlashAttention. **It never did — on any of the 14 configs.** The backend actually
selected, confirmed in the profiler on every row, was the fp32 memory-efficient CUTLASS
path `fmha_cutlassF_f32_aligned_64x64_rf_sm80`.

Two independent disqualifications, either of which alone is sufficient:

| what v1 did | what the backend said |
|---|---|
| cast q/k/v back to fp32 before the call | `Expected query, key and value to all be of dtype: {Half, BFloat16}` |
| forwarded the padding mask even when all-True | `Flash Attention does not support non-null attn_mask` |

## Why it was invisible

`scaled_dot_product_attention` selects the best backend that **accepts the arguments it
is given**. A kernel that merely fails to qualify is skipped silently — there is no
warning, no fallback notice, and the call returns correct results at a fraction of the
available speed. The code looked like it was using flash, passed every correctness check,
and posted a respectable 3.11× geomean.

This is the general lesson, and it is worth carrying into every future candidate:
**a dispatching API turns a performance bug into a silent one.** The only reliable check
is to ask which kernel actually ran — probe the backends explicitly with
`torch.nn.attention.sdpa_kernel`, or read the kernel names out of the profiler. Assuming
from the call site is not evidence.

## What the fix is worth

Isolated attention call, fp32-with-mask (what v1 did) versus fp16-no-mask (v2):

| config | v1 path | v2 path | ratio |
|---|---|---|---|
| 13 (S=1024) | 2682 µs | 280 µs | **9.59×** |
| 6 (B=10000) | 9986 µs | 2148 µs | 4.65× |
| 11 (head_dim=8) | 257 µs | 41 µs | 6.27× |

End-to-end across the matrix: **1.12×–2.84× over v1 on every row**, taking the aggregate
from **3.11× → 5.64× geomean**, with 0 failed elements everywhere and `max_abs` in
1.2e-3–1.9e-3 against the 2.0e-3 budget.

And it is the **only reason config 14 is runnable at all**: flash streams the KV axis and
never materializes the score matrix, which at S=100,000 would be 18.63 TB for a single
layer.

## The mask elision, and why it is exact rather than a shortcut

The benchmark's `valid_token_mask` is all-True whenever `padding_ratio` is 0. An all-True
mask is semantically identical to no mask, so dropping it changes nothing about the
computed function — it only changes which backends qualify.

Two details that keep this honest:

- `.all()` forces a host sync, so it is evaluated **once at priming and cached**.
  Evaluating it per call would cost more than flash wins.
- When the mask is genuinely not all-True, v2 **falls back to the fp32 masked path**.
  Handling only the all-True case would be tuning to the default CLI flags rather than
  implementing the operation — the exact move the project's own rules call fraud.

## Backend support, measured per config

Probed explicitly across all 14 configs and four argument shapes:

- **mem-efficient accepted every head_dim in the matrix** (8, 32, 64, 128, 256).
- **cuDNN rejects head_dim = 256** (config 8) even at fp16; **flash accepts it**.
- **math actively raises** under `is_causal=True` with an explicit mask
  (`Explicit attn_mask should not be set when is_causal=True`) — so if mem-efficient had
  ever refused a shape, v1 would have crashed rather than degraded. It never did.

The head_dim = 8 configs (7, 11) that we expected to need a hand-written Triton kernel
are **accepted by both mem-efficient and flash**, so that branch is not needed on this
hardware. Worth re-checking on any card with different backend support.
