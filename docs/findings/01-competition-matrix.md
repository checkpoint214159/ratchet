# Finding 01 — The announced shape matrix, and what it implies

Recorded 2026-08-29. Source: the competition problem statement (§3.2), supplied by the
repository owner. Machine-readable form: `bench/matrix.py` (single source of truth).

## The matrix

All 14 configs are **causal**, and **`ffn_dim == d_model`** on every row.

| # | B | d_model | heads | head_dim | seq | layers | ffn | regime | GFLOP | scores@fp32 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 128 | 4 | 32 | 128 | 4 | 128 | mainstream | 7.5 | 0.02 GB |
| 2 | 1 | 128 | 4 | 32 | 128 | 4 | 128 | launch-bound | 0.1 | — |
| 3 | 4 | 128 | 4 | 32 | 128 | 4 | 128 | launch-bound | 0.5 | — |
| 4 | 16 | 128 | 4 | 32 | 128 | 4 | 128 | launch-bound | 1.9 | — |
| 5 | 128 | 128 | 4 | 32 | 128 | 4 | 128 | mainstream | 15.0 | 0.03 GB |
| 6 | 10000 | 128 | 4 | 32 | 128 | 4 | 128 | throughput | 1174 | 2.62 GB |
| 7 | 64 | 32 | 4 | **8** | 128 | 4 | 32 | awkward head_dim | 0.7 | 0.02 GB |
| 8 | 64 | 1024 | 4 | **256** | 128 | 4 | 1024 | wide model | 421 | 0.02 GB |
| 9 | 64 | 128 | 1 | 128 | 128 | 4 | 128 | mainstream | 7.5 | — |
| 10 | 64 | 128 | 2 | 64 | 128 | 4 | 128 | mainstream | 7.5 | 0.01 GB |
| 11 | 64 | 128 | 16 | **8** | 128 | 4 | 128 | awkward head_dim | 7.5 | 0.07 GB |
| 12 | 64 | 128 | 4 | 32 | 32 | 4 | 128 | launch-bound | 1.7 | — |
| 13 | 64 | 128 | 4 | 32 | 1024 | 4 | 128 | long context | 120 | 1.07 GB |
| 14 | 32 | 1024 | 16 | 64 | **100000** | 2 | 1024 | extreme | **1,391,251** | **20,480 GB** |

## It is an ablation grid, not a sample

Rows 1–6 sweep batch size (1 → 10,000) with everything else fixed. Rows 7–8 sweep model
width. Rows 9–11 sweep head count at fixed width. Rows 12–13 sweep sequence length. Row
14 is an outlier by three orders of magnitude.

Each sweep isolates one axis of the dispatch decision. The organizers are testing
**whether the submission dispatches per regime** — the statement says so directly
("participants can choose different implementations for different shapes by adding shape
checks"). One kernel cannot win this matrix.

## Consequences that change the optimization calculus

**Everything is causal.** Roughly half the attention score matrix is structurally zero.
Skipping those blocks in a fused kernel is **exact**, not an approximation — masked
entries carry exactly zero softmax weight. Free work avoided on all 14 configs.

**`ffn_dim == d_model`, not the conventional 4× expansion.** The feed-forward stage is
therefore about 4× less dominant than profiling the reference benchmark's own defaults
suggests. Attention matters correspondingly more.

**head_dim spans 8 → 256.** Configs 7 and 11 land on head_dim = 8. cuDNN and
FlashAttention typically support {32, 64, 128, 256}; head_dim = 8 may silently fall back
to a slow materialized path. That is a dispatch branch — and plausibly the one place a
hand-written Triton kernel genuinely earns its keep rather than losing to the vendor.

**Config 14 is a feasibility question, not a speed question.** 1.39 PFLOP of causal
attention arithmetic — about 4.4 hours at this card's measured 88 TFLOP/s fp16 ceiling,
*assuming perfect efficiency*. Its materialized score matrix would be 20 TB, so the
reference implementation cannot run it at any batch size. Even the activations
(32 × 100,000 × 1024) are 13.1 GB at fp32 against 16 GB of VRAM. Expect the baseline to
OOM. If a chunked flash path makes it *run at all*, that is a capability result worth
more than any speedup, and should be reported as such.

## Open question — confirm before committing to per-config work

The column is headed **"QKV Dim"**. We read it as `d_model`, so `head_dim = d_model /
heads`. The alternative reading is that it names `head_dim` directly, which would make
`d_model = heads × QKV_Dim` and grow rows 9–11 by up to 16×.

We take the `d_model` reading because the `FFN Dim` column carries identical values, and
an FFN hidden size is conventionally expressed in model dims rather than per-head dims.
**This is worth confirming with the organizers** — it materially changes configs 7, 9, 10
and 11, and it decides whether the head_dim = 8 branch exists at all.
