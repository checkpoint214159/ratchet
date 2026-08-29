# Finding 09 — Config 14 runs, in 3.18 GB, and the claim has a hard limit

Recorded 2026-08-29. Probe: `bench/probe_config14.py`. Candidate: `v6_fp16_gelu`.

## What the reference cannot do

Config 14 (B=32, S=100000, d=1024, H=16, L=2, causal) is the only row of the matrix the
reference implementation cannot execute. It does not fail slowly — it **OOMs inside the
benchmark's own input generator, before `forward()` is ever called**, on a 12.21 GiB fp32
input tensor that `generate_random_case` allocates twice. Its materialized attention would
need **18.63 TB for a single layer** against 16 GB of VRAM.

## What v6 does

| | |
|---|---|
| ran | **yes** |
| per sequence | **0.54 s** |
| peak memory | **3.18 GB** |
| full batch (B=32) | **17.2 s — extrapolated, not measured** |

Correctness at proxy shapes sharing config 14's width and depth (d=1024, H=16, L=2,
causal), where the reference still fits:

| seq_len | passed | max_abs | budget |
|---|---|---|---|
| 1024 | yes | 9.03e-4 | 2.0e-3 |
| 4096 | yes | 9.03e-4 | 2.0e-3 |

Note the margin is comfortable here — 45% of budget — and **flat across a 4x change in
sequence length**, which is the expected signature of flash attention: sequence length
changes how much work is done, not what is computed.

## The limits of this claim, stated plainly

**Correctness is NOT verified at S=100000, and cannot be.** There is no baseline output to
compare against at that shape, by construction. The proxy shapes establish that the
arithmetic is right for this width and depth; they do not establish it at 100,000 tokens.
The ledger row records `correctness.passed = null`, which keeps it out of `passing()`,
`best_known()` and clade statistics. **It runs; it does not count as a passing result.**

**The B=32 figure is 32x a measured single-sequence forward**, not an end-to-end run. It
is labelled `extrapolated_full_batch_s` in the row.

**The input was streamed from pinned host memory**, one sequence at a time. This is the
honest description of the binding constraint: on a 16 GB card the harness's own input
construction, not the model, is what makes this shape unreachable. A grader whose harness
builds the input on-device will OOM before our code is called, regardless of what our code
can do.

## Why it is worth having anyway

If the shape list is graded as announced, "the reference cannot run this and we can, in
3.18 GB" is a stronger result than any speedup in the matrix — and it is a direct
consequence of the finding-04 fix. Flash attention streams the KV axis and never
materializes the score matrix, so the 18.63 TB never exists. The same change that was
worth 2.10x-9.59x on the isolated attention call is what makes this row reachable at all.
