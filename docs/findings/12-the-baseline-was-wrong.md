# Finding 12 — Our baseline was eager. The honest speedup is 1.69x, not 7.2x.

Recorded 2026-08-29. Audit: one config per process (see the method note — the first
attempt at this was itself contaminated).

## What we were doing wrong

`CLAUDE.md` rule 5, written into this project before any code existed, says:

> **The baseline is `torch.compile(mode="max-autotune")` with TF32 enabled** ... Never
> eager FP32. Roughly half of all published kernel speedups are an artifact of getting
> this wrong.

`docs/04-failure-modes.md` opens with the same warning, citing KernelBench's collapse from
a reported 1.43x to 0.88x once its baseline was corrected.

**We did exactly the thing both documents warn about.** Every measurement from v1 to v8
compared against `BaselineTransformer` in eager mode. `--compile-baseline` exists in the
custody benchmark and had never been used.

## The correction

Compiled baseline vs eager, measured one config per process:

| cfg | eager | compiled | baseline speedup |
|---|---|---|---|
| 2 | 1.954 | 0.135 | **14.45x** |
| 4 | 2.994 | 0.219 | 13.66x |
| 12 | 1.728 | 0.206 | 8.40x |
| 3 | 1.788 | 0.247 | 7.25x |
| 7 | 1.735 | 0.335 | 5.18x |
| 10 | 2.310 | 0.542 | 4.27x |
| 9 | 1.526 | 0.458 | 3.34x |
| 11 | 7.170 | 2.263 | 3.17x |
| 13 | 107.46 | 35.48 | 3.03x |
| 1 | 1.720 | 0.586 | 2.94x |
| 5 | 3.377 | 1.241 | 2.72x |
| 6 | 442.2 | 216.5 | 2.04x |
| 8 | 16.44 | 14.42 | 1.14x |

`torch.compile` beats eager on **every single config**, by up to 14x.

## What our candidate is actually worth

| | geomean |
|---|---|
| v8 vs **eager** baseline (what the ledger reports) | **7.229x** |
| v8 vs **compiled** baseline (the honest number) | **1.692x** |

And we **lose outright on two configs**: config 9 (0.94x) and config 12 (0.90x). One line
of stock PyTorch beats our hand-built candidate there.

Where we genuinely win, we win for a reason that survives scrutiny:

| cfg | vs compiled | why |
|---|---|---|
| 13 | **7.89x** | long context — flash attention beats what Inductor generates |
| 11 | 3.69x | head_dim=8, many heads |
| 6 | 3.00x | L2-sized chunking Inductor does not do |
| 8 | 2.06x | wide model |

That shape is coherent: we beat the compiler where an *algorithmic* choice matters
(streaming attention, cache-aware chunking) and lose where it is pure kernel fusion, which
Inductor does better than hand-written op sequences.

## Both numbers are real, and the report must carry both

The custody benchmark defaults `--compile-baseline` to **off**. So if the graders run
default flags, 7.2x is literally what their harness prints. That number is not fabricated.

But quoting it alone would be precisely the artifact this project was built to avoid. The
defensible framing is both, with the method attached: *"7.2x against the benchmark's
default eager baseline; 1.69x against `torch.compile(max-autotune)`, which is the stronger
baseline and the one we consider honest — and we lose to it on 2 of 13 configs."*

A tech report that omits the second number is the paper this project's own dossier files
under reward hacking.

## Method note: the first version of this audit was wrong too

The initial run compiled all 13 configs **in one process** and produced ratios near 1.00x
for the later ones — which I nearly reported as "compile does not help there". It was an
artifact: `torch._dynamo` caches per process and **silently falls back to eager** once its
recompile limit is hit, so "compiled" was eager for those configs. Compile times of 0.1-0.3s
were the tell, against 2-6s for genuine compilation.

The fix was the isolation `bench/run_matrix.py` already enforces and this ad-hoc script
bypassed. Same lesson as finding 05: the harness's discipline exists for a reason, and
stepping outside it reintroduces exactly the class of error it was built to prevent.
