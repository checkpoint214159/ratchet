# Running learnings — the loop's long-term memory

Appended by the autoresearch loop at the end of each iteration. Read at the start of the
next one. Newest last. Each entry is something that should change what the loop does, not
a log of what it did.

---

## L1 — CMP over a LINEAR history is degenerate (2026-08-29)

The first Thompson draw exposed it. Clade "successes" ranked as:

```
32bd7df2  s=176 f=19   <- the OLDEST commit scores highest
6a832aff  s=163 f=19
e15e20e2  s=162 f=19
...
fc2e159f  s=12  f=2    <- the NEWEST scores lowest
```

That ordering is an artifact, not a signal. On a single branch every later commit is a
git descendant of every earlier one, so a node's clade is just "everything committed after
it" and the ranking measures **age, not productivity**. HGM's whole premise — that a node's
subtree tells you its value as an ancestor — requires the subtree to be *chosen*, which
requires branching.

**Consequence for the loop:** until candidates live on their own branches, the Thompson
draw is close to uninformative and must not be presented as if it discovered something.
Branch first; the sampling becomes meaningful once siblings exist.

## L2 — Clade success is counted per ROW, not per candidate (2026-08-29)

v7 was **rejected** (fails config 6 on correctness) yet carries `s=12 f=2`, because 12 of
its config rows passed. Thompson consequently sampled it most often — a rejected candidate
looking like the most promising parent.

This is defensible in HGM terms (productivity, not promotion) but is a live trap: the loop
must not read a high clade score as "this candidate works". Promotion is a separate gate,
and `scoreboard()` is the thing that answers "did it work".

## L3 — The named bottlenecks are exhausted; what is left is architectural (2026-08-29)

| bottleneck | outcome |
|---|---|
| launch overhead, 36% of wall | taken — CUDA graphs, ~1.85x |
| fp32<->fp16 conversions, 12.8-26.8% | partly taken — GELU free (v6), residual immovable (v5) |
| LayerNorm + add, 12-26% | unavailable — costs the precision budget for noise (v7) |

Remaining time is matmul at 89-92% of the measured ceiling and vendor-tuned attention.
Further parametric search re-measures a space already shown flat.

## L4 — The precision budget is nearly spent, and the floor is a REPRESENTATION floor (2026-08-29)

v7 landed seven different configs on exactly `1.9384026527404785e-3` (96.9% of the 2.0e-3
budget). Identical across shapes means it is fp16 rounding hitting a fixed limit, not
error proportional to work. Any future change that adds an fp16 rounding step to the
residual path will hit the same floor and cannot be tuned out of it.

**Corollary:** `max_abs` alone does not predict pass/fail. The gate is OR
(`|d| <= atol` **or** `|d| <= rtol*|ref|`), so config 7 passes at 115.9% of the absolute
budget while config 6 fails at 100.7%. Judge by `failed_elements`, never by `max_abs`.

## L5 — BLIND SPOT: every measurement so far used padding_ratio = 0.0 (2026-08-29)

Every candidate from v2 onward takes a **fast path when the mask is all-True** — eliding
the mask so FlashAttention qualifies — and a much slower fp32 masked path otherwise. The
entire matrix has only ever been measured with `padding_ratio=0.0`, i.e. always on the
fast path.

The benchmark exposes `--padding-ratio`. If the graders run anything above zero, every
number in the ledger is off the path that would actually execute. This is unmeasured, and
it is the largest untested assumption in the project.
