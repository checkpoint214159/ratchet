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

## L6 — L5 was right, and the fix was a proof rather than a tuning (2026-08-29)

The padding blind spot was real and large: v6 retained only 51% of its speedup on configs
1 and 5 at `padding_ratio=0.5`, and **28% on config 13**. v8 recovers it — 6.730x unpadded
(unchanged from v6) and 5.853x at padding 0.5, against v6's ~2.86x.

The fix came from *reading the reference's masking semantics*, not from measurement: with
right-padding and causality the key mask is provably redundant, which is what lets the
fp16 no-mask flash path qualify. Two runs of profiling would never have found it.

**Method note for the next iteration:** the highest-value move this session was auditing an
assumption, not optimizing a hot spot. Before proposing another kernel change, check what
else has only ever been run at a default — `input_scale` is still 1.0 everywhere, `dtype`
is still float32 everywhere, and `--compile-baseline` has never been used, which means
**the baseline we quote speedups against has never been the strongest available baseline**.
That last one would deflate every number in the ledger and is the next thing to test.

## L7 — Branching is now real, but a fork still does not exist (2026-08-29)

v8 lives on `cand/g8/right-pad-redundant-mask`, the first candidate branch. But it branched
from HEAD, so the history is still linear and L1 still applies: no siblings means clade
statistics still rank by age. **A genuine fork — two candidates from the same parent —
is required before the Thompson draw carries information.** Do that next.

## L8 — We committed the exact error our own contract was written to prevent (2026-08-29)

Every number from v1 to v8 was quoted against an **eager** baseline. `CLAUDE.md` rule 5
and `docs/04-failure-modes.md` both open by forbidding this, citing KernelBench collapsing
1.43x -> 0.88x under the same correction. `--compile-baseline` existed the whole time and
was never used.

Corrected: **7.229x -> 1.692x geomean**, and we lose outright on configs 9 (0.94x) and
12 (0.90x).

The shape of what survives is informative: we beat `torch.compile` where an ALGORITHMIC
choice matters (config 13 at 7.89x from streaming attention, config 6 at 3.00x from
L2-sized chunking, config 11 at 3.69x) and lose where the win would be pure kernel fusion,
which Inductor does better than hand-written op sequences.

**Both numbers must appear in any report.** The benchmark defaults compile-baseline to
off, so 7.2x is what a default run prints and is not fabricated — but quoting it alone is
the artifact this project exists to avoid.

**Method rule this establishes:** before optimizing anything further, audit what the
comparison is against. Two of the three largest findings this session (L6/finding 11, and
this one) came from auditing an untested default, not from profiling. Remaining unaudited
defaults: `input_scale=1.0` and `dtype=float32` everywhere.

## L9 — Ad-hoc scripts reintroduce the errors the harness prevents (2026-08-29)

My first compiled-baseline audit compiled all 13 configs in ONE process and produced
~1.00x ratios for later configs, which I nearly reported as "compile does not help here".
`torch._dynamo` caches per process and silently falls back to eager past its recompile
limit; compile times of 0.1-0.3s versus 2-6s were the only tell.

`bench/run_matrix.py` already enforces one-config-per-subprocess. I bypassed it for speed
and got a contaminated result — the same class of error as finding 05. **Measure through
the harness, or accept that the harness's guarantees do not apply.**

## L11 — First real fork, and it answered a cost question decisively (2026-08-29)

v9a and v9b are true siblings: same parent (v8), same hypothesis (hand the decomposition
to Inductor), one variable changed — `max-autotune` vs `reduce-overhead`.

| sibling | mode | geomean vs eager | compile cost |
|---|---|---|---|
| v9a | max-autotune | 10.630x | 2-19s per shape |
| v9b | reduce-overhead | 10.600x | much lower |

**0.3% apart — deep inside the 3% noise floor.** The autotuning buys nothing measurable
on this matrix, so `reduce-overhead` is strictly better for a graded run: identical speed
for a fraction of the compile time, across 13 shapes.

This is what a fork is FOR. A single lineage would have picked one mode and never learned
the other was free. Two siblings differing in one variable answered a cost question in one
run each.

**Note the asymmetry worth carrying:** a null result between siblings is still a
promotion — not of speed, but of the cheaper option. Equivalence is actionable when the
two arms have different costs.
