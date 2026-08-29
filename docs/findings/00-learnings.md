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

## L10 — Our algorithm + Inductor's fusion beats either alone (2026-08-29)

v9a keeps v8's algorithm (flash attention, L2 chunking, the redundant-mask proof) and
hands the resulting op sequence to `torch.compile(max-autotune)`. Against the COMPILED
baseline:

| | geomean vs compiled | configs lost |
|---|---|---|
| v8 (ours alone) | 1.692x | 2 |
| compiled baseline (Inductor alone) | 1.000x | - |
| **v9a (ours + Inductor)** | **2.678x** | **0** |

Both of v8's losses became wins: config 9 0.94x -> 1.94x, config 12 0.90x -> 1.47x. The
range is 1.33x (config 2) to 11.02x (config 13).

**This is the division of labour L8 predicted, exploited rather than merely observed.** We
choose the decomposition; Inductor fuses it. Neither half gets there alone, and the thing
that unlocked it was reading our own loss pattern rather than profiling harder.

**Consequence for the loop:** every future candidate should be measured BOTH ways --
standalone and compiled. A hand-written change that looks like a loss standalone may be a
win once Inductor fuses around it, and v7 (rejected on precision) is worth re-testing
under compilation for exactly that reason.
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

## L12 — Three bugs found by building a second view of the same data (2026-08-29)

A live dashboard was built over `bench/results.jsonl`. Reading the same rows through a
different lens immediately exposed three defects the ledger's own CLI had been hiding:

1. **`scoreboard()` counted ROWS, not distinct configs.** A v4 parameter sweep reported
   "56 configs measured" on a **14-config matrix** — a number that cannot exist. It
   actually touched 4 distinct configs. Fixed, with a regression test, and a `rows` /
   `is_sweep` field added so a sweep is visibly a sweep.

2. **27 ledger rows are stamped ~5.5 hours in the FUTURE.** My ingest script hardcoded
   `ts="2026-08-29T17:30:00+00:00"` instead of the real measurement time, so "time since
   newest row" read `0s` forever — **a stale ledger that looks permanently fresh**. The
   rows are append-only and stay; consumers must compute age from the newest *non-future*
   row. Never hand-write a timestamp again: pass the real one or let the ledger stamp it.

3. **`git add -A` in the loop swept a subagent's in-progress files into a commit.** The
   loop must commit named paths, not the whole tree, or it silently captures concurrent
   work that was never reviewed.

**The general lesson, and the reason this is a learning rather than a changelog:** every
one of these survived because the ledger was only ever read through the one CLI that wrote
it. A second independent consumer of the same data found three defects in an afternoon.
Build the second reader earlier.

## L13 — We are fp32-only, and the audit rule is now 3 for 3 (2026-08-29)

At `--dtype float16`: v8 **crashes** (`expected scalar type Float but found Half`), v9a
**fails correctness** at max_abs 8.6e-3 against a 2.0e-3 budget. Cause is structural —
every candidate hardcodes `.float()` on the residual, which finding 08 proved is
load-bearing, but which assumed the model itself is fp32.

Probably not the graded path (the benchmark defaults to float32, and the 0.002 absolute
tolerance is nearly unsatisfiable against an fp16 reference — one fp16 ulp at |x|=1 is
half the budget). **So: declare the precondition, do not chase it.** Crashing silently is
worse than declaring `dtype == float32` and saying why.

**The audit rule (L8) is now 3 for 3**: padding (finding 11), baseline (finding 12), dtype
(finding 13) — three findings from questioning inherited defaults, zero from further
profiling. Remaining unaudited: `input_scale=1.0`, and `--benchmark-rounds`/`--repeats`,
which we replaced with our own timing loop rather than using the harness's.

## L14 — A regex merge resolution produced valid, broken code (2026-08-29)

Resolving the g9 conflict by concatenating both sides dropped
`return build(baseline_cls)` from `_v9a`. The function stayed **syntactically valid**,
imported cleanly, and returned `None` — surfacing three commits later as an opaque
`TypeError` inside a subprocess.

Never auto-resolve a conflict in executable code by concatenation. A guard test now
asserts every registered candidate actually builds and names a parent that exists —
which is the kind of invariant that should have existed before the first fork, not after.

## L15 — L11 was right for the wrong reason: max-autotune was DISABLED (2026-08-29)

Every compiled run prints `Not enough SMs to use max_autotune_gemm mode`. **Inductor
disables GEMM autotuning on this 66-SM card.** So v9a and v9b landing 0.3% apart did not
show "autotuning buys nothing" — it showed the two modes had silently collapsed into
nearly the same thing.

The action survives (use `reduce-overhead`, cheaper for identical results) but the
generalization does not: on a datacenter GPU with enough SMs, max-autotune would actually
autotune and that sibling comparison must be re-run. **A null explained by "it does
nothing" transfers; a null explained by "it was disabled" does not.**

Correct the record rather than quietly keeping the convenient conclusion.

## L16 — Validate a replaced protocol against the original (2026-08-29)

Our timing loop deliberately diverges from the benchmark's: we isolate each arm
(finding 05), it keeps both resident. The graded number comes from theirs and the two had
never been compared.

They agree — 0.5% to 6.7% across configs 1, 6, 12, 13, with config 6's baseline at
448.9 ms (theirs) vs 448.4 ms (ours). The spill that motivated our divergence does not
reproduce under v9a, whose peak memory is far lower than the candidate that triggered it.

**Rule: when you replace a harness's protocol with your own, you owe a comparison against
the original.** Ours diverged for a defensible reason and happened to agree; that was not
knowable without running it. `bench/end_to_end.py` now does this on demand.

## L17 — Loops add; only ablation subtracts (2026-08-29)

Ablating v9a's nine inherited generations under compilation:

| removed | worst-case cost of removal | verdict |
|---|---|---|
| L2 batch chunking (v3) | +5.8%, and **−0.3% on config 6** | **subsumed by the compiler** |
| fused Q\|K\|V (v1) | +20% on the launch-bound configs | still pays |
| fp16 weight cache (v1/v6) | **+395% on config 13** | essential |

**Chunking is dead weight.** It took config 6 from 3.21x to 5.72x when v3 added it — real
at the time — but Inductor now manages the working set and the win is gone, including on
the exact config it was built for.

The structural lesson: **an evolutionary loop only ever adds.** Nine generations stacked
nine justifications, each valid when written, and nothing in the loop's design would ever
revisit one after the world changed underneath it. Ablation is the only mechanism that
subtracts, and it found inert complexity in one pass.

**Make ablation a scheduled step, not an afterthought.** Every ~3 generations, re-test the
inherited stack against the current frontier. Otherwise the submission ships its most
sophisticated-looking component doing nothing, and the report credits it.

## L18 — The lean frontier matches the fat one (2026-08-29)

v11 (chunking removed) measures **10.073x vs eager / 2.514x vs the compiled baseline**,
zero losses, against v9a's 10.630x / 2.678x. Inside the noise floor on a matrix-wide
basis, so the g10 ablation's verdict holds across all 13 configs rather than only the 4 it
was measured on.

The frontier is now **five components instead of six** at no measurable cost: fused
Q|K|V, fp16-with-fp32-accumulate, flash via SDPA, the right-padding proof, and
compilation. Each has a measurement behind it; none is inherited on faith.

**First generation that improved by removing something** — and it only happened because
the ablation was run on principle rather than on suspicion. Nothing in the loop's own
signal would have flagged chunking: it was passing, correct, and sitting inside the best
candidate.

## L19 — The dirty-tree rule caught me again, and it was right (2026-08-29)

The first v11 run produced 13 rows and **all 13 were discarded**: the dashboard's files
were sitting uncommitted, so `is_dirty()` marked every row and `clean_rows()` filtered
them out. I only noticed because a downstream ZeroDivisionError had no rows to divide.

The rule is correct and stays. What is missing is a **pre-flight**: `run_matrix.py` warns
about a dirty tree but proceeds anyway, so a 10-minute run can be spent producing rows
that cannot count. It should refuse by default and require an explicit override.

Second time this has bitten (the first was the v2 run in the same session). A warning that
is routinely ignored is not a guardrail.
