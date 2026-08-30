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

## L20 — Dispatch cost is per-call and fixed; regimes derived from shape mispredict it (2026-08-29)

Config 2 was CPU-dispatch-bound, not GPU-bound: 232 us CPU vs 126 us GPU, with
**TorchDynamo cache lookup at 22.5 us every call** against ~1.2 us of arithmetic. Moving
graph ownership from Inductor to us (compile in default mode, capture the compiled
callable ourselves) reclaims it: **2.514x -> 2.712x vs the compiled baseline, +7.9%,
zero losses.**

The gain is entirely in configs 3 (+87.3%) and 2 (+40.6%). Everything else is inside the
noise floor — the exact signature of a fixed per-call cost.

**The durable lesson is the mispredict.** Our `launch_bound` regime groups configs 2, 3, 4
and 12. v12 transformed 2 and 3 and did nothing for 4 and 12. The label groups by shape;
the real predicate is **call duration relative to a fixed dispatch cost**, and 4 and 12 do
enough work that 22 us stops mattering.

Regimes derived from shape parameters are a reporting convenience. Where they disagree
with measured time, the measurement wins — and the labels should carry the caveat rather
than be quietly redrawn to fit the result.

## L21 — "Two mechanisms must not nest" was right but under-specified (2026-08-29)

v9a removed our CUDA graph to avoid nesting with Inductor's. Correct — but it silently
chose *which* mechanism to keep, and the one it kept sits behind a per-call guard check.
Inverting ownership (keep ours, drop Inductor's) obeys the same non-nesting rule and is
7.9% faster.

**When a constraint admits two solutions, measure both.** v9a treated "do not nest" as
determining the answer when it only narrowed it to two.

## L22 — Our own objective function had saturated, and inverted the ranking (2026-08-29)

The scoreboard ranked **v9b / v11 / v9a above v12**, when v12 is measurably the best
candidate (2.712x vs compiled against v11's 2.514x). Not a tie -- an inversion, in the
one table whose entire job is mapping score to commit.

Two causes, both self-inflicted:

1. **The 3.0 clip was chosen when speedups were 1-2x.** By generation 12, **17 of 18**
   config speedups exceeded it, so every candidate from v3 onward scored 2.79-2.82 and
   the number carried no information at all.
2. **It scored against EAGER**, which finding 12 had already established is the wrong
   reference — and nothing propagated that correction into the objective.

Fixed: score against the compiled baseline, read from `baseline_compiled` ledger rows
rather than a constant. The cap now bites on 4 of 13 configs instead of 17 of 18, and the
board orders correctly. The saturated eager number is kept in a labelled column rather
than dropped, so the failure stays visible.

**Two general lessons:**

*A clipped objective silently stops discriminating once the population outgrows the clip.*
Nothing errors; the number just becomes a constant. Any capped metric needs a periodic
check that its inputs still straddle the cap.

*A correction is not applied until it reaches every consumer.* Finding 12 established the
right baseline four generations ago. The scoreboard kept using the wrong one because
nobody asked what else read that number. **When a finding invalidates an input, grep for
every consumer of it.**

## L23 — A silent-wrong-answer mode our whole correctness suite was blind to (2026-08-29)

v12 can capture an **empty** CUDA graph (Dynamo re-traces inside the capture region and
touches the RNG state). `replay()` then does nothing and the candidate returns the stale
static buffer — right shape, right dtype, wrong values.

v13 verifies the graph against a freshly computed reference and falls back to the compiled
callable otherwise. **Cost: none.** 2.711x vs v12's 2.712x.

**The test-design lesson is the transferable part.** Every accuracy check we had ran one
input per trial against a reference computed for that same input. That is structurally
blind to staleness: a stale buffer holding a correct *previous* answer still matches the
reference for the *previous* input. The suite could not have caught this no matter how
many trials it ran.

The test that does catch it is the crudest one imaginable: **two different inputs must not
produce identical output.** No tolerances, no reference. Add an invariance check like this
wherever a candidate holds mutable state across calls.

## L24 — "Correct because of how the harness calls it" is not correct (2026-08-29)

v12's published numbers are genuinely sound, because the harness runs five accuracy trials
on fresh inputs before timing and a stale buffer cannot survive that. But the property was
accidental — it belonged to the caller's ordering, not to the candidate.

A grader whose harness warms up differently would have gotten silent garbage from our best
submission. **Ask of every candidate: is it correct on its own terms, or only under the
call pattern we happen to test it with?**

## L25 — Invariance and equivalence tests catch disjoint bug classes (2026-08-29)

Sweeping all 15 candidates against three crude invariants found **3 bugs, in candidates
that had all passed the full accuracy suite**:

- v12: different inputs → identical output (the empty-graph staleness of finding 17)
- v10b, v10c: the returned tensor is mutated by the next call

The second is general and worth carrying: **`torch.compile(mode="reduce-overhead")`
returns a tensor backed by a static buffer the next call overwrites.** Any candidate using
it must `.clone()` before returning. It never corrupted a measurement only because the
harness compares before calling again — L24 again, correctness borrowed from the caller.

**The methodological point.** Our correctness machinery is elaborate: locked tolerances,
nine input distributions, an FP64 reference, hand-seeded known-bad fixtures. It could not
see any of these, because every one of those checks assumes the output corresponds to the
input just given. A stale buffer holding a correct previous answer satisfies all of them.

The tests that caught it assert things that sound too obvious to write down — *a function
of its input should depend on its input*. **Add invariance checks wherever a component
holds state across calls; equivalence checks alone leave a blind spot the size of every
stateful part of the system.**

## L26 — The tolerance margin is thinner than it looks (2026-08-29)

At `input_scale=0.01` **every candidate fails**, including the pure-fp32 one (2.38e-3 vs a
2.0e-3 budget). Isolating: substituting only `F.scaled_dot_product_attention` into an
untouched fp32 baseline is enough to fail, with `max_abs` growing 9.19e-4 -> 2.29e-3.

Crucially the **output magnitude is unchanged** (mean 0.798 either way) because LayerNorm
normalizes the input scale away. So it is not a small-signal artifact — it is flash
attention's online softmax accumulating differently, amplified ~2.5x by LayerNorm's `eps`
becoming ~10% of the input variance at that scale.

**The real lesson is about margin, not this config.** At the default scale our worst
config uses **94% of the tolerance budget**. A routine, benchmark-exposed change in the
input distribution multiplies the error by 2.5. We are not passing comfortably; we are
passing narrowly and got no warning about it, because `max_abs` looked fine at the one
scale we ever tried.

**Track margin as a first-class metric, not just pass/fail.** A candidate at 94% of budget
and one at 30% are not equally correct, and only the second survives a distribution shift.

## L27 — The audit rule finished 4 for 4 (2026-08-29)

padding, baseline, dtype, input_scale. **Every inherited default that was never varied
hid something**, and two of the four (the eager baseline, the padded path) were large
enough to change the headline number. Zero comparable findings came from further
profiling in the same period.

The defaults are now exhausted. The rule that replaces it for the next phase: **when a
knob exists and you have never moved it, you do not know what it does.** Applies equally
to `--benchmark-rounds`, `--repeats`, and `--compile-mode`, none of which we have varied
either — though L16 already showed our protocol agrees with the harness's on the first two.

## L28 — Dispatch built; it changes nothing measurable, and that is the correct outcome (2026-08-29)

v14 adds shape-aware dispatch with predicates derived from measured free device memory.
On the 13 runnable configs it measures **identically to v13** (2.70x vs 2.71x, inside
noise) because it always chooses the resident path there.

**That was predicted in the docstring before the run, and it is the right result.** A
dispatcher whose branches never fire on the measured set has not failed — it has correctly
declined to change anything. Its value is entirely in the branch it cannot yet exercise:
config 14, whose 12.21 GiB input the harness itself cannot build.

Two things make it worth having anyway:

1. **It is the architecture the problem statement asks for** — shape checks are explicitly
   permitted and per-GPU methods explicitly anticipated. Every prior candidate applied one
   implementation uniformly.
2. **The predicate is stated in terms another GPU can evaluate.** Tests assert the source
   contains no config ids and no announced shape constants, and that halving device memory
   flips config 6 from resident to streamed. A dispatch that does not respond to the
   device is a hardcoded table wearing a costume.

**The lesson for the loop:** a candidate that measures flat is not automatically a failure.
Ask what it was built to change, and whether the measured set can even exercise it. Judging
v14 by its geomean would discard the one component that handles the shape nothing else can.

## L29 — CORRECTION to L28, and the noise floor is wider than 3% (2026-08-29)

L28 claimed v14 measured "identically to v13 (2.70x vs 2.71x)". **That number was wrong —
written before the run finished.** The measured aggregate was **2.605x vs 2.711x, -3.9%**,
which is outside the 3% floor I had been treating as decisive.

Re-measuring five configs resolved it: per-config deltas were **+6.8%, 0.0%, -0.4%,
-5.6%, 0.0%** — mixed, averaging near zero. So v14 really is equivalent to v13, and the
-3.9% aggregate was variance.

**But that means the 3% noise floor is too tight.** It came from L11, estimated from
accidental replicates in the search loop over four configs. Observed here on short configs:
**±7% run to run on identical code.** Every "above the noise floor" judgement this session
that rested on a 3-5% margin should be re-read with that in mind — the +7.9% for v12 and
the +7.7% for v6 survive it; a 4% claim would not.

**Two rules from this.** Estimate the floor per-config rather than globally, since short
configs vary far more than long ones. And never write a measured number into a commit
message before the measurement has actually returned — I did exactly that, and it went into
the permanent record.


## L33 — A mechanism measured in isolation measures the isolation (2026-08-30)

`v15_lifted_veto` lifted Inductor's 68-SM veto, which really had disabled every Triton
GEMM template and therefore all GEMM epilogue fusion on this 66-SM card. Isolated probe:
**1.58x** (5.576 -> 3.537 ms on config 6's FFN pattern). In the real model: **+2.4% on
config 6, -1.4% overall against its own parent.** Zero.

The work the GEMM epilogue would absorb was already being absorbed by Inductor's
pointwise fuser, into the LayerNorm kernels — the veto gates `use_triton_template` only,
and `max_autotune_pointwise` was never vetoed. Lifting it MOVED work between kernels.

The probe showed 1.58x *because* it was isolated: a bare FFN has no LayerNorm to fuse
into, so the template is the only available mechanism. **Before trusting a component-level
win, check whether the work it saves is already being saved by something else.** Sharpens
[L32]: measuring the fix is not enough if you measure it where the bug is artificially
dominant. See docs/findings/22.

**RETRACTED:** I claimed the veto was "the likeliest single explanation for the gen 11-14
plateau". Removing it changes nothing measurable, so it explains nothing. The plateau is
still unexplained. Same error as L28 — a cause asserted before the fix was measured.

## L34 — Independent convergence corroborates the READING, not the VALUE (2026-08-30)

Two research agents with disjoint territories independently found `min_sms = 68` and
independently proposed lifting it; one ranked it top. The agreement made the diagnosis
look strong, and it WAS strong — the reading was correct. It said nothing about the fix's
worth, because both were reading the same source file and neither had run the full model.
**Agreement between analysts sharing a method is not replication.** Treat converged
proposals as one hypothesis with good provenance, not two votes.


## L36 — A test can pass because its subject was never built (2026-08-30)

The lineage invariant sweep reported 113 green while **four candidates carried a live
silent-wrong-answer bug** (v9a, v9b, v11, v15 all return Inductor's static CUDA-graph
buffer instead of a clone). Whole file: 50 passed, 1 failed. One candidate per process:
v9a FAILED, v9b FAILED, v16 FAILED, v13 passed.

Cause: Dynamo's `cache_size_limit` is 8 and shared per process. Once exhausted,
`torch.compile` silently falls back to EAGER, which allocates a fresh output every call
— so the static-buffer test passes **because the candidate was never compiled**. Green
was produced by a second defect, not by correctness.

**Whenever a test's subject is produced by a lazy, budgeted or fallback-capable mechanism
— a JIT, a compiler, a cache, a feature flag — the test must assert the mechanism
actually ran.** v15's mechanism test does this (asserts a `triton_tem` kernel appears);
the invariant sweep did not. Fixed with `torch._dynamo.reset()` in the fixture.

Twin of [L24] ("correct because of how the harness calls it is not correct"): *green
because of how the test was run* is not green. See docs/findings/24.


## L38 — Verify that a check can FAIL before trusting that it passed (2026-08-30)

Built a contention guard so several agents could share one GPU without corrupting each
other's timings. The lock file works. The `nvidia-smi` foreign-process check **does not**:
with a process holding a 16 MB CUDA tensor and confirmed alive, two identical trials seven
seconds apart gave `"893453, [N/A]"` (detected) and `""` (not detected). A clean report
from it means nothing.

This is [L36] one level up, written less than an hour after L36 itself. There the lesson
was "a test can pass because its subject was never built"; here a GUARD passes because its
sensor saw nothing. **A guard is only evidence if it is capable of firing — test it against
a condition you deliberately created**, not against the quiet case you hope for.

Consequence: **the v15 sweep overlapped research agent C, which was benchmarking Triton
kernels on the same GPU** (v15 15:57-16:00 UTC, agent C finished 16:01). Finding 22's
quantitative conclusion — that lifting the 68-SM veto buys nothing — rests on margins
(-1.4% geomean, +2.4% on config 6) that a co-resident benchmark can manufacture. Its
MECHANISM argument stands (read from profiles), but the number is provisional until
re-measured under the lock. v16 and v17 sweeps were clean.

Corollary: every tool that measures must take the lock, and subagents must either take it
or not measure. See docs/findings/26.


## L39 — Some fixes are invisible to the measurement that ranks them (2026-08-30)

v18 removes a **2.25x** silent loss (graph capture fails when the caller allocates its
input outside `inference_mode`, which is what the graded harness does at line 529). On the
standard sweep it measures **2.765x against v17's 2.759x -- identical**, because our
harness runs accuracy tests first and so always exercises the masked path.

**A search that promotes strictly on measured score would rank this "no change, why
bother" and discard it.** The evidence has to come from a dedicated experiment holding one
variable, plus a test pinning the parent's degradation so the insurance cannot rot.

Consequence for spec 07: A3 (time-to-signal) and the stage-1 screen both assume the sweep
can SEE the effect. For a robustness proposal that assumption is false and the proposal
needs a bespoke falsifier, not a screen verdict. The rubric does not distinguish these yet.

Also today: v15 re-measured on an idle GPU gives **2.634x against the contended 2.618x** --
0.6% on the geomean, 3.7% on config 6. Finding 22's CONCLUSION stands; its per-config claim
("+2.4% worse on config 6") was contamination and is actually -1.4% better. A result can be
right for the wrong reason, and re-running is the only way to tell. See docs/findings/27.


## L40 — Writing down a lesson is not the same as building the thing it demands (2026-08-30)

**The user looked at the dashboard and asked why it showed one long lineage instead of
branching paths.** It was not a rendering problem. Measured: every candidate has exactly
`generation - 1` git ancestors -- a perfectly linear chain across eighteen generations.

The cause was my own branching discipline: each candidate branch was cut from `ben`'s tip
(to inherit the latest harness), and every candidate is merged back INTO `ben`, so each new
branch inherited every earlier candidate. The spurs in `git log --graph` are decorative;
`merge-base --is-ancestor` says it is a line.

**This is [L1], rebuilt by the person who wrote L1** -- which named this exact degeneracy
on day one and said "branch first". I branched, in a form satisfying the words and none of
the mechanism. Finding 21's fix to the clade *criterion* then masked it by pushing the age
correlation to -0.158.

Impact so far: the same top-3 nodes, reordered (v9b/v8/v9a either way), so no expansion
went to the wrong place. Luck, not design; the dilution grows every generation.

Fix: CMP now reads the registry's declared parents (`clade_stats_by_candidate`,
`sample_candidate`), and `tests/bench/test_lineage_topology.py` enforces from gen 19 that a
candidate's candidate-ancestors equal its declared ancestors -- i.e. it was cut from its
parent, not from the trunk.

**Every structural claim needs an executable check.** The ones that have them -- oracle
manifest, append-only ledger, tolerance lock -- have never silently broken. The ones living
only in prose -- "git is the tree", "the premises in matrix.py are true" ([L35]) -- have
both now been found false by someone LOOKING rather than by the system noticing. Third
member of the [L36]/[L38] family in two days: an assurance nobody arranged to be capable of
failing. See docs/findings/28.


## L41 — A probe may propose; it may never conclude (2026-08-30)

Three measurements of v19 on config 6: op-level probe **3.84x better**, model-level probe
**16.2% worse**, harness sweep **+0.4%** (authoritative, and v19 is flat).

Both wrong numbers came from hazards already written down in this repo, by me, in the
previous two days:

  * The op-level probe compared against `F.layer_norm` + kernel called separately in eager.
    The real candidate never does that -- **Inductor fuses the add and the norm into one
    kernel**, and that is what v19 had to beat. This is [L33] sharpened: isolation does not
    merely shrink an effect, it can INVENT one that was never available.
  * The model-level probe held the baseline and the candidate in one process -- exactly
    [finding 05], the co-residency spill that once inflated a baseline 4.1x, and the reason
    `run_matrix` times arms in isolation. I wrote `bench/gpu_lock.py` hours earlier; it
    guards processes, and I put both models in one.

`run_matrix` embodies six rules (arms isolated, one config per subprocess, correctness
before timing, min-of-N under unlockable clocks, refuse dirty tree, refuse contended GPU).
**Every ad-hoc probe opts out of all six.** Third recurrence of [L9]. The rule: a number
that will change a decision comes through the harness, and when a probe disagrees with the
harness the probe is wrong until proven otherwise -- the correct prior all three times.

Also established: the pointwise bucket is NOT the opportunity the profile suggested.
Inductor was already fusing it well enough that deleting it changes nothing. On config 6 --
84% of wall time -- what remains is attention. See docs/findings/29.


## L42 — Test the settings the harness DEFAULTS to, not the ones the spec implies (2026-08-30)

Every candidate from v5 to v23 hardcoded `is_causal=True`. On a non-causal input the
frontier returned **3/4 of its output wrong** (max_abs 1.67e+00 vs a 2e-3 tolerance), with
all 177 tests green -- because every announced config is causal so nothing exercised the
other branch. **But the reference benchmark's own default is `causal: bool = False`.**
Everything we ever measured used a setting the harness does not default to. L24 at its
most literal. Fixed in v26 by delegating non-causal to the unmodified baseline.

The audit rule is now **7 for 7**: padding ratio, eager baseline, dtype, input scale,
allocation context, process contention, causal flag. Always the same question -- *what does
this depend on that we never varied?* Not one was found by the search loop.

**For any flag, dtype or mode the harness exposes, its DEFAULT is a separate test case from
the value the specification implies, and it is the more dangerous one, because it is what
runs when nobody passes an argument.**

Also, accidentally: v26's causal path is byte-identical to v23's, so its sweep re-measured
the same code and got **3.015x -> 3.103x, +2.9% on the geomean** with total wall time
unchanged. Configs above a millisecond reproduced within 0.6%; every deviation came from
sub-millisecond rows. Direct evidence for [L29]'s floor, and a caution that the geomean
weights a 0.06 ms config equally with a 57 ms one. See docs/findings/32.


## L43 — "The reference cannot run it" is a claim with THREE different scopes (2026-08-30)

Config 14 had 28 ledger rows and no information: 27 `status="oom"` plus a truncated
traceback. Pulling it apart, "it OOMs" turned out to be three unrelated statements that
had been used interchangeably, and only one of them is universal:

1. **The reference's ALGORITHM, on any hardware that exists.** Line 97 materialises a
   [B,H,S,S] score tensor = **18.63 TiB**. Measured by asking the driver: even one head
   of one sequence is 37.25 GiB and is refused. Not a batch-size problem, not a property
   of this card.
2. **The forward signature's floor, on THIS card.** 12.21 GiB in + 12.21 GiB out =
   24.42 GiB of tensors no implementation removes, against 15.99 GiB. An 80 GiB card
   clears this and still hits (1).
3. **This box, this day.** WSL2 oversubscribes to a measured 30 GiB, so 24.42 GiB is
   nominally reachable — and the harness throws it away: `generate_random_case` frees its
   first 12.21 GiB buffer into the allocator cache, then the **3.05 MB mask splits that
   segment and pins it**. `empty_cache()` cannot release a partly-used segment. A 3 MB
   mask costs 12.21 GiB.

A report that says only "it OOMs" is not wrong, it is three claims wearing one coat, and
the strongest of them (18.63 TiB, universal) is the one that gets lost. **State the scope
of every impossibility claim: the algorithm, the interface, or the machine.**

## L44 — Correctness at a shape with no reference is available, and proxies were leaving it on the table (2026-08-30)

Finding 09 recorded `correctness.passed = null` for config 14 and checked proxy shapes.
That was right and it was weaker than what existed. Two constructions verify at the REAL
sequence length:

- **The causal-prefix theorem.** Under causality with an all-valid mask, `model(x[:, :P])
  == model(x)[:, :P]`. So the UNMODIFIED reference is an oracle at S=100000 for the first
  P rows — real model, real input, the harness's own `compare_outputs`. Measured: P=4096,
  passed, max_abs 8.66e-4, 0 failed elements.
- **A blocked fp64 evaluation of the reference's arithmetic**, query axis blocked (exact:
  softmax reduces over keys), deliberately NOT online softmax so a rescaling bug in our
  flash path cannot be mirrored by the oracle. Covers every row. Built and validated
  against the reference at S <= 4096; **its full S=100000 run has not completed** (the GPU
  has been continuously occupied), so at the time of writing only the prefix oracle has
  actually returned at the real shape. Do not report it as if it had.

The generalisation: **when the reference cannot run, look for a smaller instance of the
reference that is provably the same computation, and for a re-evaluation of its own
arithmetic in higher precision.** Both were available for two days and nobody looked;
`passed = null` had stopped the search.

What is still not available is `|candidate - reference|` at S=100000, and no amount of
this produces it. Say so.

## L45 — Our own TF32 baseline spends 40% of the tolerance budget (2026-08-30)

Validating the fp64 oracle against the reference, at four sequence lengths:

```
matmul precision "highest"  (strict fp32)   1.24e-06 (S=1024)   1.92e-06 (S=4096)
matmul precision "high"     (TF32)          8.086e-04           8.086e-04
```

Identical to four digits across a 32x change in sequence length — [L4]'s representation-
floor signature exactly. TF32 keeps 10 mantissa bits (eps ~4.9e-4) and these outputs have
mean magnitude 0.798.

CLAUDE.md rule 5 mandates the torch.compile+TF32 baseline, correctly, because it is the
strongest honest comparison. The cost had never been quantified: **the reference itself is
8.09e-04 from exact, 40% of the locked 2e-3 absolute budget, before any candidate runs.**
It does not make the harness's pass/fail unfair — both arms are TF32 — but it is exactly
half of [L26]'s "the margin is thinner than it looks", and it is the reason a fp64 oracle
can only certify at 1.19e-3 rather than 2e-3.

## L46 — A model that has only ever been called at one shape is not known to work at two (2026-08-30)

Building the config-14 capability path meant calling one warmed model with a second batch
size. The frontier returns **an (8, 128, 128) tensor for a (1, 128, 128) input**: v13
captures a CUDA graph on the first forward, keeps `_static_x` sized to it forever, and
`_static_x.copy_(x)` broadcasts a smaller input across the buffer — the first sequence
computed eight times and returned as eight rows. In the other direction it raises.

177 tests were green. **Every sweep builds a fresh model per config, so no model in this
project's history had ever been called at two shapes.** [L24] in its purest form: correct
because of how the harness calls it.

The general rule this adds to [L24]: **enumerate what your harness holds CONSTANT across
every invocation, not just what it varies.** The audit rule (7 for 7 at [L42], now 8) has
been about untested *values* of things the harness varies. This is the other half — a
dimension the harness never varies at all is not tested, it is assumed, and it does not
appear in any parameter list to remind you.

## L47 — A cleanup call destroyed a completed measurement (2026-08-30)

The config-14 capability run reached the oracles with everything already established: 32
sequences of 100,000 tokens computed, peak memory recorded, the causal-prefix check
passed. It then died on

```python
torch.cuda.empty_cache()      # AcceleratorError: CUDA error: out of memory
```

— an unguarded *cleanup* call, on a GPU another agent had filled. The child process
emitted no `__RESULT__`, so the parent recorded `status="crash"` with a traceback, and
several minutes of measurement that had already succeeded reached the ledger as nothing.

Two things generalise.

**A measurement pipeline's later stages must not be able to discard its earlier ones.**
`run_matrix` already gets this right at one level — one config per subprocess, so a
config that dies does not cost the run — and got it wrong one level down, where an
eight-stage capability path was one function with one exit. The fix is the same principle
applied inside: each stage reports its own failure into the row, and the row is returned.

**The dangerous line is rarely the one doing the work.** `empty_cache()` frees memory; it
reads as incapable of failing, which is why it was the one call not wrapped. Same family
as [L38] and [L36] — the assurance nobody arranged to be capable of failing — but from
the other side: the *operation* nobody imagined could fail.

## L48 — The optimisation that halves the work can be the thing that stops it running (2026-08-30)

The fp64 oracle truncates its key axis at the causal diagonal. Obviously right: it halves
the arithmetic, it is exact, and it is the same saving the candidate's own attention
takes. It also makes **every loop iteration a different allocation size** — at S=100000,
~1500 distinct multi-hundred-MB tiles that the caching allocator can never reuse.

It failed with `torch.AcceleratorError: CUDA error: out of memory` at S=32768 **with
14.18 GiB free**. Three things about that error misdirected the diagnosis for an hour:

- it is a *driver* OOM, not a `torch.OutOfMemoryError`, so it does not carry PyTorch's
  helpful "of the allocated memory X is allocated by PyTorch" breakdown;
- a driver OOM poisons the CUDA context, so every later call fails too — including
  `empty_cache()` ([L47]) — which makes the *last* thing to fail look like the cause;
- the GPU genuinely was contended at the time, which supplied a plausible wrong answer.
  It failed identically on an idle card.

Fixed-width tile plus in-place softmax: one allocation, reused every iteration. It costs
exactly the 2x the causal truncation saved, and it is the difference between an oracle
that runs and an oracle that is merely described.

**A memory-shape argument beats a FLOP-count argument when the allocator is the binding
constraint**, and a loop whose allocation size is a function of the loop variable is the
signature to look for.

## L49 — The candidate is as accurate as the reference, and neither is very accurate (2026-08-30)

Measured at config 14's real shape, B=1, S=100000, every row:

```
|candidate - exact|   8.0913e-04
|reference - exact|   8.086e-04         (measured at S <= 4096, flat in S)
```

Three digits apart. Both are the TF32 representation floor ([L45]), not their own
arithmetic — so **the fp16 intermediates and flash attention over 100,000 keys contribute
almost nothing on top of what the baseline already spends**. Two consequences worth
carrying:

- The precision worry this project has carried since [L4] ("the budget is nearly spent")
  is now better localised. Most of the budget is spent by TF32 in *both* arms, not by our
  fp16 path. That is why the same 8-9e-04 keeps appearing on unrelated configs.
- It also means a candidate cannot buy much margin back by being more careful internally.
  The floor is in the comparison, not in the candidate.
