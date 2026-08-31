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



## L43 — Price the COMPULSORY traffic before proposing a cache fix (2026-08-30)

A-06 proposed pinning the fp16 weight arena in L2 with a persisting-access window, priced
at up to 23% of config 6 on the assumption that a 327 MB activation stream evicts it.
Measured by contrast (`ncu` is unavailable under WSL2 — `ERR_NVGPUCTRPERM`): the frontier's
FFN kernel runs **2.1% above a pure-streaming floor measured on its own activation
traffic**, and forcing the weights to miss — a 32 MiB arena instead of 64 KiB — costs
**+75%**. The real kernel is already saving **94.8% of the full-miss worst case**.
Installing the window changes it by **−0.25%**. cuBLAS's QKV projection is in the same
position: +2.2% over its floor against a +75% worst case.

Two general things, both cheaper than the probe that established them:

**A cache optimization's ceiling is the RE-fetched traffic, never the total.** Persistence
converts capacity and conflict misses into hits; the first read of a byte is compulsory and
survives any policy. The whole four-layer weight set is 768 KiB — 0.002% of what config 6
moves — so even a perfect cache had nothing to win. That bound is arithmetic and needed no
GPU.

**The reuse distance was derivable in one line ([L37]).** A weight line is re-referenced
once per CTA, and a CTA touches 80 KB of activation, so the weight tile is touched ~600
times per sweep of a 48 MiB L2. A line touched 600 times per sweep is never the LRU victim.
The probe's measured crossover confirms it: resident at a 1 MiB arena, evicted at 32 MiB —
the boundary is exactly where the reuse distance crosses the cache size.

**And writing that condition as a test immediately found the row it does not cover.**
Config 8's four-layer weight set is **48.00 MiB, equal to this card's L2 to the byte** — the
one announced shape where eviction is genuinely possible. Prose would have shipped the
generalization; the assertion failed on it. Measured directly: a window over the whole arena
is **-0.62%**, because config 8's GEMMs run at **93-98.5% of measured tensor-core peak** and
are compute-bound, 3.8-5.4x above their bandwidth floor. A memory-system policy has no time
to give back there however the cache behaves. Fourth member of the [L36]/[L38]/[L40] family
in three days, and the first time the executable check was written *before* the claim
escaped into a finding rather than after.

**Both positive controls fired ([L38]), and that is what makes the null usable.** The
32 MiB arm proves the contrast can see weight traffic at all; the same arm plus a window
goes **4.728 → 2.707 ms, +42.7%**, proving the ctypes shim, the driver path and sm_89 all
work. *The feature works and has nothing to do* is a much stronger claim than *we measured
nothing*, and it costs one extra arm.

The [L33] objection ("a mechanism measured in isolation measures the isolation") is
answered by direction, not by an end-to-end run: the FFN megakernel is the most favourable
site in the model for this idea, isolation *inflates*, and **an inflated null is still a
null.** That is the one case where an isolated probe may be decisive without spending a
sweep — and this negative also disposes of the same intuition that produced v3's L2-sized
chunking, killed by the g10 ablation ([L17]). See docs/findings/33.


## L44 — A mechanism's speedup and its error budget can be scissors in the same variable (2026-08-30)

fp16 MMA accumulation. The hardware reading was right and three agents converged on it:
`tl.dot(out_dtype=tl.float16)` really emits `mma...f16.f16.f16.f16` against
`...f32.f16.f16.f32`, and really measures **1.62x** in an MMA-saturated loop. It is still
worth nothing, for two independent reasons that meet in one variable.

Both conditions depend on the contraction depth K, and in this architecture
`K == d_model == ffn_dim`, so one shape parameter drives both — in opposite directions:

* **fast** needs intensity above the ridge point: `d_model >= 359` on this card
* **accurate** needs `eps_fp16*sqrt(K) <= atol`: `K <= 16.8` at the locked 2e-3

A 21.4x gap, and it WIDENS on better hardware — a higher peak-FLOPs-to-bandwidth ratio
pushes the ridge up while fp16's mantissa stays 11 bits. Measured confirmation from both
ends: the fused FFN runs at **99.2-100.1% of measured bandwidth** at config 6's token
count (so all four accumulator arms measure 1.000x, 1.001x, 0.998x, 1.000x — no win, not
a small one), and an fp16 accumulator at **K=16, the shallowest MMA sm_89 can issue,
already spends 140% of the tolerance budget.** The affordable region is not narrow; it is
empty.

**The transferable move: before building, check whether the mechanism's speed condition
and its accuracy condition are functions of the SAME shape parameter.** If they are and
they point opposite ways, the question is not "does this help" but "is the window
non-empty", which is arithmetic on measured device properties and costs no GPU time.
Finding 08 is the precedent for the write-up; this is the precedent for killing it before
the build. See docs/findings/30.

Corollary on reporting: the single-site arms PASS every config — and cost 15-34 points of
tolerance budget for a measured 1.001x. "Passes" was never the bar. Per L26, a candidate
at 90-107% of budget does not survive the input-scale shift that L26 already measured.


## L45 — Fusion strands the work that was riding along for free (2026-08-30)

v34 predicted 36 -> 20 kernels per forward at config 2 and the first build measured **24**.
The four extra were `.float()` on the attention out-projection, one per layer. In the
parent that cast was invisible: Inductor had fused it into the LayerNorm kernel that
followed. Moving the residual add and norm2 *into* the megakernel deleted that kernel, and
the cast — which had never been a kernel in its own right — became one.

Fixed by handing the out-projection over in fp16 and widening inside the megakernel, which
is bit-identical (it is an fp16 GEMM over fp16 operands, so the value is already fp16) and
halves that tensor's traffic as well.

**Before claiming a fusion removes N kernels, ask what was fused INTO the thing you are
removing.** Inductor's pointwise fuser scatters small ops into whatever large kernel is
adjacent, so deleting a large kernel evicts its lodgers. The count is only knowable by
counting — which is also why v34 counted before and after rather than reasoning about it
(L36). See docs/findings/39.


## L46 — The optimization worked well enough to invalidate the statistic measuring it (2026-08-30)

Two screens of v34, same commit, same clean tree, gave **+8.1%** and **+0.7%**. Configs 7,
8 and 10 reproduced to four decimal places in both; only config 2 moved, by 33%.

That contrast is what made it diagnosable: if everything had been noisy the right response
would have been "wider floor, take more samples", and it would have been wrong.
**Reproducibility on the arms a change does not touch is a free control, and it separates
a discrete state difference from variance.** Sampled ten times against the parent's five:

    min-of-N     v34 0.0440-0.0471   v26 0.0604-0.0614    NO OVERLAP, 10/10
    median       v34 0.0451-0.0666   v26 0.0604-0.0676    overlapping

The cause is not noise and not a bug. v34 removes 16 of 36 kernel nodes and 8.6 us of
device time from a config that finding 16 measured at **232 us CPU against 126 us GPU**.
The GPU side shrank; the CPU side (`cudaGraphLaunch`, the memcpys, the output clone) did
not. The minimum still catches the GPU, but the median now samples the CPU's jitter.
`run_matrix` scores on a median.

**A speedup can move a config across the CPU/GPU boundary, and the statistic that was
appropriate on one side is not appropriate on the other.** min-of-N — which this project
already prescribes for itself because the clocks are unlockable — reports the same result
ten times where the median cannot decide. Config 2 is one of the four SCREEN configs, so
every future candidate that shrinks its GPU side inherits an unreadable screen verdict,
and the screen has no way to distinguish that from a regression.

Companion to [L29] (the floor is per-config, not global): here the floor is not even
per-config, it is per-*candidate* on the same config. See docs/findings/39.

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


## L50 — A fix can make a second, dormant defect reachable for the first time (2026-08-30)

Merging g33 and g34 -- two orthogonal children of the same parent, neither wrong on its
own -- produced a silent wrong answer that **neither candidate could exhibit alone**.

v34 latches `_nomask` once, in `_prime`, and its `_try_capture` ELIDES the mask buffer
when `_nomask` is True. That is unreachable in v34 by itself: v13's shape latch makes the
model raise on the second shape before a different mask can ever be presented. v33's whole
purpose is to remove that raise. Measured on the merge with the reset deliberately
disabled -- warm at (8,128,128) unpadded, call at (16,128,128) with padding 0.4:

    max_abs 3.1612e+00     69407 / 262144 elements past the locked tolerance

Twenty-six percent of the output wrong, on a path that did not exist in either parent.

**A candidate's correctness argument is conditional on what its siblings do NOT do.**
v34's "the mask check runs once and that is enough" was true, and stopped being true when
something else in the same lineage removed the guard that made it true. So the question to
ask at a merge is not "are both halves correct" -- both were -- but "does either half
depend on a limitation the other removes". Here the dependency was one attribute derived
in a function that runs once per model.

Companion to [L14] (when you merge, assert nothing was dropped): nothing WAS dropped. The
defect came from something being *added*. See docs/findings/41.


## L51 — The profiled kernel count carries a constant offset; only the difference is portable (2026-08-30)

Counting device events over ten profiled forwards, GPU lock held, five repeats:

                     alone        inside the full 279-test suite
    v26 / v33        36.0 x5      35.3
    v35              20.0 x5      19.3

A **constant deficit of seven events per profiled window**, identical for both models --
the profiler drops events in a loaded process. The absolute count is not reproducible
across environments. The DIFFERENCE is exactly 16.0 in both.

This nearly cost a real mechanism test: `assert n_parent == approx(36.0, abs=0.5)` passed
in isolation and failed in the suite, and the tempting reading was "widen the tolerance".
The correct reading is that the two numbers are not the same measurement. **Assert the
quantity the mechanism claims -- here, sixteen nodes removed -- not the absolute reading
that happens to contain it**, and take both readings in the same process so the offset
cancels. Same shape as [L44]'s free control: what reproduces tells you what the
perturbation is. See docs/findings/41.


## L52 — Measure with the protocol you will be scored by (2026-08-30)

The graded benchmark interleaves ABBA/BAAB rounds ("to reduce thermal/clock-order bias"),
pools the samples, and takes `baseline.median / optimized.median`. `bench/run_matrix.py`
takes `min(median, median)` per arm, **not interleaved**, with the candidate compiled and
autotuned BETWEEN the two arms — on a GPU whose clocks cannot be locked.

It inverted two signs. v34 vs v26: our ledger said **+6.1% worse on cfg 1 and +5.6% worse
on cfg 9**; the graded protocol says **-0.8% and -6.9% BETTER**. The research agent
predicted it from the kernel census — v34 launches strictly fewer kernels there, so a
regression was mechanically implausible.

`bench/end_to_end.py` was built days ago for exactly this check and its docstring says
"nobody has ever checked they agree". Nobody ran it. A session was spent ranking
candidates on a quantity that is not the score.

**When your harness differs from the one that scores you, the burden is on you to
demonstrate agreement per config before trusting any ranking.** A difference written down
in a comment is still a difference. Exposure is worst on sub-millisecond configs — which
is exactly where all remaining score lives. See docs/findings/42.


## L53 — A tuner that times two arms in sequence is a benchmark (2026-08-30)

The graded harness cannot rank candidates either, and the noise has a shape: its BASELINE
arm — byte-identical reference code — spreads **0.1% on config 8, 33% on config 2, 39% on
config 3**, i.e. inversely with config size, worst exactly where all remaining score lives.
The OPTIMIZED arm is stable to the last digit. So the reported `speedup` inherits a noisy
denominator; **rank by the candidate's own time against a FIXED reference, never by a
per-run ratio.** Mechanism: round 1 of 100 timed calls reads 932.9 us where rounds 2-3 read
250.9 us stable to 0.1 us — ~130 calls of settling after graph capture against 20 warmup
iterations.

The same error one level down: a *tuner* that times two arms in sequence is a benchmark and
inherits every benchmark's bugs. g36's predicate used `do_bench` (flushes L2, pays a launch)
to pick tiles for kernels that run L2-hot inside a replayed graph; and its first `plan()`
call read `F.linear` at 306 us against a clean process's 21.5 us — cuBLASLt setup inside the
timing window, a fake 17.6x.

**Interleave, discard the cold round, and use a timer whose regime matches the call site's,
or write in the code why not.** The protocol that worked: ABBA-interleaved, both resident,
cold round discarded, min of four, with byte-identical configs as an in-run control. See
docs/findings/42.

## L54 — If a candidate does work at construction, do not time it immediately after constructing it (2026-08-30)

The harness's job is to measure the steady state a grader will see, and a grader does not
rebuild the model between the two arms it compares. Our isolated protocol was built to
avoid finding 05's co-residency spill and did so correctly; the cost was never "a few
percent of drift" but a structural misreport of every candidate that plans, tunes or warms
at build time.

Corollary for authors: if your predicate times anything, say so in the docstring, because
it changes which measurement of your candidate is meaningful.

See docs/findings/45.

## L55 — When two candidates tie inside the noise, decide on what is not measured (2026-08-30)

Six candidates were separated by less than the 7.1% single-arm floor this session, and the
search spent real GPU time trying to rank them. The tie-break that actually held up was
never a number: it was which candidate carried more proven correctness fixes. **A
statistical tie is an instruction to stop measuring and start comparing guarantees.**

---

## Addendum — the gap was noise, and v38 wins outright

Finding 49 above concluded v36 and v38 were tied inside the floor and broke the tie on
correctness. **Replicated measurement shows there was no gap at all**, and v38 is faster.

`bench/abba.py`, 6 rounds, 200 warmup iterations, all arms resident, cold round discarded,
configs 2 and 8 as in-run controls:

    config     v36 median    v38 median      verdict
      3          52.22 us      52.22 us      IDENTICAL
      2          47.10 us      47.10 us      IDENTICAL   (control)
      8        6593.54 us    6593.54 us      IDENTICAL   (control)
     12          95.23 us      74.75 us      v38 1.274x FASTER

Every per-config difference this decision was agonised over came from **one ledger row per
candidate**. Config 3 in particular: v36 read 0.0666 ms and v38 0.0973 ms — a 46% gap, on
**identical launch counts of 20**. Under replication both read 52.22 us to the hundredth of
a microsecond.

The 200-iteration warmup is what made this resolvable. The graded harness warms 20 against
a settling time of ~130 calls after CUDA-graph capture (finding 42's addendum); at 512
tokens that leaves the measurement dominated by whatever the host was doing.

**v38 is the submission, and now for the simple reason as well as the good one.** It is
faster where the two differ, identical everywhere else, and it carries four correctness
fixes v36 lacks.

See docs/findings/49.

## L56 — A per-config difference from one row per arm is not a difference (2026-08-30)

Two candidates were separated by 0.037 of weighted_score, decomposed per config, argued
about, and resolved on correctness grounds — and the entire gap was single-sample noise
on two sub-millisecond rows. The decomposition was rigorous and the input was one
measurement each.

**Replicate before you decompose.** Cheap configs are cheap to replicate: this run cost
under two minutes and overturned a conclusion built on a careful analysis of noise. The
byte-identical control arms (2 and 8, reading identical to the hundredth of a microsecond)
are what make the config-12 result believable, and they cost nothing to include.

See docs/findings/49.

## L57 — A protocol built to avoid one distortion will find the other one (2026-08-31)

Every measurement protocol in this project was introduced to fix a specific, real defect in
its predecessor, and each introduced a new one on a disjoint part of the matrix:

    min(median,median) isolated  ->  fixed co-residency spill, broke small-config ranking
    ABBA all-resident            ->  fixed small-config ranking, broke large-config spill

That is not a sequence of mistakes; it is what happens when a single number is asked to
cover a 5000x range of problem sizes on a card with 16 GB and no clock lock. **The
resolution is not a better protocol but a per-regime one, chosen by measured shape — the
same dispatch discipline the candidates themselves are held to (rule 2).**

The tell, both times, was a byte-identical or known-stable arm reading differently between
runs. That is why every comparison in this project now carries a control it does not need.

See docs/findings/50.

## L58 — A capped win is not a small win, it is a zero (2026-08-31)
Five of fourteen configs (3, 6, 7, 11, 13) are past the clip, and config 6 alone is 83% of
matrix wall time. Work landing there scores nothing at all — not a little, nothing. This
project spent its first two days optimising config 6.

The corollary that took longer to learn: **a regression on a capped config is also free**,
which is why v37's 1.6x streaming defect on config 6 survived two sweeps and two commits
unnoticed. The objective that hides the win hides the loss with it.

Proposed as L58 in 49; renumbered on merge.

## L59 — Before you argue about a timer's regime, check that it can resolve the thing (2026-08-31)
[L53] says to use a timer whose regime matches the call site's, and three findings (48, 50,
the g41 audit) have been about cache regime — L2-hot against L2-flushed, a 2.24x gap that
made one headline wrong by that factor. That is a real axis and the reasoning about it was
correct. It is also not the only way an instrument can fail, and here it was not the one
that cost the score.

`do_bench` times each call with a pair of CUDA events. Their quantum is 1.024 µs. Ranking
eight variants of a 1.9 µs kernel with it produced a table in which five entries were the
same number — not noise around a true ordering, but **no ordering at all**. Every downstream
rule then behaved correctly on a degenerate input: `min()` returned an arbitrary tie, the
`DECISIVE` margin could not be cleared, and the tiebreak kept the incumbent. Eighteen
generations of correct decisions on a blank input.

The cheap check is one line and nobody had run it: **print the sweep, and look at how many
distinct values it contains.** A grid of N arms that yields two distinct readings has not
ranked anything. If the spread of the whole grid is comparable to the instrument's quantum,
the instrument is the wrong one, and no amount of replication, minimum-taking or
winner's-curse correction will fix it — those all assume a noisy signal, and this is an
absent one.

The general form: an instrument has a *resolution* as well as a *regime*, and resolution
failures are invisible in exactly the way regime failures are not. A mis-regimed timer gives
you confident wrong numbers you can catch by cross-checking against another regime. An
under-resolved timer gives you *ties*, which look like "the arms are equivalent" — an
answer, and a plausible one, and the reason this sat unexamined from generation 23 to 42.

Proposed as L62 in 53; renumbered on merge.

## L60 — A blunt instrument is stable; sharpening it buys accuracy with variance, and you must check where you spent it (2026-08-31)
`do_bench` could not separate config 2's eight tiles, so the sweep fell through to its
tiebreak — every time, on every shape, in every process state. That is a *systematic* error
and it has a property nobody had noticed was doing work: **it is perfectly reproducible.**
Eighteen generations of stable, wrong tile selection, and the stability was load-bearing,
because a plan that does not vary adds no variance to the measurements taken of it.

Replacing it with a timer that *can* resolve the arms fixed the shape whose true margin is
28% and destabilised the shape whose true margin is 2% — from one tile in six runs to three.
The new instrument is better and it converted a bias into a variance, and the variance
landed on exactly the shapes where the decision was closest and therefore mattered least
per-decision and most in aggregate.

The lesson is not "keep the blunt instrument". It is that **replacing a selection rule's
instrument is not complete until you have measured the rule's OUTPUT for stability, not just
its accuracy** — run the sweep several times and count distinct answers, on the shapes where
the margin is small as well as the one that motivated the change. The decision rule's
threshold was calibrated (`DECISIVE = 0.10`) against the old instrument's noise; a sharper
instrument with a different noise distribution needs the threshold re-earned, or needs the
answer required to replicate before it is acted on. Neither is expensive. Not noticing is.

Proposed as L63 in 53; renumbered on merge.

## L61 — When the noise is one-sided, replicate to REDUCE it, never to VOTE on the answer (2026-08-31)
Finding 53 established that contamination on this harness only ever makes a reading
slower, and then proposed a fix that requires two rankings to agree. Those two statements
are incompatible and it took building the fix to see it. If the noise is one-sided, it
lands on an arbitrary arm, and a `min()` over the grid returns *the arm the noise missed* —
so two sweeps disagreeing is the expected outcome even when the underlying margin is 28%,
and requiring agreement makes the rule revert more often the noisier the machine is. The
measured cost was the parent's only scoring row, lost in 5 of 10 processes.

The correct use of a replicate under one-sided noise is to take the **floor per arm** and
then decide once. That estimator is already this project's house rule for timing a card
whose clocks will not lock; the thing that was new was applying it *inside* a selection
rule rather than to the selection rule's output.

The general form: **before choosing how to combine repeated measurements, ask what shape
the noise has.** Voting, averaging and flooring are right under different noise models and
wrong under the others, and "replicate it" is not a decision until you have said which.
A vote is right for symmetric noise around a true value. A floor is right for one-sided
contamination of a true minimum. Picking the wrong one does not merely waste the
replicate — it can be worse than not replicating at all.

Proposed as L64 in 54; renumbered on merge.

## L62 — An isolation that removes the phenomenon is not a control, it is a different experiment (2026-08-31)
The first stability probe of this generation gave every arm its own process: no
co-residency, no allocator sharing, no interference. It reported both candidates perfectly
stable on both shapes and would have closed the generation as "nothing to fix". The
instability being investigated **exists only when a second model is resident**, which is
the condition `bench/abba.py` creates and therefore the condition under which every number
this project ranks candidates on is produced.

Isolation is the reflex here for good reasons — finding 05's co-residency spill, finding
45's construction-time planners, the one-config-per-subprocess rule — and every one of
those is about isolating a *measurement*. This was a measurement of a *decision*, and the
decision is made in the contaminated environment on purpose. **The probe must reproduce the
call site's environment, including the parts of it that look like contamination**, or it
measures a system that does not ship.

Corollary, and the cheap check: when a probe reports the phenomenon absent, that is a
result about the probe until it is a result about the code. Ask what the probe removed.

Proposed as L65 in 54; renumbered on merge.

## L63 — A silent fallback inside an instrument is a silent change of instrument (2026-08-31)
`hot_time` wraps `do_bench_cudagraph` in `except Exception: return do_bench(...)`, and says
so in its docstring, with a reason: failing closed on a tuner is worse than degrading. That
reasoning is defensible. What is not defensible is that the degradation is **invisible in
the number**: the caller gets a float, the reason string still says `hot_time`, and the
whole grid quietly reverts to the 1.024 µs instrument the previous generation was built to
remove.

It fires for a real, non-exotic reason — calling it outside `torch.inference_mode()` after
any model has run inside one — which is a condition every probe and test in this repo can
meet by accident, and three of them did in one afternoon.

The general form: an instrument that can silently become a *different instrument* must say
which one it was. A timer that falls back should return, or record, the path it took, and
anything that names an instrument in a reason string should name the one that actually ran.
The arithmetic tell here was free — a `do_bench` reading is an exact multiple of the event
quantum and a graph reading is not — and had it been checked in an assertion rather than by
eye, none of the three wrong probes would have gone anywhere.

Proposed as L66 in 54; renumbered on merge.
