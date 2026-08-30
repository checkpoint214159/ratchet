# The method

Twenty-four rules, distilled from 550 measured rows, 34 findings and 43 running
learnings across generations 1–34 of this repo. Each carries its origin so you can read
the evidence rather than take the rule on faith: `[Lnn]` is an entry in
`docs/findings/00-learnings.md`, `[Fnn]` is `docs/findings/nn-*.md`.

**These rules are the transferable part of this project.** The kernels are worth 3.1x on
one card; the rules are why the 3.1x is believable. Read this before you read any code.

---

## A. Measuring without fooling yourself

**A1. Correctness before timing, in the same process, always.**
A candidate that fails the gate is never timed. The time of a wrong answer is not a
datum. Tolerances (`rtol 0.02 / atol 0.002`) come from the graded harness's own CLI
defaults and are **locked** — never widened to make something pass. `[bench/run_matrix.py]`

**A2. One arm per process.**
Two models in one process once inflated a baseline **4.1x** (2037 ms against a true
446 ms) by forcing a spill to host memory. Two *processes* on one GPU is the same failure
with none of the defences: the allocator cannot see the other tenant and the timing loop
cannot know it was descheduled. **Two processes on one GPU do not produce two independent
measurements; they produce two wrong ones.** `[F05, bench/gpu_lock.py]`

**A3. Everything that will change a decision goes through the harness.**
`run_matrix.py` embodies six rules at once — arms isolated, one config per subprocess,
correctness before timing, min-of-N under unlockable clocks, refuse a dirty tree, refuse a
contended GPU. **Every ad-hoc probe opts out of all six simultaneously.** This lesson has
recurred three times, twice within hours of being written down. `[L9, L41]`

**A4. A probe may propose; it may never conclude.**
Three measurements of the same change: op-level probe **3.84x better**, model-level probe
**16.2% worse**, harness sweep **+0.4%** — the sweep is authoritative and the change was
flat. When a probe disagrees with the harness, **the probe is wrong until proven
otherwise** — the correct prior all three times. `[L41, F29]`

**A5. A mechanism measured in isolation measures the isolation.**
And not merely by shrinking the effect: isolation can *invent* an effect that was never
available, by comparing against a baseline the real system never runs (e.g. two ops
called separately in eager, when the compiler fuses them). `[L33, L41]`

**A6. Know your noise floor, publish it, and demand a margin.**
±7% here, measured from accidental replicates — not assumed. A candidate must beat the
incumbent by more than the floor or the search reports its own noise as progress. The
first search run "improved" by 2.7% inside a 3% floor. `[L29, F06]`

**A7. Beware the aggregate that hides the shape.**
Re-measuring byte-identical code moved the geomean **+2.9%**. Every config above a
millisecond reproduced within 0.6%; every deviation came from sub-millisecond rows — and
the geomean weights a 0.06 ms config equally with a 57 ms one. `[L29, L42]`

**A8. Your speedup is against the strongest baseline anyone would actually run.**
Ours was eager for twelve generations. The honest number was **1.69x, not 7.2x**, and two
configs flipped from win to loss. `[L12, F12]`

**A9. Score marginal gain over parent, never the cumulative number.**
Aimed at the cumulative ledger figure a selection rubric had rank correlation **+0.050**
— no predictive power at all. Aimed at the marginal gain, **+0.483**. Any selection
scoring candidates by their absolute number is scoring their ancestors. `[F20]`

**A10. Failures are recorded, not skipped.**
An infeasible point costs an evaluation, gets a ledger row and a large finite fitness —
never an exception that aborts the run, never a silent skip that hides the failure rate.
In comparable tuning spaces 68–78% of configurations fail to compile. **The failures are
the dataset.** `[bench/loop.py]`

---

## B. Provenance, so a number survives the session that produced it

**B1. A measurement is keyed to a commit sha.**
`(commit_sha, config_id)` is the primary key. Measurement lives *in-tree* so the sha the
ledger records actually describes the code that ran; a number whose provenance is a
throwaway script is not reproducible, and an irreproducible number is not evidence.

**B2. A dirty tree is recorded and barred, never silently accepted.**
Dirty rows are still evidence, but they are excluded from clade statistics and from
promotion. A sha that does not describe the code that ran is a false provenance claim.

**B3. Append-only means append-only.**
`bench/results.jsonl` is never edited, sorted in place, or pruned. Never rebase, squash,
amend or force-push a candidate branch — rewriting history silently reparents the tree
and invalidates every statistic derived from it. `[bench/README.md]`

**B4. A finding records what was measured and how, not what was believed.**
Where a number appears, it was produced on the machine named in the note. Where something
is inferred rather than measured, it says so. `[docs/findings/README.md]`

**B5. Negative results are first-class and are never deleted.**
A candidate that failed on 12 of 14 configs produced the finding that redirected three
subsequent generations — worth more than most successes. A rejected proposal is the
evidence that stops a future agent re-proposing it. `[F08, specs/07]`

---

## C. Assurance — the family that bit us four times in three days

This is the single most expensive theme in the project's history. Read all five.

**C1. Verify that a check can FAIL before trusting that it passed.**
A contention guard's `nvidia-smi` sensor, tested against a process holding a live CUDA
tensor, reported it on one trial and not on an identical trial seven seconds later. **A
guard is only evidence if it is capable of firing — test it against a condition you
deliberately created**, not against the quiet case you hope for. `[L38, F26]`

**C2. A test can pass because its subject was never built.** `[L36]`

**C3. Positive controls are what make a null usable.**
The L2-persistence probe ran a deliberately-evicting arm to prove the contrast could see
weight traffic at all, and the same arm plus the feature moved **+42.7%** — proving the
shim, the driver path and the hardware all worked. *"The feature works and has nothing to
do"* is a far stronger claim than *"we measured nothing"*, and it costs one extra arm.
`[L43, F33]`

**C4. Every structural claim needs an executable check.**
The claims that had one — oracle manifest, append-only ledger, tolerance lock — have never
silently broken. The claims that lived only in prose — "git is the tree", "the premises in
the matrix are true" — were **both** found false by a human looking, not by the system
noticing. `[L40, F28]`

**C5. Writing down a lesson is not the same as building the thing it demands.**
`[L1]` said "branch first" on day one. Eighteen generations later every candidate branch
had been cut from the trunk's tip, so `merge-base --is-ancestor` said the whole tree was
one line — the exact degeneracy `[L1]` named, rebuilt by the person who wrote `[L1]`, in a
form that satisfied the words and none of the mechanism. `[L40]`

---

## D. Searching, when measurement is the scarce resource

**D1. Separate the parametric level from the architectural level.**
A classical optimizer beats an LLM at picking block sizes, every time. An agent's value is
one level up: grid decomposition, what lives in shared memory, the reduction strategy, the
fusion boundary. Conflating the two is why naive agentic loops plateau. A proposal that
only changes constexpr values is rejected, not because it is bad but because the cheaper
mechanism already does it. `[bench/loop.py, prompts/proposer.md]`

**D2. Select parents by clade, not by score.**
A parent is drawn by Thompson sampling over the pooled outcomes of its **entire descendant
subtree**. A mediocre candidate that spawns good children is a good parent; ranking nodes
by their own score systematically discards stepping stones. `[bench/ledger.py]`

**D3. Clade metaproductivity over a linear history is degenerate.**
On a single chain, a node's subtree is just "everything committed after it" and the
ranking measures **age, not productivity**. It requires the subtree to be *chosen*, which
requires really branching from the sampled parent — see `[C5]`. `[L1, L40]`

**D4. Clade success is counted per row, not per candidate.**
A candidate rejected on correctness can carry the highest clade score because most of its
config rows passed. Defensible — productivity is not promotion — but a live trap: a high
clade score never means "this works". `[L2]`

**D5. Price information, not just expected speedup.**
With one GPU, a rubric that ranks ideas by promise alone queues five plausible variants of
one mechanism and burns a day proving they are within noise of each other. Quality sets a
Beta prior's **mean**; novelty sets its **strength** (inverted), so a genuinely novel idea
gets a wide posterior and is occasionally drawn — no temperature, no exploration bonus to
tune. `[specs/07]`

**D6. Time-to-signal is worth as much as headroom.**
Score the cheapest experiment that would *falsify* the idea, not the cost of implementing
it. Under a single-GPU constraint a cheap disproof is worth as much as an expensive hope.
`[specs/07 A3]`

**D7. Screen cheap, confirm expensive.**
30 s over four configs spanning four regimes, verdict only, advisory log — then 112 s over
the full matrix, recorded. Measured 3.8x saving end to end. Screen results never enter the
ledger: letting partial sweeps into clade statistics would swamp the full ones. The
screen's job is to kill what is clearly bad, not to rank what is statistically tied.
`[bench/screen.py]`

**D8. Some fixes are invisible to the measurement that ranks them.**
One candidate removed a **2.25x** silent loss and measured 2.765x against its parent's
2.759x — identical — because the harness's accuracy pass always exercised the good path. A
search promoting strictly on measured score would discard it. Robustness proposals need a
bespoke falsifier and a regression test, not a screen verdict. `[L39]`

**D9. Loops add; only ablation subtracts.**
Every generation adds a mechanism and nothing ever removes one, so the frontier
accumulates dead weight that is still being paid for. Periodically fork the frontier into
one-mechanism-removed siblings and measure. Ours found L2-sized chunking had become pure
overhead. `[L17, F15]`

**D10. Bound the ceiling arithmetically before spending GPU.**
A cache optimization's ceiling is the **re-fetched** traffic, never the total — the first
read of a byte is compulsory and survives any policy. The whole weight set here is 768 KiB
against 327 MB of activation stream: 0.002%. Even a perfect cache had nothing to win, and
that bound needed no GPU at all. `[L43, F33]`

**D11. Score the mechanism, not the framing.**
Two proposals made the identical move and realized +58.3% and +56.9%; the modestly-worded
one scored 20 points lower. **A rubric that can be talked down by modest framing can be
talked up by grandiose framing** — a direct exploit for any agent scoring its own work.
Enforcing this took the backtest's rank correlation from +0.267 to +0.483. `[F20]`

**D12. Backtest the selection rule against your own history before trusting it.**
Ours found three defects — a degenerate prior at Q=1.0, the wrong scoring target, and the
framing exploit above — before spending a single GPU minute. `[F20, bench/proposals/backtest.py]`

---

## E. The audit rule, which found seven bugs the search loop found zero of

**E1. Ask, of every result: *what does this depend on that we never varied?***
Seven for seven: padding ratio, eager baseline, dtype, input scale, allocation context,
process contention, causal flag. **Not one was found by the search loop.** Budget audit
turns explicitly; the optimizer will not do this for you. `[L13, L27, L42]`

**E2. Test the setting the harness DEFAULTS to, not the one the spec implies.**
Every candidate from generation 5 to 23 hardcoded `is_causal=True` and returned **three
quarters of its output wrong** on a non-causal input — with all 177 tests green, because
every announced config is causal. The reference benchmark's own default is
`causal: bool = False`. The default is the more dangerous case, because it is what runs
when nobody passes an argument. `[L42, F32]`

**E3. "Correct because of how the harness calls it" is not correct.** `[L24]`

**E4. Invariance and equivalence tests catch disjoint bug classes.**
Sweeping the whole lineage for a single invariant found three bugs the accuracy suite
could not see. `[L25, L12, F18]`
