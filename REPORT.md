# Ratchet: Evolutionary Kernel Optimization with an Honest Measurement Apparatus

*Devpost project description. All numbers in this document are reproducible from the
append-only ledger (`bench/results.jsonl`) at the commit that ships this file; the live
scoreboard is `python3 bench/ledger.py`.*

## 1. Summary

Ratchet is a multi-agent system that autonomously optimizes transformer inference on a
single GPU. Given the competition's pinned evaluator, a four-layer transformer with a
locked correctness rule (absolute-or-relative tolerance, atol 2e-3 / rtol 2e-2) and 14
announced configurations, it runs an evolutionary search in which candidates are git
commits, lineage is git ancestry, parent selection is Thompson sampling over clade
metaproductivity in the sense of the Huxley-Godel Machine, and every measurement ever
taken is preserved in an append-only ledger keyed to the commit that produced it.

At the time of writing the frontier candidate reaches a **3.245x geometric-mean speedup
over the `torch.compile` + TF32 baseline** across the announced matrix, passing 13 of 14
configurations. The fourteenth is the configuration on which the *reference itself* is
infeasible on any hardware (its attention score matrix is 18.63 TiB); our candidate
computes it in 6.4 GiB and verifies the output against an exact float64 oracle at the
full sequence length. The search has produced 35 candidates across 34 generations, 628
ledger rows, 40 numbered findings and 48 standing "learnings" that alter how the loop
itself operates.

The claim we consider more important than the speedup: every number above survives its
own audit trail. Roughly half of the project's effort went into the measurement
apparatus, and the findings catalogue records five separate occasions on which that
apparatus caught our own numbers being wrong before we shipped them.

## 2. The problem, and why we treated it as a search problem

The task is to accelerate a fixed transformer workload without changing its outputs
beyond a stated tolerance, scored as speedup over a baseline across a matrix of shapes
spanning five regimes: launch-bound (batch 1, tiny sequences), mainstream, throughput
(batch 10,000), long context (sequence 1,024 and 100,000), and deliberately awkward head
dimensions (head_dim 8, where vendor fast paths refuse to engage).

Single-kernel optimization folklore does not transfer across this matrix. A change that
wins the throughput regime by relieving bandwidth pressure is pure overhead in the
launch-bound regime, where the entire forward pass costs less than the CPU time of
dispatching it. The matrix therefore defines a rugged, multi-objective search space in
which mechanisms compose (a CUDA-graph capture strategy stacks with a fused kernel,
which stacks with a precision placement), evaluations are expensive (a confident verdict
on one candidate costs minutes of exclusive GPU time under a measured noise floor of
plus or minus 7 percent), and most ideas fail: in comparable kernel-tuning spaces,
68 to 78 percent of configurations fail to compile at all.

Expensive, noisy evaluations over composable mechanisms with a high failure rate is the
regime in which evolutionary search with explicit lineage bookkeeping earns its
complexity. Three properties mattered in practice:

1. **Stepping stones are preserved.** Our single most consequential failure,
   `v5_fp16_resid`, failed correctness on 12 of 14 configurations and produced the
   finding that the fp32 residual stream is load-bearing (finding 08). That negative
   result redirected three subsequent generations. A hill-climber discards such a
   candidate; a lineage-keeping search records it, and its descendants avoid the region.
2. **Recombination is real.** `v17` is a genuine merge commit: the hand-written FFN
   megakernel from one branch merged into the CUDA-graph frontier of another, gated by a
   device-derived predicate. A single-parent representation cannot express this at all.
3. **Selection can be smarter than greedy.** The best parent to expand next is often not
   the best-scoring candidate, which is the core observation behind clade
   metaproductivity (section 4).

## 3. Architecture

### 3.1 Two lanes with a hard boundary

The repository is split into two deliberately separate lanes. `research/` is a
fail-closed evidence archive: every empirical claim must pass a ratified per-vendor
qualification hierarchy, and since no code path can yet mark a backend qualified, the
archive admits only no-run and synthetic classifications. It was authored on a machine
with no GPU and is designed so that it cannot fabricate a number. `bench/` is the
empirical lane: one team machine has a CUDA device (RTX 4070 Ti SUPER, sm_89, measured
613.7 GB/s bandwidth, 2.22 us launch overhead, 48 MB L2), and all measurements happen
there, plainly labelled as working measurements rather than ratified evidence. Nothing
is promoted from `bench/` into `research/archive/` implicitly. This separation is our
answer to a failure mode we consider endemic in published agentic-optimization results:
the same process that wants a number to be large is allowed to produce it.

### 3.2 Three zones inside the empirical lane

- **The oracle is immutable.** The reference implementation, tolerance logic and input
  generation are SHA-256 manifested; `scripts/check-oracle.sh` gates every session. If a
  kernel passes only after an oracle change, the kernel is wrong.
- **The workspace evolves.** Candidates, kernels, dispatch logic and the search itself.
- **The ledger is append-only.** `bench/results.jsonl` is opened append-only, fsynced
  per row, never edited, sorted or pruned. Each row is keyed to `(commit_sha,
  config_id)` and carries its full method metadata (samples, reduction, interleaving,
  clock-lock status, exclusive-GPU flag), so any number can be re-derived or challenged
  later. Rows from a dirty working tree are recorded but barred from selection
  statistics, because a sha that does not describe the code that ran is a false
  provenance claim.

### 3.3 The agent topology

The loop runs as one orchestrator and three subordinate roles, all Claude agents
operating under role prompts checked into the repository (`docs/loop/roles/`):

- **The orchestrator** owns the GPU lock, the ledger, all merges, and scheduling. It is
  the only process that measures.
- **Researchers** (2 to 4, each with a disjoint territory) read papers, kernel
  repositories, issue trackers and vendor changelogs, and emit scored proposals with
  resolving citations. They never touch the GPU: an early sweep was corrupted by a
  researcher benchmarking Triton kernels concurrently, and the contamination was not
  re-derivable after the fact (finding 26).
- **Expanders** (2 to 5) each implement one candidate in an isolated git worktree,
  branched from the sampled parent's commit, and hand back a branch with the candidate,
  its registry entry and its tests. They do not measure and do not merge.
- **A verifier** attacks claims adversarially: can this check fail, was its subject ever
  built, is there a positive control, is this structural claim enforced by anything
  executable. This role was added after four expensive errors in three days shared the
  same shape, an assurance nobody had arranged to be capable of failing.

The topology is forced by the hardware. Two processes on one GPU do not produce two
independent measurements; they produce two wrong ones (a co-resident model once inflated
a baseline 4.1x through a host-memory spill, finding 05). Measurement is therefore
serialized through a cooperative lock (`bench/gpu_lock.py`), and the queue, not
ideation, is the system bottleneck: five expanders generate candidates faster than one
GPU can confirm them.

## 4. The evolutionary machinery

### 4.1 Clade metaproductivity, after the Huxley-Godel Machine

The Darwin Godel Machine line of work selects which agent variant to expand by its own
benchmark score. The Huxley-Godel Machine's correction, which we adopt, is that a node's
own score is a poor estimate of its value *as an ancestor*: what predicts future payoff
is the pooled outcome of the node's entire descendant subtree, its clade
metaproductivity (CMP). A mediocre candidate that spawns good children is a good parent,
and ranking nodes by their own performance systematically discards exactly the stepping
stones that evolutionary search exists to exploit.

Because our candidates are commits, CMP is cheap: a clade is forward reachability in the
commit graph, computed once from `git rev-list --parents`. Each node carries a
Beta(1 + successes, 1 + failures) posterior pooled over its descendants' ledger rows,
and the next parent is drawn by Thompson sampling, which spends most draws on productive
clades while still occasionally expanding an unpromising one, with no exploration
temperature to tune.

Two degeneracies of this scheme found us before we found them, and both are now
enforced by executable checks:

- **CMP over a linear history measures age, not productivity** (learning L1): on a
  single chain, every node's clade is simply "everything committed after it". The
  premise requires the subtree to be *chosen*, which requires real branching.
- **Writing the rule down did not implement it** (learning L40): for eighteen
  generations every candidate branch was cut from the trunk's tip rather than from its
  sampled parent, so `git merge-base` proved the "tree" was a chain even while the
  branch names looked like a phylogeny. The fix reads parents from the candidate
  registry's declared lineage, and `tests/bench/test_lineage_topology.py` now asserts
  that each candidate's ancestry equals its declaration. Every structural claim in the
  project that had such a check has never silently broken; both claims that lived only
  in prose were eventually found false.

### 4.2 Pricing ideas: quality as prior mean, novelty as prior strength

With one GPU, an idea queue ranked purely by expected speedup fails in a specific way:
it queues five plausible variants of the same mechanism and burns a day proving they are
within noise of each other. Our proposal rubric (spec 07) therefore scores each idea on
two axes of five dimensions each: quality (mechanism specificity, roofline-grounded
headroom, time-to-signal, feasibility on this device, stacking with the frontier) and
entropy, meaning information the tree does not already contain (mechanism distance from
the measured archive, information gained if it *fails*, source diversity, regime
coverage, kernel-level depth).

The two axes are deliberately not combined as a weighted sum, which would make
"interesting" and "promising" substitutes. Instead quality sets the *mean* of a Beta
prior and entropy sets its *strength*, inverted: a well-evidenced obvious idea gets a
narrow posterior centred high (sampled early, abandoned fast if it disappoints), while a
genuinely novel idea gets a wide posterior that Thompson sampling occasionally draws even
at a mediocre mean. Idea choice and parent choice then compose: both are Beta-Thompson
draws, paired per turn.

The rubric was backtested against our own history before it was trusted (finding 20),
which exposed three defects at zero GPU cost: a degenerate prior at the score boundary;
scoring against the cumulative ledger number, which measures a candidate's inherited
stack rather than its contribution (rank correlation +0.05, i.e. none; scoring marginal
gain over parent raised it to +0.48); and a framing exploit, where the same mechanism
described modestly scored 20 points below itself described ambitiously. The rubric now
scores mechanisms, not descriptions, demands a citation or scores zero, and enforces
diversity at the queue rather than trusting the scorer to claim it.

### 4.3 Two levels of search

The LLM agents are restricted to architectural moves: grid decomposition, what lives in
shared memory, reduction strategy, fusion boundaries, precision placement. Parametric
knobs (block sizes, warps, stages, chunk ratios) belong to a classical local search
(`bench/loop.py`) with an explicit noise-floor promotion gate, because a classical
optimizer picks constants better and cheaper than a language model, and conflating the
two levels is, in our reading, why naive agentic loops plateau. A proposal whose only
content is a constant change is rejected on those grounds alone.

## 5. Measurement methodology

### 5.1 The non-negotiables

Correctness runs before timing, in the same process, and a candidate that fails is never
timed. Tolerances are the evaluator's own CLI defaults and are locked; the project has a
standing rule that any tolerance reinterpretation requires explicit human sign-off.
Each configuration runs in its own subprocess, so an OOM or a wedged CUDA context costs
one row rather than the sweep, and an OOM is recorded as a result, since "the reference
cannot run this shape" is among the most informative facts the matrix yields. Clocks are
not lockable under WSL2, so timing is minimum-of-medians with per-row method metadata.
The speedup denominator is the strongest baseline anyone would actually run:
`torch.compile` with TF32, not eager. Correcting to that baseline mid-project deflated
our headline from 7.2x to 1.69x and flipped two configurations from win to loss
(finding 12); both numbers are in the ledger, and quoting the flattering one alone is
precisely the artifact this project was built to avoid.

### 5.2 Economics: screen, then confirm

Under a plus-or-minus 7 percent noise floor (measured from accidental replicates, not
assumed), a confident full-matrix verdict costs minutes; ideas arrive faster than that.
Evaluation is therefore two-stage: a 30-second screen over four configurations chosen to
span four regimes (the subset was itself derived from 411 existing ledger rows at zero
GPU cost), followed by a full recorded sweep only for survivors, a measured 3.8x saving.
Screen results are advisory and never enter the ledger, because partial sweeps would
swamp the clade statistics that full sweeps feed.

### 5.3 Measuring the measurer

Three episodes shaped our view that the harness deserves as much scrutiny as the
kernels.

**Isolation inverts effects.** The same change measured 3.84x better in an op-level
probe, 16.2 percent worse in a model-level probe, and flat end to end (finding 29). Both
wrong numbers came from hazards already documented in this repository: the op-level
probe compared against a baseline the compiler never actually runs, and the model-level
probe co-located both arms in one process. The standing rule since: a number that will
change a decision comes through the harness, and a probe that disagrees with the harness
is wrong until proven otherwise.

**A guard is only evidence if it can fire.** The contention detector built on
`nvidia-smi` reported a live CUDA process on one trial and nothing on an identical trial
seven seconds later; under WSL2 a clean report from it means nothing (finding 26). Every
guard added since is tested against a condition we deliberately create. The companion
technique is the positive control: our null result on L2 cache persistence (finding 33)
is credible because a deliberately evicting control arm moved the same kernel by
42.7 percent, establishing that the apparatus could see the effect whose absence we
report. The persistence window itself changed performance by 0.25 percent, on arithmetic
grounds we should have computed first: the entire weight set is 768 KiB against a
327 MB activation stream, so even a perfect cache had nothing to win.

**Measure with the protocol that scores you.** Late in the project we established that
our harness and the graded evaluator disagree in a specific, mechanistic way
(finding 42). The evaluator interleaves baseline and candidate in alternating rounds
precisely to cancel thermal and clock-order bias; our harness timed the baseline, then
compiled and autotuned the candidate (work that heats the GPU), then timed the candidate
on a hotter device with no clock lock. On sub-millisecond configurations this inverted
the sign of two comparisons (configurations 1 and 9), and both inversions were
predicted in advance from a kernel census: the newer candidate launches strictly fewer
kernels on those shapes, so a measured regression there was mechanically implausible.
Under the evaluator's own protocol, run unmodified via `bench/end_to_end.py` with the
candidate patched into its designated seam, the frontier dominates its parent on every
configuration tested. Repeating that check three times then exposed a second-order
fact: the evaluator's *baseline* arm varies by up to 39 percent run to run on the
smallest configurations while the optimized arm is stable to the fourth digit, so even
the graded per-run ratio cannot rank two of our candidates. The standing rule that
resulted: rank candidates by their optimized time against a fixed reference measurement,
never by a per-run speedup ratio, and note that the official score is unaffected because
the graders compute one ratio per submission rather than comparing two of ours. The
general lesson is recorded as learning L52: when your harness differs from the one that
scores you, the burden is on you to demonstrate agreement per configuration before
trusting any ranking.

## 6. What the search found

The frontier's progression, all figures geometric-mean speedup over the compiled
baseline across the announced matrix:

| generation | candidate | mechanism | geomean |
|---|---|---|---|
| 1 | v1_fused_graph | fused QKV, fp16 GEMM cache, static CUDA graph | 0.79x |
| 2 | v2_fp16_flash | fp16 q/k/v, mask elision so FlashAttention qualifies | 1.41x |
| 6 | v6_fp16_gelu | one non-accumulating fp16 round-trip removed | 1.69x |
| 9 | v9a_compiled_core | our algorithm handed to Inductor for fusion | 2.68x |
| 12 | v12_graph_over_compile | compile for fusion, then capture in our own graph | 2.71x |
| 17 | v17_dispatched_megakernel | recombination: hand-written FFN kernel merged in | 2.76x |
| 18 | v18_capture_insurance | capture no longer depends on caller's allocation context | 2.77x |
| 23 | v23_single_tile_attn | hand-written single-tile attention where scores fit on chip | 3.02x |
| 26 | v26_causal_correct | honours the evaluator's causal flag | 3.10x |
| 34 | v34_launch_bound | 16 of 36 kernel launches per forward eliminated | 3.25x |

Selected findings, each written up in `docs/findings/` with its method:

- **The padding blind spot** (finding 11). Every early number was measured at padding
  ratio zero, the only regime in which the mask-elision fast path engages. At padding
  0.5 the frontier retained as little as 28 percent of its speedup. The repair was a
  proof, not a tuning: for right-padded causal inputs the key mask is provably
  redundant, which restores the fast path legitimately (5.85x at padding 0.5, against
  2.86x before).
- **The fp32 residual is load-bearing** (finding 08). An fp16 residual stream is 1.4x
  faster and fails 11 of 13 configurations; the error floor is representational (seven
  configurations landed on the identical max-abs value), so no tuning escapes it. This
  single negative result shaped precision placement for every later candidate.
- **The causal default** (finding 32). Every candidate from generation 5 to 23
  hardcoded `is_causal=True` and returned three quarters of its output wrong on
  non-causal input, with all 177 tests green, because every announced configuration is
  causal. The evaluator's own default is `causal=False`. The audit rule that caught
  this ("what does this result depend on that we never varied?") is now seven for
  seven: padding ratio, baseline choice, dtype, input scale, allocation context, process
  contention, causal flag. The search loop found none of the seven; deliberate audit
  turns found them all.
- **Hand-written kernels win where structure permits and must decline elsewhere**
  (findings 25, 31). The FFN megakernel holds both weight matrices (64 KB) in the 99 KB
  opt-in shared memory and fuses GEMM into GEMM, which Inductor structurally cannot do.
  The single-tile attention kernel eliminates the online-softmax loop entirely on the
  eleven configurations whose score matrix fits on chip, and *declines* head_dim 128/256
  and the long-context shapes, where it measured 0.94x and 0.84x. Dispatch predicates
  are functions of measured device properties; branching on a configuration id is
  forbidden by rule G2 of the proposal gate, as benchmark special-casing.
- **The infeasible configuration** (finding 40). Config 14 (batch 32, sequence
  100,000) requires an 18.63 TiB attention score matrix in the reference formulation; it
  is infeasible on any hardware, not merely ours. Our streaming candidate computes it in
  6.37 GiB, and because the reference cannot produce a comparison output, correctness is
  certified against an exact float64 oracle evaluated at the full sequence length
  (max deviation 8.09e-4, three digits from the reference's own TF32 representation
  floor of 8.086e-4). No speedup is claimed for this row; feasibility with a
  certificate is the result.

## 7. Development tools, APIs, libraries, datasets

**Development tools.** VS Code with the Claude Code extension on WSL2 (Ubuntu 22.04);
Claude Code as the agent runtime for the orchestrator and all subagent roles (Opus for
orchestration and analysis, Sonnet-class agents for parallel research and implementation
work); git worktrees for agent isolation; pytest (518 tests passing at the last full run); the Beryl
control plane for deterministic repository checks and generated per-tool agent
instructions; a zero-dependency Node 18 dashboard (`dashboard/`) serving a live view of
the ledger, the declared-lineage evolution tree, per-config heatmaps and failure rows
over server-sent events.

**APIs.** The Anthropic Claude API (via Claude Code) for all agent roles; the CUDA
runtime API via ctypes for the L2 persistence probe (`cudaStreamSetAttribute` with
access-policy windows); `torch.profiler` (Kineto) for kernel censuses; NVML/nvidia-smi
for best-effort contention detection. No external web services are called by the system
under evaluation.

**Libraries and frameworks.** PyTorch 2.8.0+cu128 (`torch.compile`, TF32, CUDA graphs,
`scaled_dot_product_attention`); Triton 3.4.0 for all hand-written kernels; CUDA 12.8 on
an RTX 4070 Ti SUPER (sm_89); Python 3.10 standard library for the ledger, sampling and
statistics (the selection machinery is dependency-free by design); pytest and
pytest-testmon; Node 18 (no runtime dependencies) for the dashboard.

**Datasets and assets.** No external datasets. All inputs are generated by the
competition's pinned evaluator (`torch_transformer_benchmark.py`, SHA-256
`5529c96a...9a7f36e`) from fixed seeds; the announced 14-configuration matrix is encoded
once as executable data in `bench/matrix.py`; the device calibration record
(`ledger/device.json`) is measured locally by `ratchet.oracle.device`. The only
hand-authored corpus is our own: 628 measurement rows, 40 findings and 48 learnings,
which function as the system's long-term memory and are read by the agents at the start
of every turn.

## 8. Limitations, and what we would do with more time

- **Protocol agreement came late.** `bench/end_to_end.py` existed for days with a
  docstring stating that nobody had verified our harness against the graded one; the
  check was run once, passed on the candidates of that era, and was not re-run as
  candidates grew compilation-heavy enough to invalidate it. A full session of ranking
  effort went into a quantity that is not the score. With more time, protocol agreement
  would be a standing per-generation gate rather than a finding.
- **Sub-millisecond configurations strain the statistic.** The geometric mean weights a
  0.06 ms configuration equally with a 57 ms one, and run-to-run spread on the smallest
  shapes exceeds the margins being ranked. A duration-weighted or paired-per-round
  statistic would decide those configurations more honestly.
- **The two-lane boundary is still manual.** Promotion from working measurements into
  the ratified archive awaits the vendor qualification hierarchy; the paper-generation
  lane and the empirical lane therefore describe different evidentiary standards, which
  is correct but requires care to communicate.
- **Single device.** Every number is one card. The dispatch predicates are written
  against measured device properties rather than constants precisely so the search
  transfers, but no second device has verified that claim yet.
- **Agent memory is repository memory by policy**, and that policy has a cost: the
  findings corpus is now large enough that researchers spend meaningful context reading
  it. A retrieval layer over findings, rather than sequential reading, is the obvious
  next step.

## 9. Team contributions

- **Ben Goh**: the empirical lane end to end; the measurement harness, ledger, GPU
  lock, screen/confirm pipeline and proposal rubric; orchestration of the agent loop;
  all candidates, findings and learnings in `bench/` and `docs/findings/`.
- **Praneeth Suresh**: the fail-closed research lane (`research/`, the append-only
  archive, planning queue and paper generation); the Beryl control plane and
  deterministic check infrastructure; the vendor qualification and hardware-gating
  design in `docs/hardware-support.md`.

Both lanes share the repository's governing idea, which we offer as the project's
thesis: in agentic optimization, the scarce resource is not ideas but trustworthy
measurements, and the system that wins is the one whose numbers survive being audited.
