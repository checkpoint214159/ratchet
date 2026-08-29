# Spec 07 — The proposal rubric: scoring ideas into a sampling distribution

Status: proposed 2026-08-29. Consumed by the proposer agent and by the controller's
queue. Companion to spec 03 (search loop) and spec 06 (scout).

## What this is for

The parametric search (spec 03, `bench/loop.py`) turns knobs. This rubric governs the
level above it: an agent that **proposes architectures**, scores them, and emits a
probability distribution the expander agents sample from.

The design constraint that shapes everything here: we have ONE GPU and a ±7% noise floor
(L29). Measurement, not ideation, is the scarce resource. A rubric that merely ranks
ideas by promise will queue five plausible-sounding variants of the same mechanism and
burn a day proving they are all within noise of each other. The rubric must therefore
price **information**, not just expected speedup.

## Two axes, and why they are combined the way they are

Every idea gets a **quality** score Q and an **entropy** score E, each the mean of five
0-5 dimensions, normalized to [0,1].

The naive combination is a weighted sum, `w_q*Q + w_e*E`, softmaxed. We do NOT do that.
A weighted sum makes "interesting" and "promising" substitutes -- a sufficiently weird
idea outranks a well-evidenced one, and the weight `w_e` becomes a magic number nobody
can defend.

Instead the two axes play **different structural roles**, mapped onto a Beta prior:

    mu    = Q                                  quality sets the prior MEAN
    kappa = KAPPA_MAX - E*(KAPPA_MAX-KAPPA_MIN)  entropy sets the prior STRENGTH (inverted)

    alpha_0 = mu * kappa
    beta_0  = (1 - mu) * kappa

A high-quality, low-entropy idea (an obvious next step, well evidenced) gets a **narrow**
posterior centred high: it will be sampled early and, if it disappoints, abandoned fast.
A high-entropy idea gets a **wide** posterior regardless of where its mean sits: Thompson
sampling will occasionally draw it even when its mean is mediocre, which is exactly the
behaviour we want from a genuinely novel proposal, and it is achieved without a
temperature parameter or an exploration bonus to tune.

Suggested constants, to be recalibrated by the backtest below:

    KAPPA_MAX = 20.0   (a low-entropy idea is worth ~20 pseudo-observations of prior)
    KAPPA_MIN = 2.0    (a high-entropy idea is worth ~2: almost uninformed)

As real measurements land, the posterior updates conventionally:

    alpha = alpha_0 + wins,  beta = beta_0 + losses

where a "win" is a screened candidate that beat its parent by more than the noise floor.
This is the same Beta-Thompson machinery `bench.ledger.sample_parent` already uses for
node selection, so the two compose cleanly:

    parent = sample_parent(ledger)     # WHICH node to expand   (CMP over git ancestry)
    idea   = sample_idea(proposals)    # WHAT to try there      (this rubric)

Node choice and idea choice are drawn independently and paired. An idea whose
preconditions are incompatible with the drawn parent is re-drawn, and that rejection is
logged -- a proposal that keeps getting rejected is mis-specified, and we want to see it.

## GATE — evaluated before scoring; any failure discards the idea

These are not dimensions. They are pass/fail, and a proposal that fails one is not
scored, not queued, and not counted against the proposer's yield.

  G1  Does not widen or reinterpret the locked tolerances (atol 2e-3 / rtol 2e-2).
  G2  Does not branch on a config id, a shape literal, or anything else that would be
      benchmark special-casing. Dispatch predicates must be functions of MEASURED device
      properties (see v14_dispatch, spec 04).
  G3  Does not modify Zone A (`ratchet/oracle/`) or the reference implementation.
  G4  Does not write to `research/` (FG-01 gated).
  G5  Runs on sm_89 as actually configured: mma.sync only (no wgmma/TMA), 99 KB opt-in
      smem, Python 3.10, torch 2.8.0, triton 3.4.0, clocks NOT lockable.
  G6  Preserves correctness-before-timing: the idea has a stated correctness argument,
      not merely a hope that the tolerance absorbs it.

## AXIS A — QUALITY (expected value). Five dimensions, 0-5 each.

**A1. Mechanism specificity.** Does the proposal name a mechanism, or a wish?
  0  "make attention faster"
  3  names a specific op and a specific inefficiency in it
  5  names the op, the inefficiency, the hardware reason it exists, and what changes
     EVIDENCE REQUIRED: a profile line, a ledger row, or a source citation.

**A2. Headroom, roofline-grounded.** How much is actually available if it works?
  Must cite a number from the device table (613.7 GB/s, 88.2 BF16-TFLOP/s FP32-acc,
  ridge 144 FLOP/B, 2.22 us launch, 48 MB L2, 66 SMs) or from `bench/results.jsonl`.
  0  no quantified ceiling
  3  a ceiling derived from a measured number
  5  a ceiling AND an argument for what fraction of it is reachable
     An idea whose ceiling is below the noise floor scores 0 here regardless of elegance.

**A3. Time-to-signal.** How fast can this be KILLED?
  Scored on the cheapest experiment that would falsify it, not on full implementation.
  5  falsifiable by the 4-config screen (~40 s GPU)
  3  needs a full sweep (~2.3 min) 
  1  needs repeats or a new harness to say anything
  0  not falsifiable by measurement at all
     This dimension is deliberately weighted equal to headroom. Under a single-GPU
     constraint a cheap disproof is worth as much as an expensive hope.

**A4. Feasibility on THIS device, at THIS effort.** 
  0  requires hardware we do not have (Hopper TMA, wgmma) or >1 day of work
  3  a day's careful work with known tools
  5  implementable and testable in one expander session

**A5. Stacking with the frontier.** Does it compose with v13, or replace it?
  0  mutually exclusive with the frontier and worse
  3  a replacement that must win outright to matter
  5  orthogonal -- if it works it ADDS to 2.711x rather than competing with it

## AXIS B — ENTROPY / INTERESTINGNESS. Five dimensions, 0-5 each.

Entropy here means **information the tree does not already contain**. It is not weirdness.
An idea that is bizarre and teaches nothing scores LOW.

**B1. Mechanism-distance from the measured archive.** Distance from all 14 candidates in
  `bench/candidates/`, measured by MECHANISM, not by code.
  0  a knob-turn on an existing candidate (that is spec 03's job, not this one)
  3  a new mechanism in an area already touched (e.g. a different precision placement)
  5  a mechanism no candidate has used at all (e.g. a hand-written kernel, a different
     algorithm for the same math, a change in memory layout)

**B2. Information gain if it FAILS.** The dimension most likely to be under-valued.
  0  failure teaches nothing; we would not know why
  3  failure narrows one parameter
  5  failure CLOSES A REGION of the search space -- a negative result that tells the tree
     not to send further agents down a whole class of approach
     Precedent: v5_fp16_resid failed on 12/14 configs and produced finding 08 (the fp32
     residual is load-bearing), which redirected three subsequent generations. That
     failure was worth more than most successes.

**B3. Source diversity.** Where did this come from, and have we mined that vein?
  0  restates something already in `docs/findings/`
  2  a source type we have used before (a paper we already cite)
  4  a source type we have NOT mined -- an issue tracker, a vendor changelog, a
     practitioner blog, a conference talk, a kernel author's thread
  5  as above AND cross-corroborated by a second independent source
     CITATION REQUIRED: URL or precise reference. An uncited claim scores 0 here.

**B4. Regime coverage.** Does it target an under-explored region of the matrix?
  Current coverage is lopsided. Under-explored, in order:
  +5  head_dim = 8 (configs 7, 11) -- the identified spot where vendor fast paths may
      refuse and a hand-written kernel could genuinely earn its keep. NEVER INVESTIGATED.
  +5  config 14 feasibility (the only shape we cannot run)
  +3  long context (13) and throughput (6) -- attention-dominated, bandwidth-bound
  +1  the mainstream middle, which is already well covered
   0  no particular regime

**B5. Kernel-level depth.** Explicitly rewards leaving the PyTorch-composition plateau.
  Every result to date -- all 2.711x of it -- was won by ARRANGING existing kernels
  (SDPA, Inductor, CUDA graphs) so fast paths qualify. Zero hand-written kernels exist.
  That plateau is why generations 11-14 produced no movement outside noise.
  0  pure PyTorch composition (the exhausted level)
  3  a compiler-directed change (custom Inductor pass, explicit fusion boundary)
  5  a hand-written Triton kernel, a new memory layout, or an algorithmic change to the
     math itself

## Combining, sampling, and honesty

    Q = mean(A1..A5) / 5
    E = mean(B1..B5) / 5
    -> Beta prior as above -> Thompson draw -> queue position

The proposer emits a ranked list with **every dimension scored separately and every
evidence citation attached**. It does NOT emit a single number. The controller can then
audit any score, and a dimension scored without its required evidence is reset to 0 by
the controller rather than argued about.

## Anti-gaming, because the proposer scores its own work

1. **Evidence or zero.** A1, A2 and B3 have mandatory evidence. Missing evidence is not
   a low score, it is a zero.
2. **BACKTEST BEFORE TRUSTING.** Score the 14 already-measured candidates from their
   pre-measurement descriptions alone. A valid rubric must rank the g9 fork
   (2.678x, the largest real jump) and v12 (2.712x) ABOVE v5_fp16_resid (failed 12/14)
   and v7_fused_norm (bought nothing, finding 10). If it does not, the rubric is wrong
   and gets revised BEFORE it schedules a single GPU minute. This is the same discipline
   applied to the objective function in finding 12, which had silently saturated.
3. **Calibration drift.** The controller tracks realized-vs-predicted per proposal. A
   proposer whose A2 headroom estimates are systematically optimistic gets its scores
   shrunk toward the prior, and that shrinkage is recorded.
4. **Diversity floor.** No more than 2 of any 5 queued ideas may share a B1 mechanism
   class. Enforced at the queue, not by the scorer, because a scorer asked to be diverse
   will claim diversity.

## What gets written down

Each proposal is a file in `bench/proposals/NNN-slug.md` carrying the full dimension
breakdown, the citations, the gate results, the Beta parameters, and -- after evaluation
-- the realized outcome. Proposals are append-only and are NEVER deleted when they fail:
a rejected or failed proposal is the evidence that stops a future agent re-proposing it,
and it is the dataset the calibration in (3) runs on.
