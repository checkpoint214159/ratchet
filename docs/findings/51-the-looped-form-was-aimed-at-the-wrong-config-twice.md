# 51 — The looped attention form was aimed at the wrong config twice, and the census is what found the right one

**Date:** 2026-08-31. **Generation:** 40. **Branch:** `cand/g40/attn-loop-census`.
**Parent:** `v38_stream_fallback` (`7cee27c`). **Candidate:** `v40_looped_attn`.
**Picks up:** the proposal finding 48 opened and declined to build.

**Numbering note:** taken as 50 on a candidate branch while `ben` is ahead. Renumber at
merge if it clashes. `00-learnings.md` and the findings README are deliberately NOT edited
here; the proposed lessons are stated at the bottom for the merge to pick up.

---

## THE CENSUS, AND THE DILUTED CEILING, STATED BEFORE A LINE OF KERNEL CODE

`bench/probes/g40_attn_loop/probe_census.py`, config 10 on `v38_stream_fallback` — the
candidate that actually ships — inside the replayed CUDA graph, 20 forwards profiled after
200 settling calls, correctness checked before anything was timed:

```
wall (median of 100, settled)              251.90 us
device time in the graph                   222.50 us   (88.3% of wall)

  _proj_gemm                          x12   69.92 us
  ampere_fp16_s1688gemm_..._relu_..   x 4   54.67        <- qkv, still cuBLAS
  _attn_single_tile                   x 4   44.14
  triton_per_fused_..._layer_norm_*   x 9   46.15
  Memcpy DtoD                         x 2    7.62

  bucket              calls   us/fwd   % device   % wall
  projection GEMM      16.0   124.59      56.0     49.5
  layernorm             9.0    46.15      20.7     18.3
  ATTENTION             4.0    44.14      19.8     17.5
  copy                  2.0     7.62       3.4      3.0
```

**Attention is 17.5% of config 10's wall**, which confirms the 17.6% finding 48 assumed
from "F-00 §1" but had never measured on this candidate. Per call: **11.04 µs**.

### The correction finding 48 did not make, and it is the reason this was worth re-doing

Finding 48 priced its proposal by multiplying that in-graph time by an op-level ratio
measured under `do_bench`. Its incumbent read **24.757 µs per call against the 11.04 µs the
model pays** — a **2.24x regime gap**. `do_bench` flushes L2 and pays a launch; the model
runs the kernel L2-hot (48 MB of L2 on this card) inside a captured graph. The looped
form's entire claimed mechanism is *latency hiding*, and latency is exactly what changes
between those two regimes. **An op-level ratio was not transferable evidence here, and
finding 48 transferred it in full** [L33].

So the ratio was re-measured in the model's own regime.
`bench/probes/g40_attn_loop/probe_regime.py` sweeps both forms over their complete legal
grid with `sdpa+repack` as a third arm, correctness-gated per arm, and times every arm
**twice — once L2-flushed, once L2-hot**. Two independent runs of the whole probe:

```
cfg 10  H=2 hd=64 S=128     incumbent    best looped     ratio (run 2 / run 1)
   flushed                  22.528 us      18.432       1.222x / 1.375x
   HOT                      11.189          9.109       1.228x / 1.226x
```

The hot ratio **replicates to 0.2%**, and the incumbent's hot time (11.189 µs)
independently reproduces the census's 11.04 µs/call. The flushed ratio moves 12% between
runs — which is its own small result: **the regime finding 48 measured in is the noisier
of the two as well as the wrong one.**

### THE DILUTED FIGURE, IN UNITS OF `weighted_score`

At the measured hot ratio, assuming it transfers into the graph in full:

| | |
|---|---|
| attention in the graph | 44.14 µs/fwd (17.5% of wall) |
| saved at 1.228x | 8.20 µs/fwd — **3.25% of config 10's wall** |
| config 10 | 2.33x → 2.408x |
| **Δ `weighted_score`** | **+0.0056** |

**Is that above the floor?** L29 puts the project's noise floor at ±7%, and finding 47
measured 7.1% between two runs of a byte-identical arm — under which 3.25% is
unresolvable, and finding 48 stopped there for exactly that reason. **But finding 49's
addendum measured the floor of the instrument this candidate is ranked with**: `bench/abba.py`
at `--warmup 200`, byte-identical arms read **52.22 µs against 52.22 µs and 6593.54 against
6593.54** — identical to the hundredth of a microsecond. Against *that* floor 3.25% is
roughly thirty times resolvable. The ±7% figure is a property of the old protocol, not of
the hardware, and treating it as a hardware constant is what closed this lever the first
time.

## FOUR THINGS THE BUILD FOUND THAT THE PLAN DID NOT

### The margin was being applied against the wrong thing, and it declined the target config

The first working chooser required a challenger to beat `min(derived_single_tile, sdpa)`
by `DECISIVE`. On config 10 that **declined the looped form** — the one config the
candidate exists for — because the same arm is 1.228x the incumbent but only **1.096x
sdpa**, and 1.096 is inside 10%.

The bug is conceptual, not arithmetic. `DECISIVE` exists to ask *"is the gain over the
status quo bigger than the noise?"*, so it belongs against **what the model runs today**.
Asking it against the best available alternative is a different and much stricter
question, and it silently converts "the incumbent holds the ground" into "the incumbent
and every alternative hold the ground". SDPA belongs in the rule as a **floor** — never
ship a Triton kernel the vendor beats — not as a second 10% margin. Caught only because
`test_config_10_selects_the_looped_form` asserts the mechanism engaged [L36]; every
correctness test passed with the kernel switched off.

### Changing the tuner's timer silently re-tuned a kernel the candidate was not about

Switching the chooser to the hot timer made it return `single_tile(16, 2, 1)` on config 2
where v38 runs `(64, 4, 1)` — because the hot timer sees that tile win by 1.124x and the
flushed timer v23 uses does not. Plausibly an improvement, and entirely beside the point:
it would have ridden into the ledger on this candidate's row, and it would have destroyed
the byte-identical control arm the ranking depends on.

The fix is structural rather than careful: `attn_choice.autotune_looped` can return
exactly one thing — "use the looped form with this tile" — and **every other outcome
raises, whereupon the candidate calls v23's `_decide_attn` unchanged**. Shapes the looped
kernel does not win are then identical to the parent by construction, not by inspection.
The single-tile re-tune is written up as a proposal instead.

### v23's tuner is deterministic, but only within a process state

The "byte-identical" control claim was first written as a test asserting v40 and v38
settle on the same TILE on config 2. It passes in a fresh process and fails when the two
builds are separated by fifty other GPU tests in the same process.

Measured directly: five independent builds of v38 in a clean process return `(64, 4, 1)`
five times on config 2, `(64, 4, 1)` five times on config 3 and `(32, 2, 1)` five times on
config 12. **`autotune_tile` is deterministic — and it is deterministic against a
comparable process state, not absolutely**, because it ranks tiles with `do_bench` at
shapes whose arms differ by less than the timer resolves, under a 10% `DECISIVE` margin
that a warm allocator or a loaded L2 can move an arm across.

So tile equality is real and is verified where it can be — `probe_which_form.py` builds
both arms adjacently in a fresh process and reports identical tiles on all ten non-looped
configs — but it is not assertable inside a long test session. The test asserts the
structural guarantee instead: that the decision came from the parent's own routine.

### The input-scale failure was inherited, and checking that took one probe

`input_scale=0.1` on config 10 fails the locked tolerance: 298 of 1048576 elements,
max_abs 3.851e-03. Asserting an absolute pass there was testing the harness, not the
candidate. `probe_input_scale.py` compares the two arms over 3 seeds x 5 scales:

```
seed  scale     v38 max_abs  failed   v40 max_abs  failed   verdict
1234    0.1       3.923e-03     323     3.923e-03     323   BOTH FAIL -- inherited
4321    0.1       3.770e-03     291     3.770e-03     307   BOTH FAIL -- inherited
   7    0.1       4.062e-03     327     4.062e-03     292   BOTH FAIL -- inherited
   (all 12 rows at scale >= 0.5: both pass)
rows where v40 fails and v38 does not: 0 of 15
```

**`max_abs` is identical to four significant figures in every row**, which says the
dominant error at that scale is not in attention at all. The tolerance was not touched;
the test was rewritten to assert a difference against the parent, which is the claim that
was actually available. This is CLAUDE.md's standing hazard behaving exactly as documented.

## THE MEASUREMENT

### Where the candidate differs at all

`bench/probes/g40_attn_loop/probe_which_form.py` builds both arms on all thirteen runnable
configs and reports the plan each settles on. **v40 differs from v38 on exactly three
configs — 4, 10 and 12 — and is byte-identical on the other ten**, including config 6, which
is 83% of the matrix's wall time and cannot pay for a regression. That is by construction,
not by luck: `autotune_looped` can only ever return the looped form, and every other
outcome falls through to v23's `_decide_attn` run unchanged.

```
cfg    B   H   hd    S   v38 plan              v40 plan                  same?
  4   16   4   32  128   single_tile(64,4,1)   looped(32,16,2,3)          no     1.153x op-level
 10   64   2   64  128   single_tile(32,4,1)   looped(64,32,4,4)          no     1.226x op-level
 12   64   4   32   32   single_tile(32,2,1)   looped(32,16,2,4)          no     1.313x op-level
  1,2,3,5,6,7,11  single_tile, identical tile                            YES
  8,9,13          sdpa, both                                             YES
```

Config 10's prime-time chooser independently selects `(64, 32, 4, 4)` at a measured 1.226x
— the same tile and the same ratio the offline symmetric sweep found (1.228x). Every
selected tile has `n_spills = 0`, read off the `CompiledKernel` (ncu is unavailable under
WSL2).

### The A/B, replicated, with byte-identical controls in the run

`bench/abba.py --rounds 5 --warmup 200`, both arms resident, cold round discarded, two
independent runs. Configs 1, 3 and 9 carry byte-identical code in both arms and are the
in-run control:

```
cfg              v38 median   v40 median     run 1     run 2      what it is
 10                 231.42       223.23    1.0367x   1.0365x    THE TARGET
  4                  87.04        87.04    1.0000x   0.9933x    engages, no effect
 12                  75.78        76.80    0.9867x   1.0000x    engages, no effect
  1                 224.26       225.28    0.9955x   1.0000x    CONTROL (identical code)
  3                  52.22        53.25    0.9808x   0.9808x    CONTROL (identical code)
  9                 224.26       225.28    0.9955x   1.0000x    CONTROL (identical code)
```

**Config 10 replicates to 0.02%** (1.0367x, 1.0365x). The controls bound the floor
directly: every control difference is zero or exactly **one 1.024 µs event-timer quantum**
— visible in the raw medians, which are all multiples of it (52.22, 223.23, 224.26,
225.28). Config 10's delta is **8.19 µs = eight quanta**, in both runs. It is outside the
floor by a factor of eight, and the floor was measured in the same run rather than quoted
from another one.

Configs 4 and 12 engage the mechanism and deliver nothing end to end, in either direction.
Config 4 also illustrates why the controls matter: its *baseline* arm read 87.04 µs in run
1 and 151.55 µs in run 2 — a 1.74x move on byte-identical reference code between runs, with
both arms moving together. Nothing about config 4 is resolvable by this instrument.

### The device census, and the contradiction it produced first

Running `probe_census.py` separately on each arm appeared to refute the whole result:

```
arm                     wall      device    attention (x4)
v38_stream_fallback   251.90 us  222.50 us   44.14   _attn_single_tile
v40_looped_attn       237.57     223.64      45.98   _attn_looped
```

v40 faster at the wall, *slower* on the device, with its attention kernel apparently 1.8
µs/fwd **worse** — while the ABBA said 1.0366x twice. Those are two different processes,
which is precisely the comparison finding 42 showed is unsafe.

`probe_census_pair.py` builds both arms in one process, settles both, and profiles them in
ABBA order. The contradiction disappears entirely and the mechanism is confirmed:

```
bucket                v38 us/fwd  v40 us/fwd     delta
attention                  40.84       33.26     -7.58     <- the whole difference
copy                        7.17        7.00     -0.17
layernorm                  42.96       43.00     +0.04
projection GEMM           115.38      115.32     -0.05

per call:  _attn_single_tile 10.211 us  ->  _attn_looped 8.315 us   = 1.228x
device total ratio 1.0391      wall ratio 1.0410
```

**Per call, in the model, the ratio is 1.228x — the op-level hot figure to three decimal
places** — and every other bucket is flat to ±0.2 µs. The saving is attributable to the
attention kernel and to nothing else. The cross-process census was an artefact, and it is
worth recording that it was a *sign-flipping* one on a candidate whose end-to-end result
had already replicated twice.

### THE VERDICT, IN UNITS OF `weighted_score`

| | |
|---|---|
| config 10, v38/v40 | **1.0366x** (1.0367 / 1.0365, replicated) |
| config 10 speedup | 2.33 → 2.4153 |
| configs 4, 12 | mechanism engages, effect not resolvable — scored as **0** |
| configs 1,2,3,5,6,7,8,9,11,13 | byte-identical to the parent — **0** |
| **Δ `weighted_score`** | **+0.0061** |

Against a ceiling of +0.0056 pre-registered from the census before the kernel was wired
in, and against finding 48's +0.0069-total estimate of which **+0.0021 was for config 9 and
is withdrawn here as a measurement error**. The candidate delivers on the one row the
census pointed at, and the two rows the census did not price contribute nothing.

The +0.0061 slightly exceeds the +0.0056 ceiling because the ceiling divided by the census
process's wall (251.90 µs) while the ABBA measures 231.42 µs. **The absolute saving agrees
almost exactly** — 8.20 µs predicted, 8.19 µs measured end to end, 7.58 µs of it visible in
the paired device census — which is the agreement that matters, the percentage having two
different denominators.

### What it costs

The prime-time sweep is 5–48 legal looped tiles plus the single-tile grid plus sdpa, each
compiled, correctness-checked and timed. First-forward time rises to **14–67 s per config**
(config 1: 60.9 s, config 10: 41.9 s). That is construction, settled before any timing in
both the ABBA and the graded harness — but it is real, it is not free, and finding 45 is
the standing warning that `run_matrix`'s isolated arm will misreport a candidate that does
this much work at build time. **Rank this candidate on the interleaved arm only.**



## TWO RESULTS NOBODY ASKED FOR

### 1. Config 9 is closed, and finding 48's +0.0021 for it is WITHDRAWN

Finding 48 reported the looped form at **1.124x over SDPA on config 9** and priced it at
+0.0021. Re-measured here it is **0.955x flushed and 0.826x hot**, both replicated across
two runs. It loses, and it loses badly in the regime that counts.

The disagreement has a cause, and it is not noise. Finding 48's config-9 winner was
`BM=128`. At `B*H = 64` that is **64 CTAs on 66 SMs — one block per SM, one wave**, which
is precisely the configuration this candidate's predicate declines, because a loop with no
second wave to overlap has nothing to hide its memory latency behind. With that arm
excluded the best remaining looped tile is slower than the vendor. Finding 48 saw the same
grid arithmetic — it printed the wave table and wrote *"one wave, nothing to hide behind"* —
and then priced the arm anyway, because the arm was the fastest thing its sweep had found.

**The mechanism argument and the selected arm contradicted each other, and the sweep won.**

### 2. The incumbent single-tile kernel LOSES to the vendor on config 10

In the hot regime, `sdpa+repack` reads **9.987 µs against the incumbent's 11.189** — the
kernel this project ships on config 10 is ~12% slower than the vendor call it replaced.

This was pre-registered. `bench/kernels/attn_single_tile.py` carries an OPEN QUESTION
written into the source at generation 23:

> "The screen measured config 10 (head_dim 64) at -7.1% end to end — the marginal case,
> sitting at exactly MIN_RESIDENT_BLOCKS, one pass, inside the ±7% floor… It is
> deliberately NOT implemented until a full sweep confirms the regression is real."

This is that sweep, and **the pre-registered suspicion was right**. The fix is not to raise
`MIN_RESIDENT_BLOCKS` — that file's own comment explains why it cannot be (head_dim 32 is a
1.55x win at the same 4.9 blocks/SM). The fix is to stop assuming: `attn_choice` sweeps
`sdpa+repack` as a timed arm, so the shape now gets whichever of **three** options is
fastest instead of whichever of two.

## WHAT WAS BUILT

* `bench/kernels/attn_looped.py` — the looped kernel as a real source file, with the
  predicate expressed as `B * heads * cdiv(S, BM)` against the measured
  `multi_processor_count` (CLAUDE.md rule 2: shapes and measured device properties, never
  a config id), plus two **mechanism constraints on the sweep grid** stated with their
  pre-registered cost:
  * `block_n < block_m` — the loop must run more than one trip. This is **finding 47
    written down as a constraint**: F-03 was priced at +0.0138 on arms whose grid-stride
    loop ran exactly once, and nobody checked.
  * `num_stages >= 2` — pipelining is the whole mechanism; `num_stages=1` disables it.
* `bench/kernels/attn_choice.py` — the symmetric chooser. Both forms **and** `sdpa+repack`
  swept over their full legal grids, one timer, one repeat count, correctness gate per arm,
  spilling arms dropped before timing, and the incumbent holding the ground unless beaten
  by v23's existing `DECISIVE` 10% margin. The probe allocation is bounded by a fraction of
  measured `total_memory`, so seq_len 100000 (a 9.8 GiB probe tensor) declines the sweep
  instead of OOMing the model it is tuning.
* `bench/candidates/v40_looped_attn.py` — overrides `_decide_attn` and `_attention` only.
* **A refactor of three shipped ancestors.** The
  `if self.attn_used: single_tile_attention(...) else: <sdpa + repack>` block existed
  inline in four places — v23's `_core`, v34's `_core`, and both branches of v36's. It moved
  unchanged into `CandidateV23._attention` and the four sites became
  `self._attention(qkv, a, b, s)`. The alternative was copying three long `_core` bodies
  into the new candidate and keeping them in sync forever, which is the [L14] shape
  exactly. **The claim that the refactor is behaviour-preserving is load-bearing** — v38 is
  the ABBA control arm — so `tests/bench/test_attn_extension_point.py` asserts the method
  reproduces the inline expression on both branches for all four candidates, rather than
  assuming it.

## DISPOSITION, AND TWO PROPOSALS THIS DELIBERATELY DID NOT TAKE

* **`v40_looped_attn` is a candidate**, +0.0061 of `weighted_score` on config 10,
  replicated, with an in-run control and a device census that attributes the saving to the
  kernel it claims. Rank it on the interleaved arm (finding 45).
* **Finding 48's config-9 row (+0.0021) is withdrawn** as a measurement error. Its arm
  was excluded here by the one-wave predicate, and re-measured it is 0.826x.
* **PROPOSAL: sdpa+repack beats our own single-tile kernel on config 10 hot** (9.987 µs
  against 11.189). Not acted on here, because switching a config to the vendor is a
  different change and bundling it would have made this A/B unattributable. It is also now
  moot on config 10 specifically — the looped form beats both — but the same question
  applies wherever `attn_single_tile` is at `MIN_RESIDENT_BLOCKS`, and finding 31
  pre-registered it.
* **PROPOSAL: re-tune the single-tile form with the hot timer.** Doing so returns
  `(16, 2, 1)` on config 2 where v38 runs `(64, 4, 1)`, a 1.124x op-level difference.
  Config 2 is a scoring row. This was deliberately backed out of v40 to protect the
  control arm; it should be measured on its own.
* **Configs 4 and 12 are open, not closed.** The looped form is selected there on op-level
  margins of 1.153x and 1.313x and produces nothing measurable end to end. That is either
  dilution or an instrument limit — config 4's own baseline moved 1.74x between two runs —
  and it would need a census apiece to say which.

## PROPOSED LESSONS

Not appended to `docs/findings/00-learnings.md` — `ben` is ahead and it would collide.

### L57 — An op-level ratio is evidence only in the cache regime you will spend it in

Finding 48 measured attention at 24.757 µs under `do_bench` and priced a saving against an
in-graph cost of 11.04 µs — a 2.24x regime gap, in a mechanism whose entire claim is
*latency hiding*, where the regime difference IS whether the latency exists. The ratio
happened to survive re-measurement here (1.200x flushed → 1.228x hot), and that is luck,
not method: on config 9 the two regimes disagree by 16% and change the verdict's sign
relative to what was published.

`do_bench` flushes L2 and pays a launch; `do_bench_cudagraph` does neither. CLAUDE.md
already forbids comparing one to the other. **The unstated corollary is that a ratio
computed inside one of them may not be multiplied by a time measured inside the other** —
which is what pricing a census with an op-level sweep does every time. Time both arms in
the regime the model runs, or say in the write-up that you did not.

### L58 — When the mechanism argument and the swept winner disagree, the sweep is the thing that is wrong

Finding 48 printed a wave table saying config 9 had "one wave, nothing to hide behind" and
then priced config 9 at +0.0021 on a `BM=128` arm — the arm that *creates* the one-wave
grid. The sweep found it because a sweep will always return its fastest row; the mechanism
said it could not be right; the sweep was believed.

A sweep has no theory and cannot decline. That is what makes it useful for picking a tile
and useless for deciding whether a lever exists. **Where the two disagree, the sweep is
reporting a measurement of something — usually noise, here a 2.1x-unstable baseline — and
the mechanism is the only thing in the room that can say which.** The cost of getting this
backwards was a published +0.0021 that measures −17% when re-run.

### L59 — A device census across two processes can flip the sign of the thing it is auditing

Profiling v38 in one process and v40 in another said the candidate's attention kernel got
1.8 µs/fwd **slower** while its wall got 14.3 µs faster — against an end-to-end A/B that
had already replicated at 1.0366x twice. Profiled in ONE process, ABBA-interleaved, the
same two kernels read 10.211 µs/call and 8.315 µs/call and every other bucket is flat to
±0.2 µs.

Finding 42 established that a cross-run *wall* comparison is unsafe on sub-millisecond
configs. This is the same defect one level down, and it is easier to miss, because a
per-kernel device time *looks* like a physical property of the kernel rather than a
measurement of a run. **It is a measurement of a run.** Cross-process, its gap term absorbs
everything the host was doing, and the device totals it is composed from drift with it.

The rule is the one the A/B already follows and the census had not been made to: **both
arms, one process, interleaved, cold round discarded — for the profile as much as for the
timing.** A census that cannot be differenced is a description, not evidence.