# 48 — Looped attention misses its own pre-registered bar on config 9, and the payoff turns out to be where the incumbent already wins

**Date:** 2026-08-30. **Generation:** 39. **Branch:** `cand/g39/persistent-ffn`.
**Parent:** `v38_stream_fallback` (`7cee27c`). **Proposal:** F-02 (sibling: E-04).
**Verdict:** F-02 **DECLINED on its own kill condition.** A new proposal is opened in its
place, pointing at a different config, and it is a *proposal* — not a candidate and not a
conclusion [L41].

**Numbering note:** taken as 48 on a candidate branch while `ben` is ahead. Renumber at
merge if it clashes.

---

## What was pre-registered

F-02 asks for a flash-style attention kernel with the K/V axis back in a loop, so the Q/K/V
operand term leaves the register file and is staged through the 99 KB of opt-in shared
memory the loop-free `attn_single_tile` never touches. It targets **configs 9 (head_dim
128), 8 (256) and 13 (seq 1024)** — the shapes where `attn_single_tile.applies()` returns
False and the frontier falls back to the vendor.

It pre-registered two things, and both are now answered:

> **KILL CONDITION.** "If the best looped tile does not reach **1.4x** on config 9's shape,
> stop."
>
> **DISCRIMINATOR.** "If the looped form wins on config 9 (grid 64 → 256) but *not* on
> config 8 (grid already 256), the cause is occupancy… If it wins on both, the cause is
> the register working set… If it wins on neither, hand-written attention above head_dim
> 32 is closed on this card."

## The falsifier

`bench/probes/g39_persistent_ffn/probe_looped_attn.py`. GPU lock held, one process, the
kernel written to a real source file. The baseline is what the frontier **actually runs**
where `attn_single_tile` declines — `qkv.split` → three `view`/`transpose` → `SDPA(is_causal
=True)` → `transpose(1,2).reshape` — copied from `v34_launch_bound._core`, not a strawman
[L33, finding 29]. Every arm's output is checked against it at the locked tolerance before
it is timed; arms that fail are dropped, not reported.

```
shape                       baseline    best looped arm                 ratio   verdict
cfg  9  H=1 hd=128 S=128     23.774 us  BM=128 BN=16 8w st=3  21.007    1.132x  FAILS 1.4x
cfg 10  H=2 hd=64  S=128     58.461     BM=128 BN=16 8w st=4  20.419    2.863x  (see below)
cfg  8  H=4 hd=256 S=128    134.606     BM=32  BN=32 4w st=3 133.814    1.006x  FAILS
cfg 13  H=4 hd=32  S=1024   309.108     BM=64  BN=64 4w st=2 268.138    1.153x  FAILS
```

`n_spills = 0` on every winning arm. The kernel is correct, compiles across the whole
tile grid on sm_89 under triton 3.4.0, and is genuinely faster than the vendor everywhere
— **it just is not faster by enough anywhere F-02 aimed it.**

**Config 9 measures 1.132x against a pre-registered bar of 1.4x. By F-02's own rule: stop.**
Config 8 is 1.006x, i.e. nothing, which independently confirms F-05's closure from a new
direction. Config 13 is 1.153x and is past the 3.0 cap, so it is worth zero.

The discriminator resolves to its third branch — *wins on neither* — and the conclusion it
pre-committed to therefore stands: **on the shapes where `attn_single_tile` declines,
hand-written attention does not pay on this card.** Finding 31's table was already saying
so; this measures the other design and gets the same answer.

## The result nobody predicted: config 10, where the incumbent already applies

Config 10 (head_dim 64) was in the probe as a control, because `attn_single_tile.applies()`
is **True** there — it is not one of the shapes F-02 targets. The looped form appeared to reach
**2.863x over SDPA+repack**, against the incumbent single-tile kernel's 2.374x. Both of
those numbers turned out to be wrong, in opposite directions.

That comparison as first measured was not fair, and it is worth saying why rather than
quietly fixing it: the looped form was swept over 180 arms and the incumbent over four,
at one warp count — and finding 31 records that config 10's best single tile is 32 rows at
**eight** warps, which that sweep never tried. **That is precisely the best-of-N-against-
best-of-1 handicap finding 47 had just measured at 4.5%,** committed by the same session
one file later. `probe_attn_cfg10_fair.py` re-runs it symmetrically: both forms swept over
their full legal grid, then the winners ABBA-interleaved head to head with the cold round
discarded, and with the incumbent's own `autotune_tile` — the routine the candidate calls
at prime time — timed as a third arm, so the comparison is against what the model *runs*
rather than against the best arm a sweep can find for it.

```
cfg 10  (incumbent swept over its full 36-arm grid; looped over its full 180)
  looped              BM=128 BN=16 8w st=4    20.623 us   n_regs=95  n_spills=0  128 CTAs
  single_tile(tuned)  autotune_tile -> (32,4,1) 24.757     <- what the MODEL runs
  single_tile(best)   BM=64 4w st=1           24.174
  sdpa+repack                                 27.469
                                              looped / autotuned incumbent = 1.200x

cfg  9  (autotune_tile: "no viable tile" -- the model runs SDPA here)
  looped              BM=128 BN=16 8w st=3    20.848 us   n_regs=169 n_spills=0   64 CTAs
  sdpa+repack                                 23.424      <- what the MODEL runs
  single_tile(best)   BM=64 8w st=1           24.709      <- legal, correct, and SLOWER
                                              looped / SDPA+repack = 1.124x
```

**The fair number holds at config 10: 1.200x, against 1.206x from the sloppy sweep.** The
handicap did not change the answer there — but it badly changed the *baseline*: the first
probe measured `sdpa+repack` at config 10 as **58.461 µs** and the settled ABBA run
measures **27.469 µs**, a 2.1x error in an unsettled single arm. So "2.863x over SDPA" was
never real; 1.332x is. Two wrong baselines in one probe, both caught only by re-running
symmetrically.

Config 9's fair ratio is **1.124x**, confirming the falsifier's 1.132x and staying far
under the bar. The run also incidentally re-confirms finding 31's predicate from the other
side: `single_tile` *does* compile and match at config 9 (24.709 µs at BM=64/8 warps) and
`autotune_tile` still declines it — correctly, because SDPA is faster at 23.424 µs.

### Why this direction is interesting even so

The mechanism F-02 identified is real; it is the *targeting* that was inverted. The looped
form's advantage is occupancy and pipelining, and both are functions of the **grid**, not
of head_dim:

```
cfg  9  B*H =  64   ->  64 CTAs at BM=128, on 66 SMs   one block per SM, nothing to hide behind
cfg 10  B*H = 128   -> 128 CTAs at BM=128              two waves; latency has somewhere to go
cfg  8  B*H = 256   -> 1024 CTAs at BM=32              grid is ample; hd=256 is register-bound instead
```

At head_dim 128 the winning tile is `BM=128`, i.e. one program per `(batch, head)` — so the
grid *is* `B*H = 64` and no tile choice raises it without making each program re-read all of
K and V. Config 9's problem is not that its kernel is loop-free; it is that `H = 1` and the
batch is 64, so there are only 64 independent pieces of work. **That is a shape fact, and
no attention kernel fixes it.**

## L33 — THE DILUTED FIGURE, IN UNITS OF `weighted_score`

Priced on the census shares (F-00 §1: attention is 16.2% of config 9 and 17.6% of config
10), assuming the op-level ratios transfer in full — which finding 29 says they do not:

| cfg | attn µs/fwd | wall | fair ratio | over what the model runs | saved | speedup | Δ weighted |
|---|---|---|---|---|---|---|---|
| 9 | 35.72 (16.2%) | 0.252 ms | **1.124x** | SDPA+repack | 3.93 µs (1.56%) | 1.816 → 1.845 | **+0.0021** |
| 10 | 42.69 (17.6%) | 0.243 ms | **1.200x** | `attn_single_tile` (autotuned) | 7.13 µs (2.93%) | 2.229 → 2.296 | **+0.0048** |
| 8 | 396.89 (6.2%) | 6.503 | 1.006x | SDPA | 2.4 µs (0.04%) | — | ~0 |
| 13 | past the 3.0 cap | — | 1.153x | SDPA | — | — | 0 |

**Total +0.0069 of `weighted_score`**, of which **+0.0048 is on a config F-02 does not
target and +0.0021 on one of the three it does.** F-02 priced itself at +0.030 optimistic
and +0.015 realistic; the two rows it was written for supply **+0.0021 between them**, a
seventh of its realistic figure. Every per-config delta (1.6%, 2.9%) is inside L29's ±7%
floor, so neither a sweep nor a screen can resolve this — it would need a census [L39].

And the ±7% floor is not a formality here: finding 47 measured **7.1% between two runs of a
byte-identical arm**, and this probe measured **a 2.1x error in one unsettled baseline**.
A +0.0069 claim resting on op-level ratios of that instrument is not a claim the ledger can
carry.

## Disposition

* **F-02 is declined on its own pre-registered bar.** It missed 1.4x on config 9 by a wide
  margin and returned 1.006x on config 8. Honouring that matters more than the marginal
  1.132x is worth: the bar existed precisely so a 1.13x could not be talked into a
  candidate after the fact, and three of this session's best outcomes were negatives.
* **E-04 (score tile → smem) inherits the closure.** F-02 and E-04 split finding 31's
  sentence in half and share an A2. This measured the operand half — the one E-04's own
  arithmetic says is 2.4x–4.8x the larger term at these head_dims — and it does not pay.
  The smaller half will not pay more.
* **A NEW proposal, at a different address, worth +0.0048:** the looped kernel as a
  *second tile shape for `attn_single_tile` where it already applies* — config 10, head_dim
  64 — predicated on the grid (`B * heads` against `multi_processor_count`) rather than on
  head_dim. It reuses v23's launcher, predicate, prime-time sweep and `DECISIVE` margin
  verbatim; the kernel exists in `probe_looped_attn.py`, matches at the locked tolerance on
  four shapes, and spills nothing. **It is a proposal, not a candidate**: +0.0048 on a
  single config at 2.9% of that config's wall is below every floor we can measure against,
  and it needs a census rather than a sweep [L39]. Whoever picks it up should also check
  head_dim 64 at other `B*heads`, since the predicate is a guess until it is swept.
* **Config 8 is confirmed closed from a second direction.** F-05 closed it on the GEMMs
  being at 100.4% of peak; this closes its remaining attention lever at 1.006x.

## PROPOSED LESSON

Not appended to `docs/findings/00-learnings.md` — `ben` is ahead and it would collide.

### L58 — A control shape is where the surprise lives, so measure it as carefully as the target

Config 10 was in this probe only as a control: `attn_single_tile` applies there, so it was
not one of the three shapes F-02 targets, and it got the sloppy end of the sweep — four
arms at one warp count against the candidate's 180. It then produced the only interesting
number in the run. **The arm you are not trying to prove anything about is the one you will
be least careful with, and it is disproportionately likely to be the one that matters** —
because the shapes a proposal targets have usually already been reasoned about, and the
control has not. Sweep every arm over the same grid from the start; the cost is compile
time, and the alternative is discovering your best result through a measurement you have to
throw away.

The same run makes the cost of *not* doing so concrete twice over: the sloppy sweep put
config 10's baseline at 58.461 µs against a settled 27.469, and reported the incumbent at
its 4-warp arm when finding 31 had already written down that 8 warps was better. **Neither
error changed the sign, and one of them was 2.1x.**
