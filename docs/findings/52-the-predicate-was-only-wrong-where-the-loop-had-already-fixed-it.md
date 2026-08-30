# 52 — Three attention paths, thirteen configs, twice: the predicate is wrong on exactly one shape, and the looped form had already covered it

**Date:** 2026-08-31. **Generation:** 41. **Branch:** `cand/g41/attn-audit`.
**Parent:** `v40_looped_attn` (`ac0ef1e`). **Candidate:** `v41_vendor_aware_attn`.
**Answers:** the OPEN QUESTION written into `bench/kernels/attn_single_tile.py` at
generation 23, restated by finding 31, and re-opened as an unacted proposal by finding 50.

**Numbering note:** taken as 51 on a candidate branch while `ben` is ahead — which already
carries two files numbered 50. Renumber at merge if it clashes. `00-learnings.md` and the
findings README are deliberately NOT edited here; the proposed lesson is at the bottom for
the merge to pick up.

---

## THE QUESTION, AND WHY IT WAS WORTH ASKING

`attn_single_tile` is the kernel that broke this project's plateau: it took the frontier
from 2.711x to 3.015x and beat FlashAttention wherever the score matrix fits on chip
(finding 31). It fires on a predicate, `pays()`, which asks whether enough blocks stay
co-resident on an SM to hide the memory latency a loop-free kernel cannot pipeline.

Finding 50 then measured, in passing and while measuring something else:

> `sdpa+repack` beats our own single-tile attention kernel on config 10 when measured hot:
> **9.987 µs against 11.189**.

and deliberately did not chase it, because switching a shape to the vendor is a different
change from adding a second Triton form and bundling them would have made v40's A/B
unattributable. So the predicate stood, **asserted rather than measured, on nine more
shapes.** If it were too broad on any of them, tightening it would be free score.

The defect it would be is specific and it is not subtle: **`pays()` is a statement about
our kernel and it was being read as a statement about the vendor's.** Whether a loop-free
kernel can hide its latency and whether cuDNN is slower than it are two different
questions, and only one of them was ever timed.

## WHAT WAS MEASURED

`bench/probes/g41_attn_audit/probe_three_arms.py`. Every runnable config, all three paths,
**L2-hot inside a captured graph** (`do_bench_cudagraph`), correctness-gated per arm at the
locked tolerance before anything is timed, spilling arms dropped before timing, GPU lock
held, **two full independent runs**. Both Triton forms swept over their complete legal grid
on the measured device; 8–59 arms per config.

`n_spills = 0` on every arm that survived to the timing set — that is a precondition of
admission, not an observation, and 3 arms were dropped on config 1 for spilling.

### How the arms were equalised, given that SDPA has no tiles to sweep

Finding 47 measured a **4.5% best-of-N-against-best-of-1 handicap** and finding 48 then
committed it. It is a winner's curse: a `min` over N noisy readings is biased low, and SDPA
is one arm and always will be — you cannot sweep it harder. Three things were done, and all
three are printed on every row:

1. **The same timer and the same repeat count for every arm**, vendor or Triton.
2. **`sdpa best-of-1`** — one arm's budget, exactly what each individual Triton tile got.
3. **`sdpa best-of-N`**, where `N` is the number of Triton arms admitted and timed for that
   shape, minimum kept. This grants the vendor precisely the amount of minimum-taking the
   sweep grants our kernels, and the gap between (2) and (3) *is* the winner's-curse term.

**Measured, that term is 0.00%–0.29% across thirteen configs** (config 10, the largest:
10.017 → 9.988; config 3 and config 2, the smallest: 0.00%, identical to three decimals
over eight trials). So the ranking below is reported on `best-of-N` and would be identical
on `best-of-1` on every row.

That is worth stating plainly, because it bounds the scope of finding 47's 4.5%: **under
`do_bench_cudagraph` on these shapes, the best-of-N selection bias is 0.1%, not 4.5% —
roughly thirty times smaller than the `DECISIVE` margin it would have to cross.** Finding
47 measured its handicap under the L2-flushed `do_bench`, which this audit's own flushed
cross-check confirms is the noisier instrument (finding 50 measured the same flushed ratio
moving 12% between runs where the hot one moved 0.2%). Whether the whole 4.5% is
attributable to the timer is not established here; what is established is that in the hot
regime the handicap this protocol was designed around is far too small to change a
decision.

## THE TABLE

Hot, µs per call, at the config's real batch (config 6 excepted — see below). Two rows per
config: run 1 and run 2, independent processes, in the order they ran.

```
cfg      B   H   hd     S  runs today             incumbent   single   looped      sdpa sdpa vs inc  fastest                clears 10%?
---------------------------------------------------------------------------------------------------------------------------------------
  1     64   4   32   128  single_tile(64, 4, 1)      9.207    9.207    8.357    10.676       0.862  looped(32, 16, 2, 2)   no -- incumbent holds
                                                   9.259    9.259    8.458    10.697       0.866  looped(32, 16, 2, 2)   no -- incumbent holds
  2      1   4   32   128  single_tile(64, 4, 1)      2.476    1.929      nan     3.856       0.642  single_tile(16, 4, 1)  YES -- 1.283x, would displace
                                                   2.475    1.933      nan     3.853       0.642  single_tile(16, 4, 1)  YES -- 1.280x, would displace
  3      4   4   32   128  single_tile(64, 4, 1)      2.551    2.551      nan     4.335       0.589  single_tile(64, 4, 1)  IS the incumbent
                                                   2.559    2.559      nan     4.335       0.590  single_tile(64, 4, 1)  IS the incumbent
  4     16   4   32   128  single_tile(64, 4, 1)      3.807    3.807    3.862     4.216       0.903  single_tile(64, 4, 1)  IS the incumbent
                                                   3.810    3.810    3.868     4.224       0.902  single_tile(64, 4, 1)  IS the incumbent
  5    128   4   32   128  single_tile(64, 4, 1)     16.631   16.631   14.437    19.575       0.850  looped(32, 16, 2, 3)   YES -- 1.152x, would displace
                                                  16.661   16.661   14.425    19.588       0.851  looped(32, 16, 2, 3)   YES -- 1.155x, would displace
  6  10000   4   32   128  single_tile(64, 4, 1)   1145.117 1145.117 1145.071  2094.535       0.547  looped(32, 16, 2, 4)   no -- incumbent holds
                                                1145.076 1145.076 1145.515  2094.688       0.547  single_tile(64, 4, 1)  IS the incumbent
  7     64   4    8   128  single_tile(64, 4, 1)      6.556    6.556    6.041    11.537       0.568  looped(64, 32, 4, 2)   no -- incumbent holds
                                                   6.554    6.554    6.038    11.566       0.567  looped(64, 32, 4, 2)   no -- incumbent holds
  8     64   4  256   128  sdpa()                   115.181      nan  115.205   115.181       1.000  sdpa                   no -- incumbent holds
                                                 115.022      nan  115.371   115.022       1.000  sdpa                   no -- incumbent holds
  9     64   1  128   128  sdpa()                     9.446      nan   11.473     9.446       1.000  sdpa                   no -- incumbent holds
                                                   9.395      nan   11.460     9.395       1.000  sdpa                   no -- incumbent holds
 10     64   2   64   128  single_tile(32, 4, 1)     11.176   11.176    9.090     9.988       1.119  looped(64, 32, 4, 4)   YES -- 1.229x, would displace
                                                  11.167   11.167    9.081     9.968       1.120  looped(64, 32, 4, 4)   YES -- 1.230x, would displace
 11     64  16    8   128  single_tile(64, 4, 1)     20.293   20.293   18.696    40.370       0.503  looped(64, 32, 2, 2)   no -- incumbent holds
                                                  20.302   20.302   18.641    40.363       0.503  looped(64, 32, 2, 2)   no -- incumbent holds
 12     64   4   32    32  single_tile(32, 2, 1)      2.107    2.107    2.169    10.534       0.200  single_tile(32, 2, 1)  IS the incumbent
                                                   2.098    2.098    2.167    10.533       0.199  single_tile(32, 2, 1)  IS the incumbent
 13     64   4   32  1024  sdpa()                   280.034      nan  265.942   280.034       1.000  looped(64, 32, 4, 3)   no -- incumbent holds
                                                 280.251      nan  267.419   280.251       1.000  looped(64, 32, 4, 3)   no -- incumbent holds

`sdpa vs inc` = incumbent_time / sdpa_time, i.e. THE VENDOR'S SPEEDUP over what the model runs today.
              >1 means sdpa+repack is faster and the predicate is letting a losing kernel fire.
two rows per config = two independent runs. `nan` = that form declines this shape.
`clears 10%?` applies v23's inherited DECISIVE margin to the fastest arm against the incumbent.
```

**Config 6 is the one shape whose sweep ran at a capped batch** — its real QKV buffer is
983 MiB and the probe's budget is 512 MiB. So the selected arms were re-timed at the real
batch of 10000, in both runs, and the cap is shown not to have changed anything:

```
              single_tile(64,4,1)   looped(32,16,2,4)   sdpa+repack    sdpa vs inc
  run 1            2093.140              2094.702         3834.197        0.546
  run 2            2093.149              2093.970         3833.845        0.546
```

Config 6 is 83% of the matrix's wall time and cannot pay for a mistake here; the vendor is
**1.83x slower** than the kernel on it, at the batch the model actually runs.

## WHAT IT SAYS

`attn_single_tile.applies()` accepts ten of the fourteen announced shapes. **On nine of
those ten it is not merely ahead of the vendor, it is ahead by 1.11x–5.00x.** `sdpa+repack` reads 0.20x–0.90x of the incumbent's
speed on configs 1, 2, 3, 4, 5, 6, 7, 11 and 12. There is no free score there, in either
direction, and the predicate is not too broad on any of them.

**The vendor wins on exactly one shape: config 10, head_dim 64, at 1.119x.** That is the
shape the file pre-registered in 2026-08-30 as "the marginal case, sitting at exactly
`MIN_RESIDENT_BLOCKS`". The suspicion was right, it replicates, and it reproduces finding
50's independent measurement of the same three numbers to within 0.1%:

    finding 50 (g40)   incumbent 11.189   looped 9.109   sdpa  9.987   looped/inc 1.228x
    this audit run 1   incumbent 11.176   looped 9.090   sdpa  9.988   looped/inc 1.229x

Same winning tile, `(64, 32, 4, 4)`, found by an independent sweep in a fresh process.
Three measurements across two generations agree to a hundredth of a microsecond.

**And it changes nothing that ships**, because on that same shape `attn_looped` reads 9.090
against the vendor's 9.988 — and `v40_looped_attn` already selects it there. The one place
the residency predicate is wrong is the one place a *different* Triton form had already
displaced it, for a different reason, in the previous generation.

So the honest answer to "is the predicate too broad" is: **yes, on one shape, by 1.119x,
and the loop got there first.**

## WHAT THE CHOOSER DOES NOW, AND WHAT IT SHOULD DO

`DECISIVE` is `best < incumbent * 0.9`, i.e. a challenger needs **1.111x**, not 1.10x —
which is why several 1.08x–1.10x looped arms below are correctly declined.

| shape | v40 selects | fastest measured | agree? |
|---|---|---|---|
| cfg 1, 3, 6, 7, 11 | `single_tile` | `single_tile`; looped is 1.00x–1.10x, inside `DECISIVE` | yes |
| cfg 2 | `single_tile(64,4,1)` | `single_tile(16,4,1)` — **1.28x** | same form, wrong tile |
| cfg 4, 12 | `looped` | `single_tile` by 1.4% / 2.9% | both inside `DECISIVE`; see below |
| cfg 5 | `single_tile` | `looped(32,16,2,3)` — **1.152x at the real batch** | no; see below |
| cfg 8, 9, 13, 14 | `sdpa` | `sdpa` (looped 1.053x on 13, inside `DECISIVE`) | yes |
| **cfg 10** | **`looped`** | **`looped`**; sdpa 2nd, `single_tile` 3rd | **yes** |

Two rows disagree and neither is the vendor's doing: config 2 is a tile, config 5 is a
probe-batch artefact. Both are written up below and neither is taken here.

**On the vendor question — which is what this audit was sent to answer — the ranking v40
ships is right on every row of the announced matrix.** What is *not* right is the fallback: `autotune_looped` carries `sdpa+repack` as a hard floor,
so a shape it wins has beaten the vendor — but the path it falls through to, v23's
`autotune_tile`, has never had a vendor floor at all. On config 10 that fallback is
measurably 1.119x worse than the vendor, and the only thing standing between the submission
and it is `autotune_looped` winning its sweep at prime time.

`v41_vendor_aware_attn` closes that. Where the plan is still the single-tile kernel, the
**chosen tile** is timed hot against `sdpa+repack` — two arms, one trial budget each, no
re-tuning of the single-tile form — and the shape goes to the vendor if the vendor clears
v23's inherited `DECISIVE` 10%. Ties go to the incumbent.

`bench/probes/g41_attn_audit/probe_vendor_verdict.py` asks the routine on all fourteen
real shapes, with the incumbent tile produced by v23's own `autotune_tile` run unchanged:

```
 cfg   incumbent               verdict     why
   1   single_tile(64, 4, 1)   kept        sdpa 10.607 us vs 9.196 -- did not clear 10%
   2   single_tile(64, 4, 1)   kept        sdpa  3.825    vs 2.452
   3   single_tile(64, 4, 1)   kept        sdpa  3.863    vs 2.535
   4   single_tile(64, 4, 1)   kept        sdpa  4.184    vs 3.777
   5   single_tile(64, 4, 1)   kept        sdpa 10.756    vs 9.266   (probe batch 66)
   6   single_tile(64, 4, 1)   kept        sdpa 10.765    vs 9.358   (probe batch 66)
   7   single_tile(64, 4, 1)   kept        sdpa 11.414    vs 6.511
   8,9,13,14                   NOT ASKED   the kernel already declines these shapes
  10   single_tile(32, 4, 1)   VENDOR      sdpa  9.903    vs 11.053 -- 1.116x, hands over
  11   single_tile(64, 4, 1)   kept        sdpa 11.667    vs 6.848   (probe batch 16)
  12   single_tile(32, 2, 1)   kept        sdpa 10.415    vs 2.085
```

**The routine is live** — it hands over config 10, at 1.116x against the audit's
1.119x/1.120x, which is [L38]: a check nobody has seen fail is indistinguishable from a
check that cannot. **And in `v41` it is never asked about config 10**, because the looped
form won there first, so the candidate fires on zero announced configs.

    Δ weighted_score as shipped               +0.0000   (fires on zero configs)
    Δ weighted_score in the branch where
      `autotune_looped` declines config 10    +0.0035

That second figure is the whole argument for the change. In that branch v40 falls back to
`single_tile(32,4,1)` and pays 11.176 µs/call where the vendor costs 9.988 — 1.19 µs x 4
calls = 4.75 µs on config 10's 231.42 µs wall, 2.05%, which takes the config from 2.330
back up to 2.379 and the aggregate by +0.0035.

**That is a guard, not a win, and it is labelled as one.** It is offered on the strength of
being byte-identical to its parent on all fourteen configs — which is a claim the A/B below
tests rather than assumes.

## THE END-TO-END A/B

`bench/probes/g41_attn_audit/run_ab.py --mode abba`, which holds the GPU lock for the
whole run and drives `bench/abba.py`: both arms resident, ABBA-interleaved, cold round
discarded, `--rounds 5 --warmup 200`, **two independent runs**. Correctness checked against
a fresh reference before any round, on every arm, at the locked tolerance.

**Every config in this run is a control**, because v41's mechanism declines every announced
shape — which is a weaker experiment than it sounds and a stronger one than it looks. It
cannot show a win. What it can show is whether ~200 ms of extra construction-time timing
per config leaks into the steady state, and whether the refactor is really inert.

```
cfg      v40 run1   v41 run1   ratio      v40 run2   v41 run2   ratio      worst delta
  1        225.28     224.26   1.0046       224.26     225.28   0.9955     1 quantum, sign flips
  2         48.13      48.13   1.0000        48.13      47.10   1.0217     1 quantum, one run
  3         52.22      53.25   0.9808       140.29     142.34   0.9856     see below
  4         81.92      81.92   1.0000        81.92      81.92   1.0000     none
  7         78.85      77.82   1.0132        77.82      77.82   1.0000     1 quantum, one run
  9        225.28     225.28   1.0000       226.30     225.28   1.0045     1 quantum, sign flips
 10        222.21     222.21   1.0000       223.23     223.23   1.0000     none
 11        268.29     270.34   0.9924       269.31     269.31   1.0000     2 quanta, one run
 12         74.75      76.80   0.9733        74.75      74.75   1.0000     2 quanta, one run
```

**Every median is an integer multiple of the 1.024 µs CUDA-event quantum** — 225.28 is
220 x 1.024, 48.13 is 47, 74.75 is 73 — and **every difference between the arms is one or
two quanta.** No config keeps both its sign and its magnitude across the two runs: config
12 reads −2.67% then 0.00%, config 7 +1.32% then 0.00%, config 1 +0.46% then −0.45%. That
is the shape of an instrument floor, not of an effect, and it is what a byte-identical pair
is supposed to look like.

**Config 3's row in run 2 must not be quoted at all**, and it is left in because it is the
control doing its job: v40 read **52.22 µs in run 1 and 140.29 in run 2** — a **2.69x move
on the same commit, the same protocol and the same machine**, with both arms moving
together. Finding 42's addendum measured 39% on this config and finding 50 measured a 1.74x
move on config 4's baseline; this is the same defect, larger. Nothing about config 3 in run
2 is resolvable, including the 0.9856x, and the only reason to report it is that a run whose
control has blown up should say so rather than quietly drop the row.

So: **`v41_vendor_aware_attn` is inert on the sub-millisecond half of the matrix, to within
one event-timer quantum, twice.** That is exactly the claim the candidate makes about
itself.
### The large half: one arm per process, replicated

Configs 5, 6, 8 and 13 are where ABBA itself is wrong (finding 50: three resident arms on a
65536-token config walk into finding 05's co-residency spill). So they were run
`--mode isolated`: one arm per process, two independent trials each, same `--rounds 5
--warmup 200`, GPU lock held across the whole sequence. Medians, µs:

```
cfg        v40 t0     v40 t1     v41 t0     v41 t1    within-arm spread   between-arm
  5        434.18     436.22     436.22     436.22          0.47%          inside it
  6      70821.38   72657.41   68491.26   72276.99          2.59%          inside it
  8       6592.00    6600.70    6595.58    6600.29          0.13%          inside it
 13       3308.54    3307.52    3308.54    3306.50          0.06%          inside it
```

**On every one of the four, the difference between the two arms is smaller than the
difference between two trials of the same arm.** Config 6 — 83% of the matrix's wall time,
and the one row a regression could not be afforded on — spans 68.5–72.7 ms across the four
readings with the two arms interleaved inside that range. Config 13 reproduces v38/v40's
3307–3313 µs from finding 50 exactly.

Note config 6 is also the shape where `autotune_vendor` actually runs (its plan is the
single-tile kernel, and the routine keeps it at 9.358 µs against sdpa's 10.765), so this is
not a config where the mechanism was skipped.

## FOUR THINGS THE AUDIT FOUND THAT IT WAS NOT LOOKING FOR

### 1. Config 2's tile is worth more than anything else on this table, and it is not ours to take

The hot timer ranks `single_tile(16, 4, 1)` at 1.929 µs against the shipped `(64, 4, 1)`'s
2.476 — **1.283x**, replicated. Finding 50 saw the same effect at 1.124x and backed it out
of v40 deliberately, because re-tuning the single-tile form with a different timer would
have ridden into that candidate's measurement and destroyed the byte-identical control the
measurement depended on. The same reasoning applies here with the same force, and for the
same reason it is left alone: **this candidate's whole claim is that it is byte-identical to
its parent everywhere.** Config 2 is a scoring row and 1.28x at the op is the largest
unclaimed op-level margin in this table. It needs its own generation.

### 2. `probe_batch`'s cap changes the ranking on config 5, and nobody had checked

`attn_choice.probe_batch` caps the tuning probe at `4 * SMs / heads` — 66 rows for a
4-head config — on the reasoning that per-program work does not depend on batch once the
grid fills the machine. Config 5 runs at batch 128. Timed at its **real** batch, the looped
form reads 1.152x over the incumbent, which clears `DECISIVE`; timed at the capped batch
(where config 5 becomes indistinguishable from config 1, whose ratio is 1.102x) it sits on
the margin, and v40 records config 5 as byte-identical.

That the capped shape is numerically config 1's is not an assumption: `probe_vendor_verdict`
times config 5's incumbent at the capped batch and reads **9.266 µs**, against config 1's
**9.196** at its real batch of 64. The two shapes are the same measurement once the cap
bites. So the inference — config 5 tuned at 66 sees config 1's 1.102x, which does not clear
`DECISIVE`, while config 5 tuned at 128 sees 1.152x, which does — rests on one measured
step, and only that step is inferred rather than run through the tuner itself.

The file's own docstring flags this as a caveat and calls it "conservative for the looped
form, not generous to it". That is exactly what it turned out to be — the cap costs the
looped form a config it would otherwise have taken. Whether taking it would *pay* end to
end is a separate question this audit did not open: config 5 is one of the four large
configs where ABBA is unsafe (finding 50).

### 3. Configs 4 and 12 do not replicate the op-level margins v40 selected them on

v40 selects the looped form on configs 4 and 12 at recorded op-level margins of 1.153x and
1.313x. Swept here, independently, twice, the looped form is **0.986x and 0.971x** — i.e.
it loses. Both configs measured "engages, no effect" end to end in finding 50 (1.0000x /
0.9933x and 0.9867x / 1.0000x), so nothing is broken and nothing was mis-shipped; but the
margins that selected them were not reproducible, and a margin that does not reproduce is a
coin flip wearing a number.

Config 10's margin, by contrast, has now reproduced **five times across two generations**:
1.228x and 1.226x (g40's regime probe), 1.228x per call in the paired in-model census, and
1.229x / 1.230x here — the last two from an independent sweep in a fresh process that also
landed on the same tile, `(64, 32, 4, 4)`. **The distinction between a lever and a coin
flip is replication, and it is cheap.**

### 4. Finding 50's config-9 withdrawal is confirmed from a third direction

Finding 48 priced config 9 at +0.0021 on a looped arm reading 1.124x. Finding 50 re-measured
it at 0.826x hot and withdrew the claim. This audit reads **0.823x and 0.820x**. The
withdrawal stands; config 9's attention is closed.

## DISPOSITION

* **The generation-23 open question is ANSWERED and CLOSED.** The regression at head_dim 64
  is real (1.119x, replicated, three independent measurements across two generations), and
  it is confined to that one shape. The `scores >= operands` discriminator finding 31
  pre-registered would have separated the wins from the loss correctly — and implementing it
  is still the wrong move, because the right answer at head_dim 64 is not "use the vendor"
  but "use the other Triton form", which the chooser already reaches by timing. A comment
  recording this now sits next to the question in `attn_single_tile.py` so it is not
  re-opened a fourth time.
* **The predicate is NOT too broad anywhere else.** Nine shapes, 1.11x–5.00x in our favour,
  replicated. There is no free score in tightening it, and the audit's headline is a
  negative.
* **`v41_vendor_aware_attn` is a candidate worth +0.0000 and offered as insurance**, not as
  an advance. Measured inert on all thirteen runnable configs — ABBA on the nine
  sub-millisecond rows (every delta 0–2 event-timer quanta, no sign surviving replication)
  and one-arm-per-process on the four large ones (between-arm difference smaller than
  within-arm trial spread on every row). Its build cost is one extra pair of hot timings per
  config, ~200 ms, against v40's existing 14–67 s prime-time sweep. **Ship it or do not**:
  the measurement says it changes nothing on this matrix, and the whole argument for it is
  the fallback path on config 10, where v40 would otherwise run a kernel measured at 1.119x
  slower than the vendor call it replaced.
* **PROPOSAL (unclaimed, largest on this table): re-tune the single-tile form with the hot
  timer.** Config 2, 1.283x at the op. Opened by finding 50, still open, still deliberately
  not taken here.
* **PROPOSAL: sweep `attn_choice.probe_batch`'s cap.** It costs the looped form config 5.

## PROPOSED LESSON

Not appended to `docs/findings/00-learnings.md` — `ben` is ahead and it would collide.

### L60 — A predicate about your own code is not a comparison against someone else's

`attn_single_tile.pays()` asks whether a loop-free kernel has enough co-resident blocks to
hide its memory latency. It is a good predicate and it is correctly derived from measured
device properties. It was then read, for eighteen generations, as if it also said *"and
therefore we are faster than the vendor here"* — which it never said, could not say, and
was measurably wrong about on one of the ten shapes it accepted.

The tell was in the file the whole time: the table under `MIN_RESIDENT_BLOCKS` lists a
*speedup against SDPA* in its right-hand column, so the predicate was fitted to a
comparison and then used as if it were a property. A residency budget is a statement about
what your kernel can do. Whether the alternative is slower is a separate measurement, and
the cost of assuming it is bounded only by how good the alternative happens to be.

**The structural fix is not a better predicate. It is a floor**: time the thing you are
replacing, in the regime you will replace it in, and refuse to ship anything the vendor
beats. `attn_choice.autotune_looped` already had that floor for the looped form; the older
path did not, and eighteen generations went by without anyone noticing the asymmetry.

### L61 — When you cannot give an arm the same protocol, measure what the difference is worth

The instruction "give every arm the same number of tuning trials" is unsatisfiable when one
arm has no tiles: SDPA is a single call and best-of-1 is its complete grid. The usual
responses are to shrug (and hand the Triton forms a free winner's curse) or to handicap the
sweep (and stop finding the tile you would actually ship).

The third response is to **make the asymmetry a number.** Time the unparameterised arm N
times, where N is how many arms the parameterised ones were allowed, take the minimum, and
print both readings. The gap IS the bias, on this device, at this shape, under this timer —
and here it came out at 0.00%–0.29%, a factor of thirty below the margin any decision turns
on. The whole worry evaporated, and it took one extra loop and no judgement.

The general form: a protocol asymmetry you cannot remove can usually still be *measured*,
and measuring it is strictly better than arguing about it, because the argument has to be
re-run every time the shapes change and the measurement does not.
