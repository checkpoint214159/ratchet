# Finding 31 — A hand-written attention kernel beats FlashAttention where the score matrix fits, and the predicate for "where" is not the one either proposal guessed

*2026-08-30. Candidate `v23_single_tile_attn`, branch `cand/g23/single-block-attn`,
parent `v18_capture_insurance` (`d56e12d`). Proposals C-01 (agent C) and D-04 (agent D).*

**Numbering note:** taken as 31 to avoid colliding with siblings on `cand/g20`–`g22` and
with 28–30 on `ben`. Renumber at merge if it clashes.

---

## What was built

`bench/kernels/attn_single_tile.py`. One Triton program per `(batch, head, query block)`:
one `tl.dot` for `QKᵀ`, one `tl.where` for the causal triangle, **one ordinary softmax**,
one `tl.dot` for `P V`, store. No K/V loop, no running max, no running sum, no
accumulator rescale — because with `seq_len ≤ 128` there is only ever one K block, and
flash's rescale machinery exists solely to make a multi-tile reduction equal the
single-tile one.

It also reads Q, K and V straight out of the fused `[B, S, 3·d_model]` projection buffer
by stride arithmetic and writes `[B, S, d_model]` head-major. That deletes the `.split`,
the three `.transpose(1, 2)` views, and — the part that actually costs a kernel — the
`transpose(1, 2).reshape` repack, which is a real copy of a whole activation tensor per
layer. **A material part of the win is not about attention at all.**

Exactness: the causal mask is exact (a masked entry carries exactly zero softmax weight),
padding `head_dim` 8 → 16 inside the kernel contributes exactly zero to the contraction,
and the accumulator is fp32 throughout. Measured `max_abs` against the fp32 reference is
7.8e-4 – 9.8e-4, **39–49% of the 2e-3 budget** — better margin than the frontier's worst
config at 94% (L26).

## The two proposals disagreed, and the measurement settled it

C-01 specifies **one program per (batch, head)**: `block_m == seq_len`, the literal
"single block". D-04 specifies a single **tile of K/V** and leaves the query block open.

Swept (min of 5 × `do_bench`, GPU lock held, indicative only):

| shape | 64 rows | 128 rows |
|---|---|---|
| S=128, head_dim 8 | **12.3 µs** | 13.3 µs |
| S=128, head_dim 32 | **20.5 µs** | 21.5 µs |
| S=128, H=16, head_dim 8 | **30.7 µs** | 30.7 µs |

`block_m = 64` beats `block_m = 128` on every shape the timer can resolve, by ~5%.
**C-01's stricter claim is measurably wrong; D-04's shape is what we implemented.** 128
query rows put the fp32 score tile at 64 KB, halving how many blocks stay resident per SM
for no reduction in work, and the K/V tiles are cheap enough to re-read once per block.

## The predicate is the durable result

Op-level, against `SDPA + repack`, on the runnable shapes:

```
cfg12 2.43x   cfg11 1.83x   cfg 7 1.58x   cfg 1 1.55x   cfg 4 1.48x
cfg10 1.19x   cfg 5 1.11x   cfg 3 1.17x   cfg 2 1.20x
cfg 9 0.94x   cfg 8 0.84x        <- legal, correct, and SLOWER
```

Neither proposal predicted the losses. Both framed the predicate as "does the score
matrix fit on chip", which is true at head_dim 128 and 256 and yet those are exactly where
the kernel loses. The mechanism is different:

> **A loop-free kernel has nothing to software-pipeline.** Every program is one long
> dependent chain — load Q/K/V → dot → softmax → dot → store — so its only latency hiding
> is other resident blocks on the same SM. The fp32 score tile plus the Q/K/V operands
> live in registers, so the register working set caps residency *directly*. That is the
> price of deleting the tile loop: flash hides its memory latency inside the loop; we have
> to hide ours across blocks.

| head_dim | best tile | registers/block | blocks/SM | op speedup |
|---|---|---|---|---|
| 8 | 64 × 4 | 10752 | 6.1 | 1.58x |
| 32 | 64 × 4 | 13312 | 4.9 | 1.55x |
| 64 | 32 × 8 | 13312 | 4.9 | 1.19x |
| 128 | 64 × 4 | 28672 | 2.3 | **0.94x** |
| 256 | 64 × 8 | 49152 | 1.3 | **0.84x** |

`MIN_RESIDENT_BLOCKS = 4` is that measured crossover, evaluated against
`regs_per_multiprocessor` and `max_threads_per_multi_processor` read off the device — no
config ids, and a card with a smaller register file declines more without being retuned.
It refuses head_dim 128 and 256, seq_len 1024 and seq_len 100000, and falls back to v18's
SDPA path there unchanged.

## The tile is autotuned, with the derived tile holding the ground

No formula fits this card: 64×4 warps wins at head_dim 32 while 32×8 wins at head_dim 64
at an *identical* register cost. So the candidate times its viable tiles once at prime
time, before compilation and capture, on a probe batch capped from the measured SM count
— and the derived tile stands unless something beats it by more than 10%.

That last clause was not caution, it was necessary. These kernels run in 1–13 µs against a
CUDA event timer that resolves ~1 µs. Without it the autotuner picked a different tile on
consecutive runs of the same shape, which would have injected its own noise into every
measurement ever taken of the candidate. **An autotuner that cannot resolve its own
choices is a random number generator wired to the frontier.**

## Screen: PROMOTE, and one row that should not be over-read

`bench/screen.py`, configs 2/7/8/10, one pass, against `v18_capture_insurance`:

| cfg | regime | compiled | v18 | v23 | v18 | v23 | Δ |
|---|---|---|---|---|---|---|---|
| 2 | launch-bound | 0.1352 | 0.0707 | 0.0655 | 1.913x | 2.063x | **+7.8%** |
| 7 | head_dim 8 | 0.3348 | 0.1147 | 0.0922 | 2.919x | 3.633x | **+24.4%** |
| 8 | wide (declined) | 14.4230 | 6.5495 | 6.5270 | 2.202x | 2.210x | +0.3% |
| 10 | head_dim 64 | 0.5417 | 0.2417 | 0.2601 | 2.242x | 2.083x | **−7.1%** |

Geomean 2.423x against the parent's 2.292x, **+5.8% — inside the ±7% noise floor (L29)
and not a win on the geomean.** The defensible claims are per-config:

* **config 7 is +24.4%**, five times the ±7% floor, and it is the configuration the
  rubric named as never investigated. Its predicted ceiling from attention's share alone
  was ~1.20x; it beat that, which is consistent with the repack deletion being a second,
  unaccounted effect.
* **config 8 is +0.3%** — the declined path, unchanged, exactly as designed.
* **config 10 is −7.1%**, and this is the open question. head_dim 64 is the marginal case:
  it sits at exactly `MIN_RESIDENT_BLOCKS` and measured only 1.19x at the op. One screen
  pass at 0.26 ms cannot separate −7.1% from noise. **Do not resolve it by raising the
  threshold** — head_dim 32 (a 1.55x win) sits at the same 4.9 blocks/SM, so that knob
  cannot tell them apart.

A discriminator that *would*, pre-registered here so it is a prediction rather than a
fit: **require the score tile to cost at least as many registers as the operands**
(`block_m·BN·4 ≥ (block_m + 2·BN)·pad16(head_dim)·2`). Scores dominate at head_dim 8 and
32 and operands dominate at 64, 128 and 256, so it separates the measured wins from the
measured losses on every row. It is deliberately NOT implemented: it would be fitted to a
single −7.1% row inside the noise floor. If the full sweep confirms a regression at
head_dim 64, that clause is the fix; if it does not, the clause was never needed.

## L33, stated by us rather than discovered by a reviewer

Attention is 18–46% of layer time depending on config, so an op-level 1.1x–2.4x dilutes
hard. Ceilings from attention's share alone: ~1.25x (cfg 11), ~1.20x (cfg 7), ~1.15x
(cfg 12), <1.10x everywhere else, and **exactly nothing on configs 8, 9, 13 and 14, by
design**. Configs 11, 12 and 13 are not in the screen set, so the screen cannot see this
candidate's two largest predicted wins — the full sweep is what settles it.

## A defect found on the way, and fixed

`bench/screen.py` recorded a **REJECT** for v23 on its first run. Nothing had been
measured: `run_matrix` refuses outright when another process holds the GPU lock, printing
the reason to *stdout* and exiting 3, and screen.py carried only stderr on a non-zero
exit. The refusal was invisible, the result set was empty, and `decide` reported
"incomplete screen".

There is now a **BLOCKED** verdict, distinct from REJECT, with the refusal text printed,
and two regression tests. The false REJECT row is left in `screen_log.jsonl` — the log is
a record of what happened.

This is L38 one turn later: *a guard whose refusal cannot be told apart from the thing it
guards against is worse than no guard, because it manufactures a confident wrong answer.*
The v23 row was recoverable because a human was reading. In an autonomous loop it would
have killed the candidate.

## Region closed, region opened

**Closed:** "a hand-written attention kernel cannot beat the vendor's on this matrix" is
false. It can, by 1.5–2.4x at the op, wherever the score matrix fits in registers with
room left for other blocks.

**Also closed:** the same kernel is *slower* than FlashAttention at head_dim 128 and 256,
and the reason is residency, not arithmetic. Nobody needs to re-try wide heads with this
shape of kernel.

**Opened:** the losses came from a mechanism neither proposal modelled. Both scored A2
"headroom" at 4–5 on a fits-on-chip argument; the fits-on-chip argument was right about
where the kernel is *legal* and silent about where it is *fast*. Every future
"materialize what flash streams" proposal should be asked for its residency budget, not
its capacity budget.
