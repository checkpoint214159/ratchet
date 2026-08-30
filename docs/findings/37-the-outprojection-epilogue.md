# Finding 37 — the out-projection in the attention kernel's epilogue, and the accuracy claim it corrects

Generation 31, branch `cand/g31/outproj-epilogue`, parent `v26_causal_correct` (the frontier).
Reproduce with `python3 bench/probe_outproj_epilogue.py --accuracy` (parts 1–3 need no GPU
lock beyond the tuner's own micro-benchmarks); `--time` adds the op-level segment timing
and takes the lock.

Ported from `cand/g24/outproj-prologue`, which built the same fusion one kernel further out
against the OLD frontier (v18, SDPA-based). This is not a re-derivation: g24's surviving
mechanism, its tile-crossover rule and its layout falsification are all inherited. What
changed is that v23 replaced SDPA with a kernel we own, so the fusion can move inside it.

## What was built

v23/v26's attention half of a layer is three kernels:

```python
ctx = single_tile_attention(qkv, ...)          # our Triton kernel, writes fp16 [B,S,D]
o   = F.linear(ctx, out_w, out_b).float()      # cuBLAS GEMM, writes fp16 [B,S,D]
x   = x + o                                    # Inductor's widen-and-add pointwise
```

The middle kernel reads back a tile that was in the first kernel's registers a microsecond
earlier. `bench/kernels/attn_outproj.py` finishes the contraction there. Per token per
layer, epilogue traffic only (the Q/K/V reads are common to both sides and excluded from
both):

```
v23   write ctx 2D | read ctx 2D + write o 2D | read o 2D + read x 4D + write y 4D = 16D, 3 launches
v31                                           |             read x 4D + write y 4D =  8D, 1 launch
g24   (against SDPA's ctx, for comparison)                                         = 14D -> 10D, 2 -> 1
```

**The program shape had to change and that is the whole cost.** v23 runs one program per
(batch, head, query block); the out-projection contracts across *every* head, so no v23
program holds enough of a row to finish it. The alternative — keep v23's grid and
`tl.atomic_add` each head's `[BM, D]` partial — is a split-K GEMM with `heads`
read-modify-write passes over an fp32 buffer, which is *more* traffic than the thing being
removed plus a seeding kernel for the residual. Rejected on arithmetic, not tried. So one
program owns a (batch, query block) and loops over heads inside, accumulating the
projection in one fp32 `[BM, d_model]` register block. That costs a factor of `heads` in
grid size and an accumulator in registers, and both cut the occupancy that a kernel of this
shape depends on entirely for latency hiding.

## The layout premise, asked of our own code

Finding 30 killed g24's commissioned mechanism: SDPA returns a `[B,S,H,hd]`-contiguous
buffer wearing a head-major view, so the gather it was told to absorb does not exist. The
assignment for this candidate asked whether the same is true of v23's output. **It is, more
flatly** — probe part 1, all ten shapes where v23's kernel runs:

```
cfg1   D128 H4  hd32   ctx.stride() (16384, 128, 1)   contiguous True
cfg7   D32  H4  hd8    ctx.stride() (4096, 32, 1)     contiguous True
cfg11  D128 H16 hd8    ctx.stride() (16384, 128, 1)   contiguous True
cfg12  D128 H4  hd32   ctx.stride() (4096, 128, 1)    contiguous True
                                    ... 10 of 10 contiguous, no gather anywhere
```

v23 allocates `torch.empty((B, S, d_model))` and writes head `h` at column offset
`h * head_dim`, so `ctx` is token-major contiguous **by construction** — no view, no
transpose, no stride to inspect. Two consequences: the win here is **materialization, not
layout**, the same correction g24 had to make; and g24's `CONTIG` constexpr, worth
1.185x → 1.500x at head_dim 8 because Triton could otherwise prove only a 16-byte
contiguous run, has no analogue, because this kernel never addresses `ctx` at all.

## THE ACCURACY CLAIM IS SMALLER THAN g24 REPORTED, AND THE REASON IS THE REFERENCE

g24 reported the fusion **~600x tighter** against fp64 (1.4e-07 vs 1.2e-04). Measured here,
probe part 3, with two references over the identical two arms:

| shape | split (fp64 from qkv) | fused | tighter | epilogue-only reference |
|---|---|---|---|---|
| cfg1  | 1.496e-03 | 7.056e-04 | **2.1x** | 12x |
| cfg4  | 1.133e-03 | 8.046e-04 | **1.4x** | 9x |
| cfg5  | 1.186e-03 | 7.672e-04 | **1.5x** | 9x |
| cfg6  | 1.310e-03 | 7.786e-04 | **1.7x** | 7x |
| cfg7  | 1.338e-03 | 1.041e-03 | **1.3x** | 1177x |
| cfg11 | 1.252e-03 | 7.189e-04 | **1.7x** | 8x |
| cfg12 | 1.533e-03 | 7.655e-04 | **2.0x** | 15x |

The two columns differ by up to three orders of magnitude on the same two tensors. The
right-hand reference takes the fp16 `ctx` as **given** and therefore scores only the
projection — which is the only way to reach a 600x-shaped number, and is what g24's
committed probe (`bench/probe_outproj.py`, part 2) is built around. The left-hand reference
computes the fp64 answer from `qkv`, so the fp16 rounding of `ctx` — common to both arms,
and unavoidable because tensor cores take fp16 operands — is inside the comparison, where
it dominates.

**Both are honest about different things and only the left one answers "is this candidate
more accurate".** The right one credits the fusion with removing an error term it does not
touch. This is [L33]'s shape in accuracy rather than in speed: *an isolated measurement
measures the isolation.* g24's finding-30 headline should be read as an epilogue-only
number, not a candidate-level one.

End to end through four layers the effect shrinks again, because the FFN megakernel and the
LayerNorms contribute error the epilogue does not touch. Seven seeds at the cfg7 shape,
max_abs against the fp32 baseline, zero failed elements in all fourteen runs:

```
seed      5        6        7        8        9       10       11
v26   1.641e-3 1.453e-3 2.022e-3 1.449e-3 1.674e-3 1.318e-3 1.465e-3
v31   1.874e-3 1.476e-3 1.471e-3 1.354e-3 1.398e-3 1.261e-3 1.301e-3
```

v31 is tighter on six of seven. **Seed 5 is the counterexample and it is kept here rather
than dropped**: at 1.874e-03 it is 94% of the locked 2e-3 budget, and v26's own seed 7 is
2.022e-03, *over* the absolute bound and still passing on the OR rule. That is [L4]
verbatim — judge by failed elements, never by max_abs — and [L26]'s warning that the margin
is thinner than it looks. A one-seed margin comparison is a lottery; the seven-seed one is
worth about 8%, not 600x.

## The predicate, and what it declines

Three conditions, shapes and measured device properties only (CLAUDE.md rule 2):

1. **legality** — an `mma.sync.m16n8k16`-shaped tile whose working set fits the register file;
2. **residency** — ≥ 4 resident blocks per SM. This is v23's *measured* crossover reused
   **unchanged and therefore conservatively**: v23's kernel is loop-free and has no latency
   hiding but other blocks, while this one has a head loop whose iterations are independent,
   so its true crossover should be *lower*. It is not lowered without a measurement;
3. **saturation** — `programs >= props.multi_processor_count`. Fusing the epilogue costs a
   factor of `heads` in grid size, so a batch that fills the card under v23's grid may not
   fill it under this one. This is exactly the rule **g24 measured** for the out-projection
   GEMM's tile crossover — the sign flipped at 66 SMs, not at a token count.

Evaluated on this card:

| accepted | declined |
|---|---|
| 1, 4, 5, 6, 7, 11, 12 | 2, 3 (8 and 32 programs against 66 SMs) · 9, 10, 13 (under 4 resident blocks) · 8, 14 (no legal tile, as for v23) |

Every declined shape falls back to v23's split path, which is the frontier and is already
fast. Note that **config 6 — 1.28M tokens, 48.5 s of a 112 s full sweep, and the shape the
commissioning profile was taken on — is in the accepted set.**

Unlike g24, the **masked path is supported rather than declined**: one `tl.where` on the
projection output before the residual add, which is exactly the `masked_fill` v8's fast path
performs at that point, and a test asserts an invalid token is left holding the residual
bit for bit. Padding ratio is a benchmark-exposed knob and finding 11 exists to serve it
([L5]).

## The screen ran, and the useful number is not the verdict

```
python3 bench/screen.py --candidate v31_outproj_epilogue --parent v26_causal_correct
screen configs : (2, 7, 8, 10)
VERDICT        : PROMOTE
screen 2.512x vs parent 2.534x (-0.9%) -- within or above noise
```

**PROMOTE here means almost nothing, and the per-config rows say why.** Of the screen set,
the predicate accepts config **7 only**; 2, 8 and 10 are declined and run the parent's code,
so three of the four rows are measuring v26 against v26 by construction:

| config | v26 (ledger) | v31 (screen) | delta | fused? |
|---|---|---|---|---|
| 2  | 0.06144 ms | 0.06115 ms | −0.5% | no |
| 7  | 0.08704 ms | 0.09062 ms | **+4.1% slower** | **yes** |
| 8  | 6.54851 ms | 6.54496 ms | −0.1% | no |
| 10 | 0.24474 ms | 0.24474 ms |  0.0% | no |

So the aggregate is carried by declined configs measuring flat, and **on the one config
where the fusion actually fires the candidate is 4.1% slower** — one pass, inside [L29]'s
±7% floor, so not decisive in either direction, but it is the opposite sign from the
hypothesis and it must not be hidden behind a PROMOTE. Config 7 is also the least
favourable accepted shape: `d_model = 32`, so the projection GEMM the fusion absorbs is
32×32 while the occupancy the fusion spends is the same as everywhere else.

The screen cannot settle this, and re-running it would only resample the same four configs.
The shape the claim actually lives on — config 6, `d_model` 128 and 1.28 M tokens — is
deliberately excluded from the screen set because it alone is half a full sweep.

## The op-level probe and the harness disagree, and the probe loses

Probe part 4, under the GPU lock, min-of-3 × `do_bench`. The baseline arm is the split path
**`torch.compile`d**, deliberately, so that Inductor fuses its widen and its add — timing
them as separate eager ops is the exact mistake that made v19's op probe read 3.84x on a
candidate the harness measured flat ([L41]). Batch is capped at 256, so the cfg6 row is
B = 256 rather than 10 000.

| shape | split (compiled) | fused | gain |
|---|---|---|---|
| cfg1  B64  D128 hd32 | 0.0414 | 0.0332 | **1.249x** |
| cfg4  B16  D128 hd32 | 0.0188 | 0.0157 | **1.196x** |
| cfg5  B128 D128 hd32 | 0.0632 | 0.0667 | **0.947x** |
| cfg6  B256 D128 hd32 | 0.1326 | 0.1307 | **1.015x** |
| cfg7  B64  D32  hd8  | 0.0214 | 0.0154 | **1.388x** |
| cfg11 B64  D128 hd8  | 0.0508 | 0.0540 | **0.941x** |
| cfg12 B64  D128 hd32 | 0.0186 | 0.0144 | **1.292x** |

Geomean 1.135x, **two losing shapes**, and the segment is roughly break-even in exactly the
large-batch regime the candidate was aimed at (cfg5 0.947x, cfg6 1.015x). Unlike g24's
"1.28x–1.58x, no losing shape", this fusion does not win everywhere it fires.

**And the probe contradicts the harness on the one config both saw.** cfg7 measures 1.388x
at the segment and **+4.1% slower** end to end. Per [L41] the harness is right until proven
otherwise — three recurrences and counting — and the mechanism is available: in the real
candidate the split path's residual add is followed immediately by `norm2`, so Inductor has
somewhere to fuse it that this probe's isolated baseline does not. The probe removes a
launch the candidate was not actually paying for, which is [L33] precisely: *isolation does
not merely shrink an effect, it can invent one that was never available.*

Recorded because it was measured, not because it is evidence. **Nothing in this section is
a verdict**, and the two losing shapes are the more useful half of it: they say the
occupancy the fusion spends is real and is not always bought back.

## Honest expectation, stated before the sweep ([L33])

The commissioning profile of the frontier at config 6's shape:

```
_ffn_block (ours)                 18.7%
LayerNorm bucket                  33.6%
_attn_single_tile (ours)          15.5%
ampere_fp16 GEMM (QKV proj)       15.4%
Memcpy DtoD                        7.2%
cutlass GEMM (the OUT proj)        6.8%   <- the target
```

**This candidate does not delete the 6.8%.** The projection's arithmetic still happens, now
inside our kernel instead of cuBLAS's, and cuBLAS is very good at it — while our version
runs it at K = head_dim per step (32, or 16 after padding at head_dim 8), which is a thin
`mma` contraction. What is deleted is the `ctx` round trip and two launches of three, i.e.
the memory-bound half of that bucket plus a share of the widen-and-add that Inductor
currently fuses into the LayerNorm bucket.

So the diluted end-to-end ceiling is **3–5% on config 6**, and **less on the 13-config
geomean because six of thirteen configs decline outright**. That is inside the ±7% floor.
Anything above it should be disbelieved before it is celebrated.

The claims that do not depend on resolving 3% are structural: half the epilogue's HBM
traffic, two launches of three removed per layer per forward, a deleted fp16 rounding step
worth 1.3–2.1x of segment margin, a masked path kept rather than declined, and a predicate
that says which of three conditions refused rather than silently falling back.

## What the controller should decide

Per [L39], a candidate whose value the screen cannot see needs a bespoke falsifier rather
than a screen verdict. Here the falsifiers are `bench/probe_outproj_epilogue.py` parts 1–3
and the 37 tests in `tests/bench/test_v31_outproj_epilogue.py`, all of which are
unambiguous. What is *not* established is the speed, and **both speed signals available are
weak and they disagree**: the op-level probe says 1.135x on the segment with two losing
shapes, the screen's single firing config says −4.1% end to end, and [L41] says believe the
harness.

So the honest recommendation is *not* "promote". It is: **this candidate is correct, better
conditioned, structurally cheaper, and unproven on speed — and the only measurement that
would settle it is a full sweep, which is expensive and which the evidence does not
currently justify demanding.** If one is spent, config 6 and config 11 are what it is for;
config 7 is the shape both signals agree least about.

One thing deliberately *not* done: tightening the saturation or residency threshold until
config 7 declines. It would raise the geomean on this evidence and it would be fitting the
predicate to a single screen pass inside the noise floor — the [L29] error, committed on
purpose. If a sweep confirms config 7 is a losing shape, the threshold that excludes it
should be derived from what the sweep shows, not reverse-engineered from a verdict.
