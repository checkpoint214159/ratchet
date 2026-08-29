# Finding 30 — the out-projection's head-major gather does not exist, and what replaced it

Generation 24, branch `cand/g24/outproj-prologue`, parent `v18_capture_insurance`.
Reproduce with `python3 bench/probe_outproj.py` (part 1 needs no GPU lock).

## The claim that was commissioned

Proposal D-02 asked for a GEMM whose **prologue** does a head-major gather. The premise:
attention writes `ctx` as `[B, H, S, hd]`, the out-projection wants `[B*S, D]`, so

```python
x = x + F.linear(ctx.transpose(1, 2).reshape(B, S, D), out_w, out_b).float()
```

reads through a transposed view and forces a copy — the mirror image of
`bench/kernels/qkv_headmajor.py`, whose epilogue **scatter** is worth 1.163x on its own
segment. A profile of config 6 supported it: `Memcpy DtoD` at 6.6% of forward time,
sitting right next to the GEMM bucket.

## It is false, at every shape in the matrix

`F.scaled_dot_product_attention` on this card does not return a `[B, H, S, hd]`-contiguous
tensor. It returns a `[B, S, H, hd]`-**contiguous** buffer wearing a head-major view:

| config | d_model | `ctx.stride()` | head-major? | `transpose(1,2)` |
|---|---|---|---|---|
| 1–6, 12 | 128 | `(16384, 32, 128, 1)` | no | **contiguous** |
| 7 | 32 | `(4096, 8, 32, 1)` | no | **contiguous** |
| 8 | 1024 | `(131072, 256, 1024, 1)` | no | **contiguous** |
| 9 / 10 / 11 | 128 | `(16384, 128/64/8, 128, 1)` | no | **contiguous** |
| 13 | 128 | `(131072, 32, 128, 1)` | no | **contiguous** |

All thirteen runnable configs, head_dim 8 through 256. So `ctx.transpose(1, 2)` is
already contiguous, `.reshape(B, S, D)` is a **free view**, no copy is emitted, and the
gather D-02 was built to absorb does not exist to be absorbed. The `Memcpy DtoD` in the
profile is something else and is still unattributed.

**This cost one script and no GPU time to establish**, and it should have been the
proposal's own falsifier rather than the expander's first act. Same shape as L27's audit
rule: the proposal inferred a layout from the *source expression* and never printed
`.stride()`. A tensor's strides are observable in one line; an argument about them is not
evidence. Third time in this project that reading the actual artifact beat reasoning about
it (finding 11's mask semantics, finding 23's head_dim=8 backend table, this).

## What survived, and it is a different mechanism

The two-kernel path still materializes an fp16 `[M, D]` temporary that the very next
kernel reads back to widen and add to the fp32 residual. Per token:

```
two kernels   read ctx 2D + write o 2D | read o 2D + read res 4D + write y 4D = 14D
fused         read ctx 2D              | read res 4D + write y 4D            = 10D
```

29% of the segment's traffic and one launch of two. `bench/kernels/outproj_resid.py` does
the GEMM with the fp32 widening and the fp32 residual add in its **epilogue**. Measured
against the `torch.compile`d two-kernel path at all thirteen configs: **1.28x–1.58x, no
losing shape**, and ~600x tighter against fp64 (1.4e-07 vs 1.2e-04 max_abs) because the
fusion *deletes* an fp16 rounding step rather than adding one.

Note what changed: the proposal's mechanism was **layout**, the surviving mechanism is
**materialization**. They are not the same claim and the scores in D-02 belong to the dead
one. A proposal whose premise dies should be re-scored, not inherited.

## Two things the tuning taught

**The tile crossover is SM saturation, not token count.** A wide tile (BM 64, BN 128,
BK 64, 8 warps) wins above 8,192 tokens and loses below 2,048; a small tile (32, 32, 128,
8) does the reverse. The wide tile emits `ceil(M/64) * ceil(D/128)` CTAs — 32 at
M=2,048 and 128 at M=8,192, against this card's **66 SMs**. So the predicate is
`programs >= props.multi_processor_count`, a measured device property, not a token
threshold fitted to this matrix. Guessing one tile would have cost 1.0x on half the
matrix — the v20 lesson (0.88x guessed, 1.163x tuned) reproduced.

**Triton vectorizes what it can prove, and it could not prove this.** Config 11
(heads=16, head_dim=8) ran at 1.185x where config 1 ran at 1.486x on *arithmetically
identical* work — same M, same D, same ctx strides, same addresses. The gather form
`h * sc_h + e * sc_e` with `h = k // HD` lets Triton prove a contiguous run of only `HD`
elements, which at head_dim 8 is 16 bytes — half a sector. Writing the same addresses as
`m * sc_row + k` when the token-major view is contiguous restored it to **1.500x**. The
general form is kept behind a `CONTIG` constexpr so a backend that really does return
head-major `ctx` is still correct.

Two identical-work shapes measuring 35% apart is a signal, not noise. It was nearly
dismissed as run-to-run variance under L29's ±7% floor; it reproduced across three runs,
which is what made it worth chasing.

## Honest expectation (L33)

The out-projection plus the pointwise add it absorbs is on the order of **8–10% of config
6's forward time**, so 1.3x on the segment is worth **2–3% end to end** — inside L29's
±7% noise floor, and less on a 13-config geomean. **The screen cannot resolve this and
neither can one sweep.** The defensible claims are the per-segment number, one fewer
launch per layer, and the tolerance margin returned on a path that runs `num_layers` times
per forward (L26: our worst config sits at 94% of the 2e-3 budget).
