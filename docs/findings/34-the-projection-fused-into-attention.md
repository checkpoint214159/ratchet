# Finding 34 — the QKV projection fused into attention, and the byte count that decides it

Generation 27, `cand/g27/qkv-fused-attn`, parent `v26_causal_correct`. 2026-08-30.

## The question as it was posed

A fresh profile of config 6 — 84% of matrix wall time — showed the fused Q|K|V projection,
a cuBLAS `ampere_fp16` GEMM, at **15.4%**, sitting immediately before our own
`_attn_single_tile` at 15.5%. The GEMM writes a `[B, S, 3*d_model]` buffer and the very
next kernel reads it straight back. The proposal was to delete the buffer, and it framed
the feasibility question as:

> At d_model 128 the fused QKV weight is `128 x 384` fp16 = 96 KB, against this device's
> measured 99 KB opt-in shared memory. Does it fit?

## It fits, and it is the wrong question

96 KB against 101,376 bytes leaves 3 KB, which would indeed leave no room for a score tile.
But **a program that owns one head never needs the whole weight.** Head `h` of Q uses
columns `[h*DH, h*DH+DH)` of `Wq`, and likewise K and V, so the slice that must be resident
is

    3 * d_model * pad16(head_dim) * elem

which on the announced rows is 24 KB at head_dim 32 (configs 1–6, 12), 12 KB at head_dim 8
with d_model 128 (config 11), 3 KB at config 7, 48 KB at head_dim 64 (config 10), and
96 KB only at head_dim 128 (config 9). The 96 KB figure is reached by exactly one shape,
and that shape is declined anyway.

The operand that actually binds is the other one: the **input tile**,
`[next_pow2(S), d_model]` fp16, which all three projections read. 32 KB at d_model 128,
**256 KB at d_model 1024** (config 8), **280 KB at seq_len 1024** (config 13). That is what
refuses the wide model and the long sequence, and it is a shape fact evaluated against
`shared_memory_per_block_optin`, not a config id.

The head decomposition is what makes the whole idea work, and it is not the decomposition
the "does 96 KB fit" framing suggests. Worth carrying: **before asking whether a resource
fits, ask what the unit of work actually needs** — the framing had the program holding
16x more weight than it uses.

## The predicate the mechanism argument did not predict

Every program owns one head and needs the **full model width** to project it, so `heads`
programs each re-read the same input tile. That factor of `heads` is the fusion's one
structural cost and it is invisible until it is counted:

| | fused reads | pair's reads+writes | |
|---|---|---|---|
| config 1/6 (4 heads, d_model 128) | 128 KB | 224 KB | fusion wins |
| config 10 (2 heads, head_dim 64) | 64 KB | 224 KB | fusion wins |
| config 7 (4 heads, d_model 32) | 32 KB | 80 KB | fusion wins |
| **config 11 (16 heads, d_model 128)** | **512 KB** | **320 KB** | **fusion LOSES** |

per sequence per layer, taking no L2 credit. At sixteen heads the fusion moves more bytes
than the buffer it deletes. The op-level probe measured **0.822x** there, so the arithmetic
and the measurement agree, and `moves_fewer_bytes` declines it on the byte count with no
fitted constant — it reduces to roughly `heads <= 7` at this width, derived rather than
tuned.

This is the second time on this lineage that a fusion's cost was a re-read nobody had
counted (finding 25's amortization crossover was the first). **When a kernel is
parallelized over an axis the operand does not carry, count the re-reads before counting
the savings.**

## v23's swept tile reverses, and for an arithmetic reason

v23 swept and found `block_m = 64` beat `block_m = 128` at S = 128 by ~5%: a 128-row fp32
score tile halves resident blocks per SM for no reduction in work. **That reasoning does
not survive the fusion**, because now there is a reduction in work. A query block needs
every key row, so K and V are re-projected once per query block:

    fused rows  = cdiv(S, BM) * (BM + 2*BN)      BM=64, S=128 -> 640
    actual rows = 3 * S                                       -> 384

`block_m = 64` does **1.67x the projection arithmetic** the GEMM did; `block_m = BN` does
exactly 1.0x. The traffic predicate reaches the same tile independently (at `BM != BN` the
input tile is read `BM + BN` rows deep per block, which fails `moves_fewer_bytes` at
d_model 128), and the op probe measured `128x8` fastest on config 6's shape. Three
arguments, one answer.

**A swept constant is only valid under the mechanism it was swept for.** v23's tile was
correct for a kernel that read Q/K/V out of HBM and had no projection to redo.

## The roofline, written down before the measurement

The GEMM being 15.4% of wall time does not mean 15.4% is available. Measured on config 6:

* the QKV GEMM moves 5.24 GB in 8.8 ms = **595 GB/s, 97% of this card's 613.7 GB/s
  roofline**. Its arithmetic intensity is 96 FLOP/B against a 144 FLOP/B ridge, so it is
  bandwidth-bound and at the wall. It is not a badly written kernel with headroom in it.
* v23's attention: 336 GFLOP in 8.9 ms = 37.7 TFLOP/s, 43% of the 88.2 TFLOP/s peak.
* the pair together: 840 GFLOP in 17.7 ms = **47.5% of peak**.

Fusing does not delete the projection's FLOPs, it moves them into our kernel. Traffic falls
~5x (3.28 GB → 0.66 GB per layer) and intensity rises to ~328 FLOP/B, past the ridge: the
stage stops being bandwidth-bound and becomes compute-bound at a 9.5 ms floor. **So the
fused kernel wins if and only if it exceeds 47.5% of peak.** That was a genuine coin flip
when it was written down, and recording it first is what makes the measurement a test
rather than a rationalization.

## What was measured

**Op-level probe — INDICATIVE ONLY (L41).** GPU lock held, min of 5 × `do_bench`, nothing
recorded to the ledger. Fused kernel against `F.linear` + v23's kernel on the same
operands:

| shape | ratio |
|---|---|
| config 6's shape at B=800 | **1.293x** |
| config 7 | **1.500x** |
| config 12 | **1.300x** |
| config 10 | 1.154x |
| config 1 (B=64) | 0.962x |
| config 11 | 0.822x → now declined |

A batch sweep at config 6's shape is stable at **1.25–1.34x for B ≥ 512** and pure noise
below it: 1.53x at B=16, 0.96x at B=64, 0.92x at B=256, on 10–90 µs kernels inside a ±7%
floor. **Do not read the middle of that sweep as signal**, and do not build a batch-size
predicate out of it — that is exactly the fitted-to-noise constant [L29] warns about.

**Screen (configs 2, 7, 8, 10): PROMOTE, 2.559x against the parent's 2.534x, +0.9%.**
Per config, against v26's ledger rows:

| config | v26 ms | v27 ms | ratio | |
|---|---|---|---|---|
| 2 (B=1) | 0.0614 | 0.0676 | **0.91x** | launch-bound; 4 programs on 66 SMs |
| 7 | 0.0870 | 0.0755 | **1.15x** | the real signal in the screen |
| 8 | 6.549 | 6.569 | 1.00x | declined — a control, and it is inert |
| 10 | 0.2447 | 0.2461 | 0.99x | flat |

Config 8 landing at exactly 1.00x is worth its own line: it is the shape the predicate
refuses, and it confirms the decline path is genuinely the parent's code and costs nothing.

## The honest verdict

**The screen cannot see this candidate's claim.** The screen set is (2, 7, 8, 10) and the
candidate was built for config 6, which is not in it — by design, since config 6 alone is
48.5 s of a 112 s sweep. The +0.9% geomean is neither support nor refutation.

Diluting the op-level number properly [L33]: the GEMM and the attention kernel are 30.9% of
config 6 between them, so 1.29x on the pair is a ceiling of
`1 - 0.309*(1 - 1/1.293)` = **7.0% on config 6**. If configs 1–7, 10 and 12 each gained the
full 7%, the geomean would move `1.07^(9/14)` = **+4.5%, inside the ±7% noise floor**. The
defensible claim is per-config, on the matrix's largest shape. Anyone reading a large
geomean change off this candidate should disbelieve it before celebrating it.

## Pre-registered, so it is a prediction and not a fit

Config 2 measured 0.91x. The op probe at that shape measured 0.988x, so the screen's figure
is probably prime-time variance on a 67 µs config rather than the kernel. **If a full sweep
confirms configs 2, 3 and 4 regress**, the discriminator to test is one full wave of
programs:

    heads * cdiv(S, block_m) * batch >= multi_processor_count

which declines config 2 (4 programs), 3 (16) and 4 (64) and accepts config 1 (256) and 6
(40000), and is device-derived rather than fitted. It is deliberately **not** implemented,
because implementing it now would be fitting a predicate to one sub-100 µs measurement.

## Correctness

48 tests in `tests/bench/test_v27_qkv_fused.py`, at the LOCKED tolerance (atol 2e-3 OR
rtol 2e-2, judged by failed elements, never by `max_abs` alone [L4]). End-to-end margin
**62% of budget**, better than the parent's typical 89–97%. The suite covers every viable
tile rather than only the derived one (the tile is autotuned, so all of them ship), both
fallback paths on real models, v26's causal contract (finding 32), the head-major output
layout, exact head_dim padding, and the input-dependence invariant [L23]/[L25].

The projection accumulates in fp32, adds the bias in fp32, and rounds to fp16 once —
structurally identical to `F.linear` on fp16 operands with an fp32-accumulating cuBLAS
epilogue, so no rounding step is added or removed. The softmax stays fp32 (finding 08).

## Handoff note on the lineage guard

`tests/bench/test_lineage_topology.py` currently fails for `v23_single_tile_attn` and
`v26_causal_correct` (both pre-existing, both documented in finding 28 / [L40]). v27 was
cut from v26's commit, so it adds no third case, but it will inherit v26's extra ancestor
(`v19_norm_fused`) and trip the same assertion once it has ledger rows. The guard subtracts
only `PRE_FINDING_28`; it does not subtract violations inherited *through* a declared
ancestor, which are already counted against that ancestor. That is a decision about an
invariant check and is left to the controller — editing a guard test to make one's own
candidate green is the anti-pattern this file keeps rediscovering.
