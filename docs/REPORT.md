# Ratchet — technical report

TikTok TechJam 2026, GPU kernel optimization track. Target: the pinned reference
`benchmarks/reference/torch_transformer_benchmark.py`, a four-layer transformer forward,
across the 14 announced shape configurations, at the evaluator's own locked tolerance
(`atol 2e-3`, `rtol 2e-2`, OR-combined).

This document is written for a reviewer who wants to check us. Every number below names
the file it came from. Where a claim rests on a single measurement, it says so. Where the
project's own record disagrees with itself, this document says which reading it prefers
and why, rather than picking the flattering one.

**Companion documents.** `README.md` at the repo root is the Devpost project description
and covers the same ground at lower resolution; where the two disagree numerically, this
one is later and was derived from the ledger directly. `docs/loop/method.md` is the
24-rule distillation of the method and is the most transferable artifact in the repo.
`docs/findings/00-learnings.md` (48 entries, L1–L53, with gaps where numbers were reserved
on candidate branches) and `docs/findings/NN-*.md` (45 numbered
findings) are the primary sources cited throughout.

---

## 0. Provenance of the numbers in this report

Every timing, correctness margin and speedup in sections 4 and 7 was recomputed from
`bench/results.jsonl` for this document. **Nothing was copied out of prose**, including
out of the findings — several findings quote figures from the branch they were written on,
and the branch numbering and the trunk numbering differ (see the numbering notes at the top
of findings 43 and 44).

The ledger state used:

```
HEAD                ce0cc18  (branch docs/tech-report, cut from ben)
rows read           636
newest row          2026-08-30T11:46:00Z
statuses            570 ok · 30 oom · 14 crash · 12 incorrect · 7 reference_infeasible
```

**A `v37_recombined2` sweep was appending to the ledger while this was written.** Its rows
for configs 1–8 are included above; configs 9–14 had not landed. Re-deriving these tables
later will therefore show more rows, and section 4.5 flags the one place that matters.

`bench/results.jsonl` is opened append-only and is never edited, sorted or pruned. Each row
is keyed to `(commit_sha, config_id)` and carries its own method metadata — sample count,
reduction, whether the arms were interleaved, whether the clocks were locked, whether the
GPU was exclusive. That is what makes this section possible at all.

---

## 1. Summary, in three numbers that disagree

There is no single headline, and manufacturing one would be the exact error this project
was built to avoid. The three defensible framings, all against
`torch.compile(mode="max-autotune")` with TF32 on — never against eager:

| framing | value | which candidate | source |
|---|---|---|---|
| geometric-mean speedup, 13 runnable configs | **3.10x** | `v26_causal_correct` | ledger, isolated arm |
| | **3.25x** | `v34_launch_bound` | ledger, isolated arm |
| | **3.32x** | `v36_gemm_gelu` | ledger, interleaved arm (§4.3) |
| total wall time over the 13 runnable configs | **76.9 → 69.2 ms** | v17 → v26 | ledger |
| `matrix.weighted_score`, cap 3.0 | **2.4129 → 2.5287** | v26 → v36 | ledger |

They disagree because they weight the matrix differently, and the disagreement is
informative rather than embarrassing:

* The **geomean** weights a 0.06 ms configuration exactly as heavily as a 57 ms one.
  Re-measuring byte-identical code once moved it by +2.9% (finding 32).
* **Total wall time** is dominated by one row: config 6 is **83.0%** of the frontier's
  matrix wall (57.44 ms of 69.16 ms). Anything that does not move config 6 does not move
  this number.
* **`weighted_score`** clips each configuration's speedup at 3.0, so five of the thirteen —
  **configs 3, 6, 7, 11 and 13** — are already past the cap and score nothing further
  however much faster they get. Config 6 is one of them, and it is 83% of the wall. The cap
  exists so a spectacular win on one regime cannot carry a submission that is mediocre
  everywhere else; the cost is that the metric is blind to most of the machine.

Against the reference benchmark's **default** eager baseline the same frontier measures
**12.3x** (v26) — a number a default run of the graded harness will literally print, and one
we consider dishonest to quote alone. Correcting the baseline mid-project deflated our
headline from 7.2x to 1.69x and flipped two configurations from win to loss (finding 12).
Both numbers stay in the ledger.

Correctness: **13 of 14 configurations pass** at the locked tolerance with **0 failed
elements**, at 60.9%–78.6% of the absolute budget (§4.4). The fourteenth is the one where
the *reference itself* cannot run — §7.

---

## 2. The target, and why it is a dispatch problem

The shape matrix is `bench/matrix.py`, encoded once as executable data; cite it rather than
restating rows. Read as an ablation grid it sweeps batch size (rows 1–6), model width
(7–8), head count (9–11), sequence length (12–13), and one extreme (14).

Two facts about the announced matrix change the calculus and are both in `matrix.py`'s
docstring:

* **Every announced row is causal**, while the reference's own default is
  `causal: bool = False`. Half the score matrix is structurally zero and skipping it in a
  fused kernel is *exact*, not an approximation.
* **`ffn_dim == d_model` on every row**, against the reference's default 4x expansion. The
  FFN is ~4x less dominant than profiling the reference's defaults would suggest, which
  pushes attention up in relative cost — and, as it turns out, is the single structural fact
  that makes the fused FFN kernel possible at all (§3.1).

The device (`docs/00-mission.md`, cached in `ledger/device.json`, measured 2026-08-28):
RTX 4070 Ti SUPER, sm_89 (Ada — `mma.sync` only, no wgmma or TMA), 66 SMs, 99 KB opt-in
shared memory per block, 48 MB L2, **613.7 GB/s measured** bandwidth (91% of the 672 GB/s
theoretical), ridge point **144 FLOP/B**, peak **88.2 BF16-TFLOP/s** at FP32 accumulate,
2.22 µs eager launch overhead. Clocks are **not lockable** under WSL2.

Under those constraints one implementation cannot win the matrix. Config 2 (B=1) spends
232 µs of CPU against 126 µs of GPU (finding 16) and is bound by dispatch; config 6 (1.28M
tokens) runs its FFN at **99.2–100.1% of measured bandwidth** (finding 30) and is bound by
HBM. A change that relieves bandwidth pressure is pure overhead in the first regime, and a
change that removes kernel launches is invisible in the second.

---

## 3. What was built

Five mechanisms, each with a *predicate* that decides where it fires. **Every dispatch
predicate is a function of measured device properties — never a configuration id, never an
announced shape constant.** That is a deliberate contract (CLAUDE.md rule 2; gate G2 of the
proposal rubric treats branching on a benchmark's own shapes as benchmark special-casing)
and it is enforced by tests that inspect the source rather than trust prose —
`tests/bench/test_feasibility.py` asserts structurally, via `co_consts` and the function
signature, that no config id and no `100000` literal appears in any predicate. The
operational check is that halving the device property in the test flips the decision:
halving the SM count drops config 12 out of the one-wave set, and halving free memory flips
config 6 from resident to streamed.

### 3.1 The fused FFN block — `bench/kernels/ffn_fused.py`

One Triton kernel for `res + (gelu(xn @ W1 + b1) @ W2 + b2)`: both GEMMs, the GELU and the
fp32 residual add, in one launch.

**Mechanism.** Inductor fuses elementwise work into a GEMM epilogue but does *not* fuse GEMM
into GEMM — that needs the intermediate tile held in registers across two `mma` chains
instead of round-tripping through HBM. That is a structural gap in the compiler, and this
matrix exposes it because `ffn_dim == d_model`: at `d_model = 128` in fp16, W1 and W2
together are `2 × 128 × 128 × 2 = 64 KB`, inside this device's measured 99 KB opt-in shared
memory. On a conventional 4x-expansion transformer they would be 512 KB and this kernel
could not exist. Per token the activation traffic drops from four tensor passes (read x,
write h, read h, write y) to two.

Two correctness points that are easy to get wrong and are both load-bearing: the reference's
GELU is `approximate="none"`, the **exact erf** form — the tanh approximation differs by up
to ~1e-3 relative, half our entire budget spent on an approximation nobody asked for — and
the **residual add is fp32**, because finding 08 established that an fp16 residual stream is
1.4x faster and fails 11 of 13 configurations at 3.3–5.3x over budget.

**Predicate — `amortizes()`.** The kernel's whole advantage is loading the weights once and
streaming activations past them, so it must ask whether enough tokens reuse them:
`weight_bytes / tokens <= 0.002 × activation_bytes_per_token(d_model)`. Below the crossover
the program moves more bytes of weights than of data. It was calibrated against a measured
sign flip (finding 25): at M=128 the kernel is **+113% slower**, at M=512 +49%, at M=2048
break-even, at M=1.28M a win. Expressed as a fraction of activation traffic rather than as a
token count, so it carries to other widths and cards.

### 3.2 Single-tile causal attention — `bench/kernels/attn_single_tile.py`

One Triton program per `(batch, head, query block)`: one `tl.dot` for `QKᵀ`, one `tl.where`
for the causal triangle, **one ordinary softmax**, one `tl.dot` for `PV`, store. No K/V
loop, no running max, no running sum, no accumulator rescale.

**Mechanism.** FlashAttention's online-softmax machinery exists solely to make a multi-tile
reduction equal the single-tile one. Ten of the fourteen announced rows have
`seq_len == 128` and one has 32, so per `(batch, head)` the whole score matrix is 128×128 —
64 KB of fp32, inside the register file of one thread block on this card (65536 32-bit
registers = 256 KB per SM). With one K block, flash's rescale is bookkeeping for a loop that
runs exactly once, and removing it removes its rounding too: this is numerically **no worse**
than the path it replaces, not a tolerance gamble. Causal masking is exact (a masked entry
carries exactly zero softmax weight) and zero-padding `head_dim` from 8 to 16 inside the
kernel contributes exactly zero to the contraction.

**A material part of the win is not attention at all.** The kernel reads Q, K and V straight
out of the fused `[B, S, 3·d_model]` projection buffer by stride arithmetic and writes
`[B, S, d_model]` head-major — bit-for-bit the layout `transpose(1,2).reshape` produces. The
`.split`, three transposed views and the repack (a real copy of a whole activation tensor per
layer) all disappear with it. This is why config 12 — `seq_len 32`, where attention itself is
trivial — gained 25%.

**Predicate — `MIN_RESIDENT_BLOCKS = 4`, evaluated against `regs_per_multiprocessor` and
`max_threads_per_multi_processor`.** This is the durable result of finding 31, and it is
*not* the predicate either of the two proposals that commissioned the kernel guessed. Both
framed it as "does the score matrix fit on chip", which is true at head_dim 128 and 256 —
exactly where the kernel **loses**:

| head_dim | registers/block | blocks/SM | op speedup vs SDPA + repack |
|---:|---:|---:|---:|
| 8 | 10752 | 6.1 | 1.58x |
| 32 | 13312 | 4.9 | 1.55x |
| 64 | 13312 | 4.9 | 1.19x |
| 128 | 28672 | 2.3 | **0.94x** |
| 256 | 49152 | 1.3 | **0.84x** |

A loop-free kernel has nothing to software-pipeline: every program is one long dependent
chain, so its only latency hiding is *other resident blocks on the same SM*. Flash hides its
memory latency inside the tile loop; deleting the loop moves that burden to occupancy. The
kernel therefore **declines** head_dim 128 and 256 and the long-context rows and falls back
to SDPA unchanged — which is why configs 8 and 9 measure within 0.3% of the parent on the
declined path, an in-run control that also demonstrates the sweep was not contended.

The tile itself is *swept at prime time*, not derived: no formula fits this card (64×4 warps
wins at head_dim 32 while 32×8 wins at head_dim 64 at identical register cost), so the
candidate times its viable tiles once on a probe batch sized from the measured SM count and
**the derived tile stands unless something beats it by more than 10%**. That margin is not
caution: these kernels run in 1–13 µs against a CUDA event timer that resolves ~1 µs, and
without the margin the autotuner picked a different tile on consecutive runs of the same
shape. An autotuner that cannot resolve its own choices is a random number generator wired to
the frontier.

### 3.3 The projection GEMM with a GELU epilogue — `bench/kernels/proj_gemm.py`

**Mechanism, from a census of config 9 — the #1 headroom row, which nobody had ever
profiled** (finding 43). Per forward inside the replayed graph, 220.0 µs over 35 nodes: the
sixteen projection GEMMs are **55.0% of device time at 47.3 TFLOP/s, 53.6% of this card's
measured peak**. cuBLAS selects `ampere_fp16_s1688gemm` on all twelve narrow-K calls;
`s1688` names `mma.sync.m16n8k8`, where sm_89 also issues `m16n8k16` — twice the K-depth per
instruction — and `m16n8k16` is what `tl.dot` emits. It does **not** do this at
`d_model = 1024`, where it picks `cutlass_80_tensorop_f16_s16816gemm` and hits 100.4% of
measured peak. The bad selection is specific to narrow K.

And the GELU is its own kernel on every layer of every config, because it sits between two
cuBLAS calls with no pointwise neighbour for Inductor to fuse into, and cuBLAS takes no
epilogue. So it moves into the `ffn_in` epilogue, in the **exact erf** form, applied to the
fp32 accumulator before any downcast.

**The epilogue, not the instruction, is the win.** In-model census at config 1: the
hand-written GEMM beats cuBLAS by **1.05x**, not the 1.20–1.33x the L2-flushed probe
reported — because in the model the weight arena is L2-resident (§6.2 measured that at
94.8%) and the probe's flushed cache understates the vendor. Of a 23.7 µs total saving, the
`s1688` story is worth 3.7 µs and the deleted GELU is worth ~16 µs. Kernel count falls by
exactly four, one per layer, counted from device events.

**Predicate: a measurement, at each of four call sites independently.** `plan()` times the
vendor call against 18 swept Triton tiles on the real operand shapes at prime time and keeps
the vendor unless Triton wins by more than 10%. No shape literal is involved. On the
announced matrix it takes all four sites on config 11, three on 1/9/10, two on 4/12/7, one on
3 and 5, and **none** on 2 (every site ties at ~5 µs; the shape is launch-bound), 6 and 13
(Triton loses at M ≥ 65536), and 8 — where the vendor is at 100.4% of peak and the predicate
declines it without being able to see which config it is on.

Precision is an **identity argument, not a tolerance argument**: this replaces one
fp16-operand/fp32-accumulate GEMM with another, `tl.dot` over the same fp16 operands into an
fp32 accumulator, bias in fp32, one rounding to fp16 — the same single rounding `F.linear`
performs.

### 3.4 Kernel-count reduction: 36 → 20 per forward — finding 39

The frontier launched **36 kernels per forward on every config**, with an identical
decomposition at L=4: 16 GEMMs, 9 LayerNorms (already carrying the residual add and the fp16
downcast, fused in by Inductor), 4 attention, 4 GELU, 3 device-to-device memcpys.

The cost of a node was calibrated on this card by capturing a graph of N identical trivial
kernels and fitting replay time against N over N = 1…256:

```
fit: replay(N) = 1.886 + 0.7984 · N  µs        device duration of one trivial node: 775 ns
```

**Every kernel node costs ~0.8 µs whatever it computes.** 36 nodes is a 28.7 µs floor: 47% of
config 2's entire wall, 28% of config 12's, 0.45% of config 8's. On the launch-bound rows the
lever is not a faster kernel; it is fewer kernels.

**Predicate — `one_wave()`.** `amortizes()` (§3.1) asks a *bandwidth* question and correctly
declines every launch-bound row. `one_wave()` asks an *occupancy* question and fires exactly
where `amortizes` declines: when every thread block of the fused segment is resident at once
there is no second wave, so per-launch latency is pure overhead and collapsing five launches
into one is a structural win independent of bandwidth. Inputs are `multi_processor_count` and
`shared_memory_per_multiprocessor`, read at run time. On the announced matrix `one_wave`
selects **2, 3, 4, 12** and `amortizes` keeps **6, 7, 13**; the sets are disjoint and a test
asserts it rather than assuming it.

Three deletions produced the 16: v19's norm-fused megakernel under the new predicate (five
nodes per layer become one), the attention out-projection handed over in fp16 and widened
inside the kernel (bit-identical — it is an fp16 GEMM over fp16 operands, so the value is
already fp16), and a provably dead mask memcpy.

The count was **verified, not reasoned**: the first build measured 24 nodes, not the 20
predicted, because moving the residual add into the megakernel deleted the LayerNorm epilogue
that the attention output's `.float()` had been living inside, and the cast became four new
kernels. *Fusion relocates work; it also strands work that was only ever free because it was
riding along inside something else* (L45).

### 3.5 Streaming for the extreme shape — `bench/candidates/v33_streamed_long.py`

One predicate, imported unchanged from `v14_dispatch` so the two cannot drift: compare the
estimated working set against `torch.cuda.mem_get_info()` at the first forward. RESIDENT
takes the frontier path with graph capture; STREAMED slices the batch, runs the core per
slice, and skips compile and capture — because on a shape that is streamed *because* the
input barely fits, capture's `x.clone()` static buffer is the largest allocation in the
process and buys nothing.

Its value is entirely in the branch the measured set could not exercise. It is also a
**portability fix rather than a rescue**: without it the frontier plans a ~73 GiB resident
working set at config 14 and would fail that shape on an 80 GiB accelerator too, on
arithmetic that fits in about 4 GiB. §4.5 records what it costs on config 6, which is not
what its docstring predicts.

### 3.6 What holds it together

The candidate stack is compiled by Inductor for fusion and then captured in **our own** CUDA
graph rather than Inductor's, which reclaimed 22.5 µs per call of TorchDynamo cache lookup
(finding 16) — a fixed per-call cost, so the gain landed entirely on configs 3 (+87.3%) and 2
(+40.6%) and was invisible everywhere else. Capture is verified against a freshly computed
reference on every prime, because v12 could capture an **empty** graph and return a stale
buffer — right shape, right dtype, wrong values, and invisible to every accuracy check we had
(finding 17, §5.5).

---

## 4. The numbers

### 4.1 Per-configuration, milliseconds

All rows: `gpu_exclusive=True`, `padding_ratio=0.0`, `dtype=float32`, `input_scale=1.0`, 300
samples, median, arms isolated, clocks unlockable. Source: `bench/results.jsonl`.

| cfg | regime | eager | `torch.compile` | v26 | v34 | v36 (iso) | v36 (interleaved) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | mainstream | 1.6732 | 0.5857 | 0.2365 | 0.2510 | 0.2386 | **0.2243** |
| 2 | launch-bound | 1.7592 | 0.1352 | 0.0614 | 0.0471 | 0.0451 | 0.0563 |
| 3 | launch-bound | 1.7551 | 0.2468 | 0.0707 | 0.0594 | 0.0522 | 0.0666 |
| 4 | launch-bound | 1.6640 | 0.2191 | 0.1106 | 0.0922 | *0.3021* | **0.0819** |
| 5 | mainstream | 3.3270 | 1.2411 | 0.4434 | 0.4516 | 0.4342 | 0.4352 |
| 6 | throughput | 459.1217 | 216.5053 | **57.4372** | 62.0851 | 59.0756 | — |
| 7 | head_dim 8 | 1.6868 | 0.3348 | 0.0870 | 0.0850 | *0.3246* | **0.0778** |
| 8 | wide (head_dim 256) | 16.7086 | 14.4230 | 6.5485 | 7.0391 | 6.5761 | 6.5843 |
| 9 | head_dim 128 | 1.5012 | 0.4577 | 0.2386 | 0.2519 | 0.2253 | 0.2253 |
| 10 | head_dim 64 | 1.7295 | 0.5417 | 0.2447 | 0.2427 | 0.2458 | 0.2324 |
| 11 | head_dim 8, 16 heads | 7.3339 | 2.2630 | 0.2796 | 0.2775 | 0.2826 | 0.2703 |
| 12 | seq 32 | 1.6919 | 0.2058 | 0.1034 | 0.0850 | *0.1802* | **0.0891** |
| 13 | seq 1024 | 111.9078 | 35.4806 | 3.3014 | 3.2932 | 3.2983 | 3.3055 |
| 14 | extreme | — | — | — | — | — | — |
| | **total** | 611.86 | 272.64 | **69.16** | 74.26 | 71.28 | — |

*Italicised* v36 cells are the ones finding 45 identifies as wrong; see §4.3.

### 4.2 Speedups against the compiled baseline

| cfg | v26 | v34 | v36 (preferred arm) |
|---:|---:|---:|---:|
| 1 | 2.476 | 2.333 | 2.612 |
| 2 | 2.201 | 2.870 | 2.401 |
| 3 | **3.493** | **4.155** | **4.726** |
| 4 | 1.981 | 2.377 | 2.675 |
| 5 | 2.799 | 2.748 | 2.852 |
| 6 | **3.769** | **3.487** | **3.665** |
| 7 | **3.847** | **3.939** | **4.302** |
| 8 | 2.202 | 2.049 | 2.191 |
| 9 | 1.918 | 1.817 | 2.032 |
| 10 | 2.213 | 2.232 | 2.330 |
| 11 | **8.095** | **8.155** | **8.371** |
| 12 | 1.990 | 2.421 | 2.310 |
| 13 | **10.747** | **10.774** | **10.734** |
| **geomean** | **3.103** | **3.245** | **3.323** |
| **weighted_score** (cap 3.0) | **2.4129** | **2.4892** | **2.5287** |

**Bold rows are past the 3.0 cap and score nothing further.** They are configs 3, 6, 7, 11
and 13 in all three columns. Everything the weighted score can still see lives in the seven
uncapped rows, six of which are sub-millisecond — which is precisely where the measurement is
least trustworthy (§8).

Config 14 contributes **1.0** to every weighted score, the same as if we had never looked at
it. That understatement is deliberate (§7).

Against **eager**, the same frontier reads geomean 12.285x (v26) / 12.844x (v34) / 13.153x
(v36 preferred arm), and `weighted_score` saturates at 2.81–2.83 for every candidate from
generation 3 onward — which is why the objective was re-based on the compiled arm in the
first place (finding 12, L22). The eager rows come from a single sweep at `32bd7df2`; they
are one measurement of a fixed reference, not a per-run denominator.

### 4.3 Which timing arm, and why it matters

`bench/run_matrix.py` times the baseline arm, then **builds** the candidate —
`torch.compile`, Inductor autotuning, Triton JIT, all of which run the GPU hard — then times
the candidate. Finding 45 established that this systematically misreports any candidate that
does significant work at construction, which is most of what this project builds, because the
dispatch contract (§3) *requires* predicates be derived from measured device properties and
therefore run measurements at prime time:

```
cfg    isolated   interleaved   ratio
  4     0.3021       0.0819     3.69x
  7     0.3246       0.0778     4.17x
 12     0.1802       0.0891     2.02x
  9     0.2243       0.2243     1.00x
  8     6.5761       6.5843     1.00x
 13     3.2983       3.3055     1.00x
```

The three that disagree are exactly the configs where v36's predicate selects Triton at
several sites; the three that agree are where it selects nothing or where the work dwarfs the
planning. **Where an interleaved arm exists, this report uses it**, and says so.

The cost of that choice, stated plainly: **only `v36_gemm_gelu` (12 of 13 configs) and the
in-flight `v37_recombined2` carry an interleaved arm.** The ~600 rows that predate finding 42
do not. So the v36 column of §4.2 is not strictly comparable with the v26 and v34 columns — it
is a better measurement of a different candidate, and the honest reading is that **the ledger
cannot currently rank v36 against v34 on equal terms**. What supports v36 ≥ v34 is finding
43's dedicated ABBA comparison (parent and child resident, cold round discarded, min of four,
configs 8 and 13 running byte-identical code as an in-run control that put the floor at
±0.4%): no regression on any config, and configs 1, 9, 10 and 5 — three of which have
identical GEMM shapes — landing at +5.0%, +5.0%, +4.9% and +5.0%, agreeing to 0.15 percentage
points. That agreement is the check that the protocol is not measuring itself.

Finding 43 estimated v36's weighted score as **2.571**, by applying those ABBA ratios to v34's
swept speedups. Recomputing from v36's own ledger rows gives **2.5287**. Both derivations are
stated because they are different quantities and neither is wrong; the smaller one is the
measured one and is what §1 quotes.

### 4.4 Correctness margins

`max_abs` against the fp32 reference as a percentage of the 2e-3 absolute budget, with **0
failed elements on every configuration** at the OR-combined locked tolerance:

| candidate | range across 13 configs | worst config |
|---|---|---|
| v26_causal_correct | 60.9% – 78.6% | cfg 9 |
| v34_launch_bound | 62.5% – 78.6% | cfg 9 |
| v36_gemm_gelu | 62.5% – 74.1% | cfg 6 |

Three things about this table matter more than the numbers.

**`max_abs` alone does not predict pass/fail.** The gate is OR: an element passes if
`|diff| <= atol` *or* `|diff| <= rtol·|ref|`. Finding 08 recorded a candidate passing at
115.9% of the absolute budget and another failing at 100.7%, because the joint distribution of
error and reference magnitude decides it. Judge by `failed_elements`.

**Roughly 40% of the budget is spent before we run.** Finding 40 measured the *reference
itself* under TF32 at 8.086e-4 from an exact fp64 evaluation — 40% of the 2e-3 budget, flat
across a 32x change in sequence length, which is the signature of a representation floor
rather than accumulated error. That is a fact about the mandated baseline setting, not about
our kernels, and it is why the same 8–9e-4 keeps appearing on unrelated configs.

**Margin is not a fixed property.** At `input_scale=0.01` — a flag the benchmark exposes —
**every candidate fails**, including a pure-fp32 one, at 2.38e-3 against a 2.0e-3 budget
(finding 19). LayerNorm normalises the input scale away, so this is not a small-signal
artifact; it is flash attention's online softmax accumulating differently, amplified ~2.5x by
`eps` becoming ~10% of the input variance. A candidate at 78% of budget and one at 30% are not
equally correct, and only the second survives that shift.

### 4.5 What "the frontier" is right now, and one unexplained regression

The project has two live lines off `v26_causal_correct`, rejoined twice:

```
v26 ─┬─ v33 (streaming) ─── v35 ────┐
     └─ v34 (launch floor) ─ v36 (proj GEMM) ─ v37   ← merge: v36 parent, v35 contributor
```

`v37_recombined2` is the only candidate that simultaneously returns the right shape when a
model is called at a second shape, can compute a shape too large to hold resident, re-derives
mask state on a shape change, and carries v36's projection GEMMs. On capability grounds it is
what we would submit. Finding 44 measured it against v34, v35 and v36 on eight configurations
and found it a null against v36, which is what it should be — in the steady state at one input
shape they execute the same code.

**Configs 5, 6, 7, 11, 13 and 14 were not in that comparison, and the ledger now says config 6
is a problem.** Recomputed for this report:

| candidate | cfg 6 candidate arm (ms) | cfg 6 eager-baseline arm, same row (ms) | commit |
|---|---:|---:|---|
| v26_causal_correct | **57.437** | 441.251 | bd4d9ce2 |
| v33_streamed_long | **92.178** | 447.399 | 52c6640a |
| v34_launch_bound | 62.085 | 476.630 | 52c6640a |
| v36_gemm_gelu | **59.076** | 463.853 | 690c2f1a |
| v37_recombined2 | **90.974** | 443.889 | 58487d53 |

The candidate arm splits cleanly into two clusters — ~57–62 ms for the non-streaming line and
~91–92 ms for the streaming line — a **1.54–1.60x regression**, reproduced across two
independent sweeps, at two different commits, over two different parents. The eager-baseline
arm on the streaming rows (447.4 and 443.9 ms) sits within 1.5% of v26's row (441.3 ms), so
this is not ambient drift; the baseline arm's full spread across all five rows is 441–477 ms
(8%), still far below the 55% being reported.

`v33_streamed_long`'s own docstring states that streaming "changes nothing on the 13 configs
that fit (`choose()` returns resident)". **On config 6 the ledger contradicts that**, and
config 6 is 83% of the matrix wall. We have not diagnosed it — the obvious hypothesis is that
the working-set estimate against `mem_get_info()` selects the streamed path at 1.28M tokens,
which would mean the streamed path is taken and costs ~1.55x, but neither the row nor any
finding records which branch fired. This is stated as an **open defect**, not a result. It is
the single largest unresolved risk in the submission and the first thing a reviewer should
check.

Consequently:

* the fully-swept, ledger-recorded best on the isolated arm is **`v34_launch_bound`**,
  2.4892 / 3.245x;
* the best measurement taken with a protocol we trust is **`v36_gemm_gelu`**, 2.5287 /
  3.323x, on 12 of 13 configs;
* **`v37_recombined2` is the capability superset and is not yet defensible on speed**, because
  its only swept config 6 number is 1.54x worse than its parent's.

---

## 5. Method: an evolutionary search over commits

### 5.1 The representation

A candidate is a **git commit**. Lineage is ancestry. Clade metaproductivity is forward
reachability. There is no second tree store. This buys four things a bespoke tree would have
to reimplement: reproducibility is `git checkout <sha>`; merges are expressible as real
two-parent commits (recombination is where a lot of evolutionary value lives, and a
single-parent representation cannot express it at all); distribution across machines is free;
and CMP is `git rev-list --parents`, computed once.

The rules that representation imposes: a measurement is keyed to `(commit_sha, config_id)`
and measurement code lives **in-tree** so the sha describes the code that ran; a dirty tree is
recorded and *barred from clade statistics*, because a sha that does not describe the code
that ran is a false provenance claim; and a candidate branch is **never** rebased, squashed,
amended or force-pushed, because rewriting history silently reparents the tree and invalidates
every statistic derived from it.

38 candidates across 37 generations, on 31 `cand/*` branches. `bench/candidates/__init__.py`
is the registry and carries each candidate's declared parent and a summary of what it changed.

### 5.2 Selection: clade metaproductivity and Thompson sampling

We adopt the Huxley-Gödel Machine's correction to Darwin-Gödel-style selection: a node's *own*
score is a poor estimate of its value *as an ancestor*. What predicts future payoff is the
pooled outcome of its entire descendant subtree. Each node carries a
`Beta(1 + successes, 1 + failures)` posterior over its descendants' ledger rows, and the next
parent to expand is drawn by Thompson sampling — which spends most draws on productive clades
while still occasionally expanding an unpromising one, with no exploration temperature to tune.

**The success criterion took two attempts and the obvious fix was worse than the bug**
(finding 21). Originally a row counted as a clade success at `speedup > 1.0` against *eager*:
88.1% of rows succeeded, the Beta posterior was nearly flat, and the sampler that decides where
every expansion attaches was barely discriminating between lineages at all. Swapping in the
compiled baseline gives a healthier-looking 70.6% success rate and a clade ranking that
correlates with **commit age at ρ = +0.660** — considerably *worse* than the +0.269 it
replaced, because every late commit clears the compiled bar and old nodes accumulate credit for
wins their descendants merely inherited.

| criterion | success rate | ρ(clade rank, commit age) |
|---|---:|---:|
| `speedup > 1.0` vs eager | 88.1% | +0.269 |
| beats the compiled baseline | 70.6% | **+0.660** |
| improves on nearest ancestor | 19.2% | −0.192 |
| **both (adopted)** | **15.4%** | **−0.130** |

A success must now clear both bars: beat the compiled baseline for its config, *and* improve on
the nearest ancestor commit that measured the same config by more than the measured ±7% noise
floor. A descendant that merely carries its parent's win forward scores nothing — which is the
entire point of metaproductivity, since the parent already holds that credit.

A related trap that is defensible but live: clade success is counted **per row, not per
candidate**, so a candidate rejected on correctness can carry the highest clade score because
most of its config rows passed. That is right in HGM terms — productivity is not promotion —
but a high clade score never means "this works".

### 5.3 Pricing ideas: quality as prior mean, novelty as prior strength

With one GPU, a queue ranked purely by expected speedup fails in a specific way: it queues five
plausible variants of the same mechanism and burns a day proving they are within noise of each
other. `specs/07-proposal-rubric.md` scores each proposal on two axes of five dimensions —
quality (mechanism specificity, roofline-grounded headroom, time-to-signal, feasibility on this
device, stacking with the frontier) and entropy (mechanism distance from the measured archive,
information gained *if it fails*, source diversity, regime coverage, kernel-level depth).

They are deliberately **not** combined as a weighted sum, which would make "interesting" and
"promising" substitutes and make the weight a magic number nobody can defend. Instead quality
sets the *mean* of a Beta prior and entropy sets its *strength*, inverted: a well-evidenced
obvious idea gets a narrow posterior centred high, sampled early and abandoned fast if it
disappoints; a genuinely novel idea gets a wide posterior that Thompson sampling occasionally
draws even at a mediocre mean. No temperature, no exploration bonus.

**It was backtested against our own history before it was trusted** (finding 20,
`bench/proposals/backtest.py`, no GPU), by scoring 12 already-measured candidates using *only*
each candidate's docstring as written at the commit that introduced it, recovered with
`git show`. That exposed three defects:

1. A degenerate Beta at Q = 1.0 (`beta_0 = 0` raises). Found by running it, not by reading the
   formula.
2. **It was scoring the wrong quantity.** Rank correlation against cumulative
   `geomean_vs_compiled` was **ρ = +0.050** — no predictive power at all, and structurally so,
   because a cumulative number measures the stack a candidate *inherits*. Re-targeting to
   marginal gain over parent raised it to **+0.267** with no change to any score.
3. **A framing exploit.** v9a and v9b make the identical move and realised +58.3% and +56.9%;
   the modestly-worded one scored **20 points lower**. A rubric that can be talked down by
   modest framing can be talked up by grandiose framing — a direct exploit for any agent
   scoring its own work. Scoring the mechanism rather than the presentation took ρ from +0.267
   to **+0.483**.

**Stated honestly: +0.483 is not significant.** For n = 9 the two-tailed 5% critical value is
about 0.68. The defensible claim is that the rubric is not obviously broken and is free of
three specific defects it demonstrably had an hour earlier. It has not been shown to work.

### 5.4 Economics: screen, then confirm

Under a ±7% noise floor, a confident full-matrix verdict costs minutes, and ideas arrive faster
than that. Evaluation is two-stage: a 30-second screen over four configurations spanning four
regimes, then a full recorded sweep only for survivors. Measured end to end at **29.6 s against
112 s, a 3.8x saving**.

The screen set was *derived* from the 411 rows already in the ledger at zero GPU cost — config
6 alone is 48.5 s of a 112 s sweep, so any subset containing it is not a screen at all. Screen
results never enter the ledger, because partial sweeps would swamp the clade statistics that
full sweeps feed. **The screen's job is to kill what is clearly bad, not to rank what is
statistically tied**: every subset tested retains the true top-1 candidate and none retains the
true top-3, but the top five span a 4% spread *inside* the noise floor, so nothing separates
them and a screen should not pretend to.

### 5.5 What worked, and what did not

**The centrepiece of what did not work is finding 28: the tree was a chain.**

`bench/README.md` states the premise plainly — git branches are the evolutionary tree.
Measured, at generation 18:

```
candidate                    git-ancestors that are candidates    declared parent
v9a_compiled_core                          8                      v8_padfast
v13_safe_capture                          12                      v12_graph_over_compile
v17_dispatched_megakernel                 16                      v13_safe_capture
v18_capture_insurance                     17                      v17
```

**Every candidate had exactly `generation − 1` git ancestors. A perfectly linear chain, for
eighteen generations.** The cause was the branching discipline: each candidate branch was cut
from the trunk's tip so it would inherit the latest harness, and every candidate is merged back
into the trunk — so cutting from the trunk inherits every earlier candidate and the topology
collapses however the branches are named. The spurs in `git log --graph` are decorative;
`merge-base --is-ancestor` says it is a line.

**On a chain, CMP measures age, not productivity** — a node's clade is simply "everything
committed after it". That is exactly what L1 said on day one, in this repository, with the
instruction "branch first". It was implemented in a form that satisfied the words and none of
the mechanism, and finding 21's fix to the clade *criterion* then masked it by pushing the age
correlation to −0.158.

The measured impact was, so far, harmless: CMP over declared lineage and CMP over git ancestry
produce **the same top three nodes, reordered**, so no expansion went to the wrong place. That
is luck, not design, and the dilution grows every generation.

The fix has two halves. CMP now reads the registry's *declared* parents. And
`tests/bench/test_lineage_topology.py` asserts, from generation 19, that a candidate's
candidate-ancestors equal its declared ancestors — i.e. that it was cut from its parent, not
from the trunk. Generations ≤ 18 are grandfathered, because the topology cannot be corrected
without rewriting history, which the rules forbid.

The general lesson, and the reason this is the honest centrepiece: **every structural claim
needs an executable check.** The claims that had one — the oracle manifest, the append-only
ledger, the tolerance lock — have never silently broken. The two that lived only in prose —
"git is the tree", and the premises written into `matrix.py` — were *both* found false by a
human looking, not by the system noticing.

Two related failures with the same shape, all within three days:

* **A test can pass because its subject was never built** (finding 24, L36). A lineage
  invariant sweep reported 113 green while four candidates carried a live silent-wrong-answer
  bug. Dynamo's `cache_size_limit` is 8 and shared per process; once exhausted `torch.compile`
  silently falls back to **eager**, which allocates a fresh output every call — so a
  static-buffer test passes *because the candidate was never compiled*. Green was produced by a
  second defect. Running one candidate per process turned 50 passed / 1 failed into three
  genuine failures.
* **A guard can pass because its sensor saw nothing** (finding 26, L38). The contention detector
  built on `nvidia-smi` reported a live CUDA process on one trial and nothing on an identical
  trial seven seconds later, with the same process confirmed alive. Under WSL2 a clean report
  from it means nothing. The lock file itself works and is tested; the foreign-process check
  does not. The consequence was concrete: one sweep overlapped a research agent benchmarking
  Triton kernels on the same GPU, and that sweep's quantitative conclusion had to be marked
  provisional and re-measured.

And the two things that most reliably *did* work:

* **The audit rule went seven for seven.** Ask of every result: *what does this depend on that
  we never varied?* Padding ratio, eager baseline, dtype, input scale, allocation context,
  process contention, causal flag. Each was found by asking, and **not one was found by the
  search loop.** Two of the seven were large enough to change the headline. The sharpest form is
  E2 in `docs/loop/method.md`: **test the setting the harness DEFAULTS to, not the one the
  specification implies.** Every candidate from generation 5 to 23 hardcoded `is_causal=True`
  and returned three quarters of its output wrong on a non-causal input — max_abs 1.67e+00
  against a 2e-3 tolerance — with all 177 tests green, because every announced config is causal
  and the reference's own default is `causal=False`.
* **Ablation, run on principle rather than on suspicion** (finding 15, L17). An evolutionary
  loop only ever *adds*: nine generations stacked nine justifications, each valid when written,
  and nothing in the loop's design would revisit one after the world changed underneath it.
  Forking the frontier into one-mechanism-removed siblings found that L2-sized batch chunking —
  real when v3 added it, taking config 6 from 3.21x to 5.72x — had become dead weight, costing
  **−0.3% on the exact config it was built for**, because Inductor now manages the working set.
  The lean frontier measured 2.514x against the fat one's 2.678x, inside the floor, with one
  fewer component. Nothing in the loop's own signal would have flagged it: it was passing,
  correct, and sitting inside the best candidate.

The search itself, when first run on the parametric level, **found noise and said so**
(finding 06). It reported a 2.7% improvement over its seed; the space turned out to be
degenerate — two parameters that only ever appear as a quotient — so the "best" point was the
seed wearing different coordinates, and the accidental replicates it produced measured the
run-to-run spread at 1.4–3.0%. Re-run with the axes collapsed and a promotion margin, it
evaluated 10 of the 12 points the space contains and **promoted nothing**: the peak sits exactly
where the analytic L2 derivation put it. That is a null result for the search and a positive
result for the calibration, and it is the ablation that any claim of the form "evolutionary
machinery produced the win" ought to be asked for.

---

## 6. Negative results

These cost real GPU time and are, in our view, the most reusable part of the project. Each is
recorded with its method, its positive controls, and what would falsify it.

### 6.1 fp16 MMA accumulation — a 21.4x scissors (finding 30)

Three research agents independently proposed it, and the hardware reading is correct, verified
from generated PTX rather than assumed: `tl.dot(out_dtype=tl.float16)` really emits
`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` against the `...f32.f16.f16.f32` form, and
in a loop of 1024 `mma` instructions the f16 form measures **1.62x** faster (38.2 → 23.6 µs). A
test pins that, so if it ever stops holding this finding is closing the wrong thing.

It is worth nothing, for two independent reasons that meet in one variable. Both conditions
depend on the contraction depth K, and in this architecture `K == d_model == ffn_dim`, so one
shape parameter drives both — in **opposite directions**:

* **Fast** requires being above the ridge point. Intensity is linear in `d_model`, so this
  device needs `d_model ≥ 359`.
* **Accurate** requires `eps_fp16 · sqrt(K) ≤ atol`, which at the locked 2e-3 and unit output
  magnitude needs `K ≤ 16.8`.

**A 21.4x gap, and it widens on better hardware** — a higher peak-FLOPs-to-bandwidth ratio
pushes the ridge up while fp16's mantissa stays at 11 bits. Both bounds are computed from
measured device properties in `ffn_accum.no_shape_satisfies_both()`, so the claim is checkable
on a card nobody here has seen.

Measured confirmation from both ends. At config 6's token count the fused FFN runs at
**99.2–100.1% of measured bandwidth**, and all four accumulator arms measure 1.000x, 1.001x,
0.998x, 1.000x — not a win inside the noise floor; no win. And at **K = 16, the shallowest
contraction sm_89's `m16n8k16` can issue**, an fp16 accumulator already spends **140% of the
tolerance budget**:

| K | fp32 accumulator | fp16 accumulator | % of the 2e-3 budget |
|---:|---:|---:|---:|
| 16 | 9.54e-07 | 2.800e-03 | **140.0%** |
| 32 | 1.43e-06 | 2.941e-03 | 147.0% |
| 128 | 4.29e-06 | 6.227e-03 | 311.4% |
| 256 | 4.53e-06 | 7.485e-03 | 374.3% |

The region is not narrow. It is empty. The candidate ships with the predicate declining on
every shape, making it numerically identical to its parent (pinned by `torch.equal`), and
carries a `RATCHET_FORCE_ACCUM` escape so the refused path can be re-measured rather than
re-argued. Worth noting what "passes" is worth here: the single-site arms pass every config and
cost **15 to 34 percentage points of tolerance budget for a measured 1.001x**. Passing was never
the bar.

**The transferable move, which costs no GPU time:** before building, check whether a mechanism's
speed condition and its accuracy condition are functions of the *same* shape parameter. If they
are and they point opposite ways, the question is not "does this help" but "is the window
non-empty", and that is arithmetic on measured device properties.

*A note on how easily this went green and vacuous:* the first version of the margin report
showed all four arms producing **byte-identical** numbers on every config, which looked like a
clean null. The FFN amortization predicate was declining the fused kernel at small token counts,
so the forced accumulator never ran. A falsifier that cannot distinguish "the mechanism is
harmless" from "the mechanism did not run" is not a falsifier.

### 6.2 L2 weight persistence — 94.8% of the prize was already being collected (finding 33)

The proposal argued that config 6's 327 MB activation stream evicts the fp16 weight cache under
LRU, priced at up to 23% of config 6. `ncu` is unavailable here (`ERR_NVGPUCTRPERM` — WSL2
denies GPU performance counters and the fix is a modprobe option on the *host* driver), so the
falsifier the proposal itself named could not be read. The substitute is a contrast, not a
roofline: the same kernel plus one per-program offset into a weight arena, so the *only*
difference between arms is how many distinct weight copies the grid touches, and the delta
between two arms **is** weight DRAM traffic.

| arm | ms |
|---|---:|
| activation-only floor (same three streams, no arithmetic) | 2.647 |
| the real kernel, one 64 KiB weight pair | **2.702** |
| 1 MiB weight arena | 2.706 |
| **32 MiB weight arena — forced to miss** | **4.728** |
| the real kernel + a persisting window over the weights | 2.709 |
| 32 MiB arena + a persisting window over the arena | **2.707** |

The real kernel runs **2.1% above a pure-streaming floor measured on its own activation
traffic**. Full-miss weight traffic would be 1.311 GB = 2.136 ms at 613.7 GB/s; measured
cold − hot is 2.026 ms, so **the kernel is already saving 94.8% of the theoretical worst case**.
Installing the window moves it **−0.25%**.

**Both positive controls fire, and that is what makes the null usable.** The 32 MiB arm proves
the contrast can see weight traffic at all (+75%); the same arm *plus* a window goes
4.728 → 2.707 ms, **+42.7%**, proving the ctypes shim, the driver path and sm_89 all work.
*"The feature works and has nothing to do"* is a much stronger claim than *"we measured
nothing"*, and it costs one extra arm.

**The bound that should have preceded the probe.** A cache optimization's ceiling is the
*re-fetched* traffic, never the total — the first read of a byte is compulsory and survives any
policy. The whole four-layer weight set at `d_model = 128` is 768 KiB, 0.002% of what config 6
moves. And the reuse distance was one line: a weight line is re-referenced once per CTA, a CTA
touches 80 KB of activation, so the tile is touched ~600 times per sweep of a 48 MiB L2 and is
never the LRU victim. The probe's measured crossover confirms it — resident at a 1 MiB arena,
evicted at 32 MiB, exactly where reuse distance crosses cache size.

**Writing that condition as a test immediately found the row it does not cover.** Config 8's
four-layer weight set is **48.00 MiB, equal to this card's L2 to the byte** — the one announced
shape where eviction is genuinely possible. Prose would have shipped the generalization.
Measured directly: a window over the whole arena is **−0.62%**, because config 8's GEMMs run at
**93–98.5% of measured tensor-core peak**, 3.8–5.4x above their bandwidth floor. A memory-system
policy has no time to give back there however the cache behaves.

The same holds for cuBLAS's QKV projection at config 6: **+2.2% over its activation floor against
a +75% worst case.** Nothing in config 6 is paying for weight traffic. This also disposes of the
recurring intuition that produced v3's L2-sized chunking, which the g10 ablation had already
killed.

### 6.3 LayerNorm fusion — the compiler is already there (findings 10, 29)

Two attempts, both negative, for different reasons.

**Folding the fp16 downcast into the norm's epilogue** (finding 10) measured **+2.0% geomean,
below the noise floor**, and **failed config 6 outright** at `max_abs` 2.013e-3 against the
2.0e-3 budget. Seven different configs landed on *exactly* 1.9384026527404785e-3 — not seven
coincidences, but fp16 rounding hitting a fixed representation limit independent of shape. The
change does not add error proportional to work; it adds a floor, and there is no version of it to
tune toward. Correctness is a gate, not a term in the objective: the candidate is nominally
faster and scores **nothing**.

**Folding the residual add and both norms into a megakernel** (finding 29) measured **+0.8%
geomean, inside the floor**, and identical total wall time. Config 13 gained a genuine 8.0%;
config 6 — 84% of wall — was flat. The reason is that the nine LayerNorm kernels were already at
the roofline (finding 29 measured them at **661–672 GB/s**, above the card's 613.7 GB/s streaming
figure because part of the traffic is served by L2 and the accounting is logical bytes) *and*
Inductor had already fused the residual add and the downcast into each, so nine is the fused
count, not the naive one. There is nothing left to win by better fusing; only by making them not
exist — which is what §3.4 eventually did, and it paid on the launch-bound rows for the
*launch-count* reason, not the bandwidth reason.

*This report cannot substantiate a "594.9 GB/s against a 592.7 GB/s memcpy ceiling" figure for
the pointwise bucket; no such numbers appear anywhere in the repository. The measured figures are
the 661–672 GB/s above.*

Finding 29 is also where two probes were wrong in opposite directions, by hazards this repository
had already documented: an op-level probe said **3.84x better** because it compared against
`F.layer_norm` and the kernel called separately in eager — a decomposition the real system never
executes, so isolation did not merely shrink the effect, it *invented* one — and a model-level
probe said **16.2% worse** because it held both models in one process, recreating the
co-residency spill of finding 05. The harness said +0.4%. **A probe may propose; it may never
conclude.**

### 6.4 SDPA backend selection — the premise was false (findings 04, 23, 36)

`bench/matrix.py`'s single source of truth asserted for a week that "cuDNN and FlashAttention
typically support {32, 64, 128, 256} and may silently fall back to a slow path" at head_dim 8,
and that this was "possibly the one place a hand-written Triton kernel genuinely earns its keep".
It was cited into the proposal rubric, into three research-agent briefs, and into every discussion
of where to look next. **It was never checked, and it is false.** Measured, in 40 seconds:

| head_dim | flash | mem-efficient | cuDNN | math |
|---:|---|---|---|---|
| 8 | ok | ok | ok | ok |
| 16–128 | ok | ok | ok | ok |
| 256 | ok | ok | **refuses** | ok |

Every backend accepts head_dim 8. The one refusal in the whole matrix is cuDNN at head_dim
**256** — the opposite end of the range, landing on config 8, a config nobody was watching for
backend reasons.

Two adjacent results:

* **A dispatching API turns a performance bug into a silent one** (finding 04). v1 called
  `scaled_dot_product_attention` and we assumed we were getting FlashAttention. It never did, on
  any of the 14 configs: the backend actually selected was the fp32 memory-efficient CUTLASS path
  `fmha_cutlassF_f32_aligned_64x64_rf_sm80`, because v1 cast q/k/v back to fp32 and forwarded an
  all-True mask, either of which alone disqualifies flash. There is no warning and no fallback
  notice; the code looked right, passed every correctness check, and posted a respectable number.
  The only reliable check is to ask *which kernel actually ran*. Correcting it was worth 4.65x to
  9.59x on the isolated attention call.
* **A hand-written kernel for head_dim 8 is real and small** (finding 36). PyTorch's bundled
  FlashAttention-2 has no head_dim 8 kernel — `HEADDIM_SWITCH` rounds anything ≤ 32 up to 32, so
  the vendor contracts over 32 lanes where 8 carry data. Our kernel pads to Triton's
  `min_dot_size` floor of 16 instead. Measured **1.40x at the op**, against a source-code argument
  that predicted ~4x, because these shapes are nowhere near mma-bound. Diluted end-to-end it is
  ~1.24x on config 7 and ~+3% on the geomean, inside the floor — and config 11 already sits above
  the 3.0 cap, so its entire gain scores **zero**.

That second one also produced the sharpest single warning in the project about hand-written
kernels inside a compiled region: the first version measured **2.18x SLOWER** on config 7 with
every correctness test green, `graph_verified True`, and a successful capture. Attention was
1.63x faster; what was lost was Inductor. The launch wrapper resolved its tile plan at the call
site, so a `try/except` import and a `get_device_capability()` call executed inside Dynamo's
traced region, Dynamo dropped the frame to eager, and our CUDA graph faithfully captured an
*eager* op sequence — nine ATen LayerNorm kernels at 151 µs where Inductor's fused ones cost ~13.
**Audit what a hand-written kernel's launch wrapper does, not only what the kernel does.**

### 6.5 The 68-SM veto — real, correctly diagnosed, worth nothing (findings 22, 43)

`torch/_inductor/utils.py` contains `min_sms = 16 if device.type == "xpu" else 68  # 3080`. This
card has 66 SMs, so `is_big_gpu()` is False, which gates `use_triton_template`, which gates
whether Inductor may emit a Triton GEMM template — and without a template node there is no
epilogue fusion into a GEMM. All verified in our own install. Lifting it works: the profiler
shows the template kernels appear.

An isolated probe promised **1.58x** (5.576 → 3.537 ms on a bare FFN pattern). In the real model
it is worth **−1.4% on the geomean**, later re-measured on an idle GPU at −0.8%.

**Why the probe lied is the finding.** The elementwise work the GEMM epilogue would have absorbed
was *already* being absorbed by Inductor's pointwise fuser, into the LayerNorm kernels — the veto
gates `use_triton_template` only, and `max_autotune_pointwise` was never vetoed. Lifting it
**moved work between kernels** rather than eliminating it. The probe showed 1.58x precisely
*because* it was isolated: a bare FFN has no LayerNorm to fuse into, so the template is the only
mechanism available.

Two honest amendments to that finding, both recorded rather than quietly absorbed:

* A claim that the veto was "the likeliest single explanation for the generation 11–14 plateau"
  was **retracted**. Removing it changes nothing measurable, so it explains nothing. The plateau
  remains unexplained.
* Finding 22's headline was **too broad** (corrected in finding 43). It was right about LayerNorm
  and wrong about GELU: GELU has no LayerNorm to hide in and was its own kernel on every layer of
  every config — which is the entire prize §3.3 eventually collected. Worse, the per-config rows
  had said so all along, at two separate commits, reproducing to 0.5%: configs 9, 10 and 12 were
  4.8–6.5% *better* under the lift while config 5 was 5.8–6.5% worse. **The aggregate null hid the
  shape, and the signal sat in the ledger for eleven generations.**

Also worth carrying: two research agents with disjoint territories independently found
`min_sms = 68` and independently proposed lifting it, and one ranked it top. That made the
diagnosis look strong, and the diagnosis *was* strong — but it said nothing about the fix's worth,
because both were reading the same source file and neither had run the full model. **Agreement
between analysts sharing a method is not replication.**

### 6.6 Two more, briefly

* **The fp16 residual stream** (finding 08). 1.4x faster on the launch-bound configs and **fails
  11 of 13** at 3.3–5.3x over budget. The residual *accumulates* across layers; an elementwise op
  does not. That distinction shaped precision placement for every later candidate — and its
  immediate corollary, that the GELU round-trip could be removed, was **bit-identical on all 13
  configs and 5–12% faster on 12 of them**, because PyTorch's fp16 GELU already computes in fp32
  internally and rounds once on write. *An explicit upcast around a library op may be buying
  nothing.*
* **The output-copy elimination and the out-projection epilogue** (findings 35, 37, 38). Each
  removed a real cost and none reached the frontier, mostly because the screen set contains no
  shape where they can win. Finding 38's two screens both said REJECT, disagreed with each other,
  and neither was about the candidate: each run had exactly one wild sub-millisecond row and it
  was a different row each time (config 10 at 3.6x the parent in one run, config 2 at 1.9x in the
  other, on identical code). A screen verdict on a candidate whose prize is ~1.3% inside a ±7%
  floor is a coin flip dressed as a decision.

---

## 7. Config 14: no speedup, and a correctness certificate

Config 14 is `B=32, S=100000, d_model=1024, heads=16, layers=2, causal`. It produced 28 ledger
rows and no information before finding 40 — 27 of them `status="oom"` with a truncated traceback
— and it is 24.3% of the remaining score headroom.

### What cannot be claimed

**No speedup, and there cannot be one.** The reference's `BaselineSelfAttention.forward`
materialises `scores = q @ kᵀ` of shape `[B, H, S, S]`. At this configuration that single tensor
is **18.63 TiB**, and the next three lines each produce another of the same shape. It is not an
estimate; it is one allocation on one line of the reference's own source, and it was confirmed by
asking the driver:

| what | bytes | result |
|---|---:|---|
| the whole batch `[32,16,100000,100000]` | 19073.49 GiB | refused |
| one sequence `[1,16,100000,100000]` | 596.05 GiB | refused |
| **one head of one sequence `[1,1,100000,100000]`** | **37.25 GiB** | **refused** |

That last row settles it: this is not a batch-size problem. A single attention head of a single
sequence needs 2.3x this card's entire memory, and the reference needs 512 of them. The reference
tops out between S=12288 and S=16384 at batch size **one**. No GPU or single node in 2026 has
18.63 TiB of accessible memory.

A ratio needs two measured times and **the denominator does not exist**. Timing our own slower
reimplementation of the baseline and dividing by it would be a number about us, not about the
reference. `timing.speedup` is `None` and stays `None`, and the configuration therefore
contributes **1.0** to `weighted_score` — exactly as if we had never looked. That understatement
is deliberate; putting a manufactured 3.0 into our own headline is precisely the error findings
12 and 21 were written about. A grader who credits "the baseline produces no output and this
produces a verified one" will score it higher; that is the grader's judgement to make, not ours
to pre-book.

Nor is the configuration runnable end to end on this card by anything. `forward(x) -> y`, both
`[B, S, d_model]` fp32, is **12.21 GiB in and 12.21 GiB out = 24.42 GiB** of tensors that no
optimisation removes, against 15.99 GiB of VRAM. (Returning a mutated view of the input would
remove one and is not available: it corrupts the caller's tensor and gives a wrong answer on the
second call with the same buffer.) On an 80 GiB card this floor clears easily — and the 18.63 TiB
impossibility still stands there.

### What is established, and how

**The computation runs**, one sequence at a time, at the announced shape: **32 of 32 sequences,
3,200,000 tokens at S=100000, peak device memory 3.54 GiB.** Ledger row
`status="reference_infeasible"` at commit `975c69b`, clean tree, GPU lock held.

**Correctness is checked at the real sequence length, by two independent oracles**
(`bench/feasibility.py`), not at proxy shapes.

*Oracle A — the causal-prefix theorem.* Under causal masking with an all-valid token mask, every
operation in the reference is either position-wise or attends strictly backwards, so for any
`P ≤ S`, `model(x[:, :P])` equals `model(x)[:, :P]` in exact arithmetic. So: run the candidate on
the full 100,000-token input, run the **unmodified reference** on the first P tokens of that same
input, and compare with the harness's own `compare_outputs` at the locked tolerance. Result:
**passed at P=4096, max_abs 8.658e-04, 0 failed elements**, covering rows 0..4095. The theorem's
own slack, measured on the reference against itself, is 3.881e-04 (a TF32 GEMM reduces differently
over a different K extent) — reported, not absorbed. The negative control matters as much as the
result: the same P rows taken as a *suffix* give 1.841e+00, so the comparison is capable of
failing. P is derived from the same feasibility predicate against measured free memory; a
hardcoded prefix length would be the config-id branch this project forbids, in a different
costume.

*Oracle B — a blocked fp64 evaluation of the reference's own arithmetic.* The query axis is
blocked, which is exact because softmax reduces along the key axis. It is deliberately **not**
online/streaming softmax — it does max-subtract-exponentiate-normalise in the same three steps and
the same order as the reference, so an online-rescaling bug in the candidate cannot be mirrored by
an oracle that has no online rescaling. Validated where the reference runs: in strict fp32 it
reproduces the reference to ~2e-6, 0.1% of the budget. Negative controls all fire (causal vs
non-causal 1.371e+00; one bias perturbed by 0.05 gives 2.354e-03; query blocking 64 vs 512 gives
< 1e-12, which is the exactness of the blocking itself).

The certificate is a triangle inequality. `|ref − oracle|` under TF32 is 8.086e-04 and flat in S,
so a sufficient condition for passing is `max|candidate − oracle| ≤ 2.0e-3 − 8.09e-4 = 1.19e-3`.
Measured at the announced sequence length, every query row including the last:

```
max |candidate − fp64 oracle|  at B=1, S=100000  =  8.0913e-04     (525 s of fp64, 6.37 GiB)
threshold                                        =  1.191e-03      CERTIFIES
⇒ |candidate − reference| ≤ 8.091e-04 + 8.086e-04 = 1.618e-03  <  2.0e-3
```

The figure reproduces to every digit across two different processes with different query block
sizes (`0.000809131791957074` both times).

**The candidate is as close to the exact answer as the reference implementation is** —
8.0913e-04 against 8.086e-04, three digits apart — because both are dominated by TF32's 10-bit
mantissa on outputs of magnitude ~0.8. With fp16 intermediates and flash attention over 100,000
keys, our own arithmetic contributes almost nothing on top of the floor the baseline already sits
on.

**The gaps in that argument, stated plainly:**

* `|ref − oracle|` at S=100000 is not measurable, by construction — the reference cannot run
  there. It is measured at S ≤ 4096 and is *identical* at S=128, 512, 1024 and 4096, which is good
  evidence of a representation floor rather than a growing error, and the candidate's own figure
  moves only +2% between S=32768 and S=100000, which is the same flat-in-S signature. **But it is
  an extrapolation, and the conclusion `< 2e-3` inherits that.**
* The fp64 oracle was run on **one** sequence. Batch elements are independent by construction and
  that independence is tested — with a negative control against a rolled batch — but 31 of the 32
  sequences are covered by the argument rather than by the oracle.
* The full-batch call, run last because it poisons the allocator, **fails** after 4.5 s with
  30.6 GiB reserved, recording `stream_path: "streamed"`. The dispatch fired, the working set
  stopped being the binding constraint, and the two tensors the signature requires still do not
  fit. A grader on a 32 GiB card would see this succeed.

*A method note worth carrying.* The oracle's first three attempts failed, and not for the reason
they appeared to. Truncating the key axis at the causal diagonal halves the arithmetic and is
obviously right — and makes every loop iteration a different allocation size, ~1500 distinct
multi-hundred-MB tiles the caching allocator can never reuse. It failed with a **driver** OOM at
S=32768 with 14.18 GiB free, which carries no PyTorch allocation breakdown and poisons the CUDA
context so that everything after it fails too, making the *last* thing to fail look like the
cause. A fixed-width tile with in-place softmax costs exactly the 2x the truncation saved and is
the difference between an oracle that runs and an oracle that is merely described. **A
memory-shape argument beats a FLOP-count argument when the allocator is the binding constraint**,
and a loop whose allocation size is a function of the loop variable is the signature to look for.

---

## 8. Threats to validity

A reviewer who finds these themselves trusts the rest less, so they are here.

**1. The clocks are not lockable.** `nvidia-smi -lgc` fails with "Unknown Error" under WSL2
(persistence mode works). Every timing is minimum-of-medians with the fact recorded per row as
`clocks_locked: false`. The measured noise floor is **±7%**, estimated from accidental replicates
rather than assumed, and it is **per-config, not global**: rows above a millisecond reproduce
within 0.6% while sub-millisecond rows do not. It is arguably not even per-config — finding 39
documents the same configuration being stable for one candidate and unstable for another, because
the candidate had moved config 2 across from GPU-bound to CPU-bound and the median started
sampling the host's jitter. On min-of-N the two distributions did not overlap in 10 runs out of
10; on the median they did, and `run_matrix` scores on a median.

**2. `ncu` is unavailable.** `ERR_NVGPUCTRPERM`: WSL2 denies GPU performance counters and the fix
is a modprobe option on the *host* driver, which we do not control. So no `dram__bytes_read.sum`,
no occupancy counters, no memory-workload analysis. Every memory-traffic claim in this report is a
**contrast between arms** — two kernels differing in exactly one property — or a roofline
derivation, never a counter reading. Section 6.2 is the worked example of doing this responsibly,
including the positive controls that show the instrument can see the thing whose absence is being
reported. Register counts and spill counts come from Triton's `CompiledKernel` rather than from a
profiler.

**3. Our harness measured a different quantity than the grader, and inverted two signs**
(finding 42). Three protocols were in play and all three differ:

```
GRADED       interleaved ABBA/BAAB rounds, samples pooled, baseline.median / optimized.median
PRESCRIBED   minimum-of-N, interleaved (what CLAUDE.md asks for)
ACTUAL       min(median, median) per arm, NOT interleaved, candidate compiled and autotuned
             BETWEEN the two arms — which produced all ~600 pre-finding-42 ledger rows
```

The ordering is the cause: the harness times the baseline, then *builds* the candidate — compile,
autotune, JIT, all of which heat the GPU — then times the candidate with no clock lock. On `v34`
against `v26`: configs 1 and 9 read **+6.1% and +5.6% WORSE** in our ledger and **−0.8% and −6.9%
BETTER** under the graded protocol. The research agent predicted the inversion in advance from a
kernel census, because v34 launches strictly fewer kernels there and a regression was mechanically
implausible. `bench/end_to_end.py` was written days earlier for exactly this check, with a
docstring saying nobody had ever run it. Nobody ran it, and a session was spent ranking candidates
on a quantity that is not the score. Correctness rows are unaffected — accuracy is checked per
trial against a fresh reference, not by comparing runs.

**4. The isolated arm misreports runtime planners by 2–4x** (finding 45, §4.3). This is worse than
drift: it is a systematic error correlated with the very design pattern the project's own contract
*mandates*, since predicates must be derived from measured device properties and therefore run
measurements at construction. Only two candidates carry a corrected arm.

**5. The graded harness cannot rank two of our candidates either.** Running its own `main()`
unmodified, three times, the **baseline** arm — byte-identical unmodified reference code —
spreads:

```
cfg 8   16.2580  16.2652  16.2662     0.1%
cfg 1    1.8848   1.7958   1.8396     5.0%
cfg 9    1.7316   1.6794   1.5928     8.7%
cfg 4    1.8381   2.0054   2.2486    22.3%
cfg 2    2.1646   2.4544   1.8427    33.2%
cfg 3    1.8024   2.5104   2.0343    39.3%
```

The noise scales inversely with config size, and the worst rows are exactly the ones carrying all
the remaining score, while the *optimized* arm is stable to the last digit. The mechanism is
settling time: round 1 of 100 timed calls reads 932.9 µs on config 1 where rounds 2–3 read 250.9 µs
stable to 0.1 µs — about 130 calls of settling after CUDA-graph capture, against 20 warmup
iterations. **Both arms pay it, so a single submission's score is fair; a five-sample median of it
cannot separate two candidates 5% apart on a 0.25 ms row.** The consequent rule is to rank by a
candidate's own time against a fixed reference, never by a per-run ratio.

Finding 44 pushed this one row further down the size axis: on config 3 (512 tokens), per-forward
device time reproduced to **0.3% across four runs** while the wall-clock median for the same arms
read 53.3, 55.9, 56.6, 65.6, 80.1, 85.1, 92.2, 105.5 and 162.5 µs. The device does 43–48 µs of work
and the Python side needs 25–50 µs to submit it, so the wall measures whichever of the two lost the
race. **Where host submit time is within 2x of device time, the wall is still what the grader
scores, but it is not what tells you whether the kernel got faster.**

**6. One device, one operator.** Every number is one RTX 4070 Ti SUPER under WSL2. The dispatch
predicates are written against measured device properties precisely so the result transfers, and
the tests flip them by halving the property — but **no second device has verified that claim.** The
most concrete transfer risk is `MIN_RESIDENT_BLOCKS = 4` (§3.2), a crossover measured on this
card's register file, and `AMORTIZE_FRACTION = 0.002` (§3.1), a sign flip located between two
measured points.

**7. Dynamo's recompile limit is a deployment hazard we avoid by accident.** `cache_size_limit` is
8 and shared per process; once exhausted `torch.compile` falls back to eager **silently**, and the
profile shows pure ATen kernels. A graded run compiling 13 shapes in one process could hit it. Our
harness forks per configuration, so we are safe by accident rather than by design.

**8. Two structural claims are still unenforced.** `tests/bench/test_lineage_topology.py` currently
fails for v23 and v26 (pre-existing, verified identical with the newest generation stashed), and
finding 41 flagged that the topology invariant **cannot express a two-parent recombination at all**
— which has now happened twice. Two recombinations recorded as one-parent lineage plus a summary
string, and the declared tree is no longer the tree.

**9. The declared frontier and the measured frontier disagree on config 6.** §4.5. This is the most
serious open item and it is unresolved at the time of writing.

**10. The strict correctness gate is nearly saturated by bf16 itself.**
`tests/golden/test_reference_floor.py` records that the both-bounds floor sits at max_abs 1.95e-03
against the locked 2e-03, and that near-zero output elements make the relative bound a seed lottery
on ≥256k-element shapes. Expect honest kernels to trip it. **The tolerances are locked and have
never been widened**, and any reinterpretation requires explicit human sign-off — but a reviewer
should know the margin is thin on that surface specifically.

**11. Two environment gaps unrelated to the kernels** (finding 07). A `tests` package in system
`dist-packages` shadows this repo's `tests/` under pytest's default import mode (fixed in
`pyproject.toml` with `--import-mode=importlib`), and `git merge-tree --write-tree` needs git ≥ 2.38
where this machine has 2.34.1, so the experiment-worktree consolidation mechanism cannot run here.
The `bench/` lane does not depend on it.

---

## 9. How to reproduce

None of the derivations in sections 1, 4 and 5 need a GPU; they read the ledger.

```bash
# integrity of the immutable oracle zone — run before and after any session
./scripts/check-oracle.sh

# the announced matrix as executable data, with derived feasibility per row
python3 bench/matrix.py

# the live scoreboard: score -> commit, clade metaproductivity, Thompson draw
python3 bench/ledger.py

# the loop's own invariants (lineage topology, registry integrity, predicate structure)
python3 -m pytest tests/bench/ -q

# the manifest-protected correctness surface
python3 -m pytest tests/golden/ -q
```

To re-derive §4.1 and §4.2 exactly: read `bench/results.jsonl`; take the compiled baseline from
rows with `candidate == "baseline_compiled"`, `status == "ok"` and `dirty == false`; take each
candidate's per-config `timing.candidate_ms`, preferring `timing_interleaved.candidate_ms` where
present (per §4.3); and divide. `bench/ledger.py::scoreboard()` does this for the isolated arm, and
`compiled_baseline_ms()` is the function that reads the denominator as *evidence* rather than as a
constant someone typed. `matrix.weighted_score()` applies the 3.0 cap and scores an unmeasured
configuration as 1.0 rather than skipping it, because skipping would reward not measuring.

With a GPU, and **only one process at a time** — two processes on one GPU produce two wrong
numbers, not two measurements:

```bash
python3 -m ratchet.oracle.device                              # recalibrate the device record
python3 bench/screen.py     --candidate <name>                # 30 s, 4 configs, advisory only
python3 bench/run_matrix.py --candidate <name>                # 112 s, all configs, recorded
python3 bench/run_matrix.py --candidate <name> --ids 6 13     # a subset
python3 bench/run_matrix.py --candidate v33_streamed_long --ids 14 --oracle-sequences 1
python3 bench/abba.py       --candidates v36_gemm_gelu v37_recombined2   # interleaved ranking
python3 bench/end_to_end.py --candidate <name>                # the graded harness, unmodified
```

`bench/run_matrix.py` refuses a dirty tree and a contended GPU, runs one configuration per
subprocess, checks correctness before timing, and records failures as rows rather than skipping
them. `bench/abba.py` ranks and **does not** write to the ledger. The GPU lock is
`bench/gpu_lock.py`, and every tool that measures takes it.

**Warning about the ad-hoc route.** `run_matrix.py` embodies six rules at once — arms isolated,
one config per subprocess, correctness before timing, min-of-N under unlockable clocks, refuse a
dirty tree, refuse a contended GPU — and **every ad-hoc probe opts out of all six
simultaneously.** This project has produced at least four wrong numbers that way. A probe may
propose; it may never conclude.

---

## 10. Where things live

| | |
|---|---|
| the 14 configs, as executable data | `bench/matrix.py` |
| every measurement ever taken, append-only | `bench/results.jsonl` |
| ledger API, CMP, Thompson sampling, scoreboard | `bench/ledger.py` |
| the measurement harness | `bench/run_matrix.py`, `bench/screen.py`, `bench/abba.py`, `bench/end_to_end.py` |
| hand-written Triton kernels | `bench/kernels/` |
| candidate registry and declared lineage | `bench/candidates/__init__.py` |
| config 14's predicates and both oracles | `bench/feasibility.py` |
| the immutable oracle zone (SHA-256 manifested) | `ratchet/ratchet/oracle/` |
| 45 numbered findings, each with its method | `docs/findings/` |
| 48 running learnings, the loop's long-term memory | `docs/findings/00-learnings.md` |
| the 24-rule distillation — read this first | `docs/loop/method.md` |
| the proposal rubric and its backtest | `specs/07-proposal-rubric.md`, `bench/proposals/backtest.py` |
| the device calibration record | `ledger/device.json`, `docs/00-mission.md` |

The governing idea, offered as the project's thesis: **in agentic optimization the scarce resource
is not ideas but trustworthy measurements.** Roughly half of this project's effort went into the
measurement apparatus, and the findings catalogue records that apparatus catching our own numbers
being wrong — a 4.1x inflated baseline, a 7.2x headline that was really 1.69x, a 2.7%
"improvement" that was a re-measurement of the starting point, a 1.58x isolated win worth nothing
in the model, and two inverted signs — on every occasion before the number escaped. The kernels
are worth ~3.1x on one card; the apparatus is why the 3.1x is believable, and it is the half that
transfers.
