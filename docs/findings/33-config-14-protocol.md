# Finding 33 — Config 14: three different impossibilities, two real oracles, and no speedup

Recorded 2026-08-30. Branch `cand/g33/config14`, parent `v26_causal_correct` (9bb30ed).
Code: `bench/feasibility.py`, `bench/run_matrix.py` (capability path),
`bench/candidates/v33_streamed_long.py`. Tests: `tests/bench/test_feasibility.py`,
`tests/bench/test_v33_streaming.py`.

Config 14 (`B=32, S=100000, d_model=1024, heads=16, layers=2, causal`) had produced 28
ledger rows and no information: 27 of them `status="oom"` with a truncated traceback, one
capability row from finding 09. It is the only row of the matrix that has never been
measured, and it is 24.3% of the remaining score headroom.

This finding establishes what is true about it, what can be checked, and what may be
claimed. The short version: **the reference cannot run this config on any hardware that
exists; we compute all 32 sequences of it, and the answer is now checked against the
unmodified reference at the real sequence length rather than at proxy shapes; and there
is still no speedup, because a ratio needs two measured times.**

STATUS OF THE TWO ORACLES. The causal-prefix oracle (§2.1) has run at the real shape and
passed. The blocked fp64 oracle (§2.2) is built, validated against the reference at
S <= 4096, and has **not yet completed a full S=100000 run** — the GPU has been
continuously occupied by other agents and its two attempts died on driver
out-of-memory. Everything §2.2 claims about the oracle is measured; the one thing not yet
measured is the candidate against it at S=100000. Re-running it is a queued single command
(`bench/run_matrix.py --candidate v33_streamed_long --ids 14 --oracle-sequences 1`) and
this document must not be read as if it had already returned.

---

## 1. Three impossibilities, with three different scopes

The previous discussion of config 14 blurred these together. They are not the same claim
and only one of them is universal.

### 1.1 The reference's ALGORITHM — universal

`BaselineSelfAttention.forward` line 97 materialises

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale     # [B, H, S, S]
```

At config 14 that single tensor is **18.63 TiB**. It is not an estimate: one allocation,
one line, in the reference's own source. The next three lines each produce another tensor
of the same shape (`masked_fill` for the causal triangle, `masked_fill` for the key mask,
`.float()` before the softmax), so a realistic peak is 2-3x that.

Measured, by asking the driver:

| what | bytes | result |
|---|---|---|
| the whole batch, `[32,16,100000,100000]` | 19073.49 GiB | refused |
| one sequence, `[1,16,100000,100000]` | 596.05 GiB | refused |
| **one head of one sequence**, `[1,1,100000,100000]` | **37.25 GiB** | **refused** |

The last row is the one that settles it. **This is not a batch-size problem.** A single
attention head of a single sequence at S=100000 needs 2.3x this card's entire memory, and
the reference needs 512 of them. No GPU or single node that exists in 2026 has 18.63 TiB
of accessible memory; the largest single-node HBM configurations are ~30x short.

The reference's practical ceiling at config 14's width and depth, measured at `B=1`:

| seq_len | reference runs | peak |
|---|---|---|
| 1024 | yes | 0.22 GiB |
| 4096 | yes | 2.21 GiB |
| 8192 | yes | 8.40 GiB |
| 12288 | yes | 18.62 GiB (only via host spill, see §1.3) |
| 16384 | **no** | OOM at line 97, requesting 16.00 GiB |
| 24576 | no | OOM at line 97, requesting 36.00 GiB |

So the reference tops out somewhere between S=12288 and S=16384 **at batch size one**.
Config 14 asks for 8x that sequence and 32x that batch.

### 1.2 The forward signature's own floor — this card

`forward(x) -> y`, both `[B, S, d_model]` fp32, is **12.21 GiB in and 12.21 GiB out =
24.42 GiB** of tensors that no optimisation removes. Returning a mutated view of the
input would remove one of them and is not available: it corrupts the caller's tensor and
returns a wrong answer on the second call with the same buffer, which is precisely the
defect [L25] catalogued in `torch.compile(mode="reduce-overhead")`.

24.42 GiB against 15.99 GiB of VRAM is **1.53x over, before any arithmetic**. On an 80 GiB
card this clears easily — and impossibility 1.1 still stands there.

### 1.3 What actually happens on this box — one machine, one day

WSL2's WDDM driver oversubscribes into host memory. Measured directly, by allocating
2 GiB blocks until refusal: **30 GiB held on a 15.99 GiB card**, so 24.42 GiB is nominally
reachable. (This also explains the 18.62 GiB "peak" at S=12288 above, and the nonsense
`this process has 17179869184.00 GiB memory in use` in the old OOM messages.) Setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` removes the oversubscription entirely
and the input generator fails immediately with `CUDA driver error: out of memory`.

The harness's own allocation order then throws the margin away:

1. `x = torch.randn(...)` — segment A, 12.21 GiB.
2. `x = x * input_scale` — segment B, 12.21 GiB; **A is freed into the allocator's cache**.
   (Note this happens at `input_scale=1.0`. The multiply is unconditional.)
3. `valid_token_mask = torch.ones(32, 100000, dtype=bool)` — **3.05 MB**, which the large
   pool satisfies by **splitting segment A**.

A partly-used segment cannot be released, so `empty_cache()` reports `reserved=24.41 GiB,
free=0.00 GiB` and the 12.21 GiB output tensor must come from a third segment. 36.6 GiB is
past the ceiling and it is refused.

**A 3 MB mask costs 12.21 GiB.** Measured: `generate_random_case` completes in 1.9-3.2 s,
and the very next `torch.empty_like(x)` fails.

This is why every recent ledger row's traceback pointed at `F.linear` inside the *baseline*
rather than at the input generator: the generator had started succeeding once the driver
began spilling, and the failure moved one step later.

---

## 2. Correctness with no baseline output

Finding 09 recorded `correctness.passed = null` and checked **proxy shapes** — same width
and depth, at S=1024 and S=4096 where the reference fits. That was right at the time. It
is also weaker than what is available, because it never exercises a query row attending
over 100,000 keys, which is the only thing about this config that is new.

Two constructions do, and neither needs the reference to fit.

### 2.1 The causal-prefix theorem — the reference IS an oracle at the real shape

Under causal masking with an all-valid token mask, every operation in the reference is
either position-wise (LayerNorm, GELU, the projections, the residual adds, the final
zeroing) or attends strictly backwards. Therefore for any `P <= S`:

> `model(x[:, :P])` equals `model(x)[:, :P]`, exactly, in exact arithmetic.

So: run the candidate on the full 100,000-token input, run the **unmodified reference** on
the first P tokens of *that same input*, and compare. No proxy model. No proxy input. The
comparison uses `ref.compare_outputs` at the locked 2e-3/2e-2.

Measured on the reference against itself (d=1024, H=16, L=2, P=512, S=4096, TF32 on):

```
max |out(x)[:, :P] - out(x[:, :P])|  =  3.881e-04     the theorem's own slack
negative control, same P rows taken as a SUFFIX  =  1.841e+00
```

The slack is not zero because a TF32 GEMM reduces differently over a different K extent.
It is ~19% of the absolute budget, spent before the candidate is involved, and it is
reported rather than absorbed. The negative control matters as much as the result: a
suffix is not a closed computation under causality, and if it had also matched, the first
number would have been measuring nothing ([L38]).

**What it covers:** rows 0..P-1 at the real S. **What it does not:** any query attending
over more than P keys. P is derived from the same feasibility predicate against measured
free memory — a hardcoded prefix length would be the config-id branch this project forbids
in a different costume. In practice P=4096.

### 2.2 The blocked fp64 oracle — every row, including the last

`bench/feasibility.blocked_reference_forward` evaluates **the reference's own arithmetic**
in float64 with the **query axis blocked**. Blocking queries is exact: softmax reduces
along the key axis, so a block of query rows is a closed computation. It is deliberately
**not** online/streaming softmax — each query block computes its scores against every key
it may attend to, subtracts the row max, exponentiates and normalises, in the same three
steps and the same order as `BaselineSelfAttention.forward`. That matters: the candidate
uses flash attention, so an online-rescaling bug in the candidate cannot be mirrored by
an oracle that has no online rescaling.

Peak memory is O(q_block · S) rather than O(S²), so it runs at S=100000.

Validated against the reference where the reference runs:

| matmul precision | S=1024 | S=4096 |
|---|---|---|
| `high` (TF32, the mandated baseline setting) | 8.086e-04 | 8.086e-04 |
| `highest` (strict fp32) | 1.238e-06 | 1.915e-06 |

Two things fall out of that table.

**The oracle is sound.** In strict fp32 it reproduces the reference to ~2e-6, which is
0.1% of the locked 2e-3 budget. It is a legitimate stand-in.

**And, unexpectedly: the reference under TF32 is 8.086e-04 away from exact — 40% of the
absolute tolerance budget, before anything of ours runs.** The value is *identical* at
S=128, 512, 1024 and 4096, which is [L4]'s signature exactly: flat across a 32x change in
work means a representation floor, not accumulated error. TF32 carries 10 mantissa bits
(eps ~4.9e-4) and the outputs have mean magnitude 0.798. This is a fact about the
project's own mandated TF32 baseline (CLAUDE.md rule 5) and it deserves separate attention:
40% of every candidate's error budget is spent by the baseline's arithmetic.

Negative controls, all firing:

```
oracle run causal vs non-causal        1.371e+00
one bias perturbed by 0.05             2.354e-03
query blocking 64 vs 512               < 1e-12      (exactness of the blocking itself)
```

### 2.3 The certificate, and its one gap

Triangle inequality:

```
|candidate - reference|  <=  |candidate - oracle| + |reference - oracle|
```

The second term is measurable only where the reference runs. Under TF32 it is 8.086e-04
and flat in S across the range we could test. So a **sufficient** condition for passing at
S=100000 is

```
max |candidate - oracle|  <=  2.0e-3 - 8.09e-4  =  1.19e-3
```

**The gap, stated plainly:** `|reference - oracle|` at S=100000 is not measurable, by
construction. It is flat in S over the 32x range we could measure, which is good evidence
that 8.09e-04 is a representation floor rather than a growing error, but it is an
extrapolation and this document does not pretend otherwise. That residual belongs to the
reference's own fp32 softmax over 100,000 keys, not to us.

---

## 3. Batch slicing is exact, and the reference agrees

The whole capability protocol runs config 14 one sequence at a time. That is exact in
exact arithmetic — attention is within-sequence, the token mask is per-sequence, and
everything else is position-wise, so batch elements never interact.

In floating point it is not bitwise, and the reason is not ours. The batch axis is a
GEMM's M dimension, so cuBLAS picks a different tiling and reduces in a different order:

```
REFERENCE   whole batch vs sliced        3.461e-04     <- the harness's own arithmetic
candidate   whole batch vs sliced        6.632e-04
candidate whole  vs reference whole      8.567e-04
candidate sliced vs reference whole      8.252e-04     <- slicing did not hurt
```

So the assertion worth making is not a threshold on the slicing gap. It is that **slicing
does not move the candidate further from the reference**, which is the only distance a
grader measures. `tests/bench/test_v33_streaming.py` asserts exactly that, with a negative
control (comparing against a rolled batch) so the comparison is capable of failing.

---

## 3.5 What was measured, on the row

`python3 bench/run_matrix.py --candidate v33_streamed_long --ids 14`, commit 3db6faf,
clean tree, ledger row `status="reference_infeasible"`:

| | |
|---|---|
| sequences completed | **32 / 32** |
| tokens computed | **3,200,000**, at the announced S=100000 |
| peak device memory | **3.54 GiB** |
| per sequence | 0.518 s min, 0.634 s mean (`host_wallclock`, **not** a timing measurement) |
| causal-prefix check | **passed**, P=4096, max_abs **8.658e-04**, 0 failed elements |
| dispatch | `resident`, slice=1 — the capability path feeds it one sequence at a time |
| attention | SDPA; the single-tile kernel declines at S=100000, as designed |
| speedup | **null** |

This supersedes finding 09's B=32 figure, which was 32x a measured single-sequence
forward. **All 32 sequences are now actually run**, with the harness's own
`generate_random_case` producing each one, and correctness is checked against the real
reference at the real sequence length instead of at proxy shapes.

Two things it does not show. The **full-batch call** — the single `forward` a grading
harness makes — fails, as §1.2 says it must; the row records the stage and the driver's
words. And the **fp64 oracle** did not complete on the first attempt: it died with a
*driver* out-of-memory while another agent's process held ~15.8 GB, which the nvidia-smi
contention check did not see (finding 26 / [L38] again, and the row was written with
`gpu_exclusive=True` because our lock *was* held). The capability path now releases the
allocator cache before the oracle and retries it at a quarter and a sixteenth of the
query block.

---

## 4. What was built

**`bench/feasibility.py`** — the derived requirement, the two predicates
(`reference_feasible`, `signature_floor_bytes`), the causal-prefix availability check, and
the blocked fp64 oracle. Predicates take shapes and one measured device property. A test
asserts structurally — via `co_consts` and the signature, not via prose — that no config
id or `100000` literal appears in any of them.

**`bench/run_matrix.py`** — a capability path, taken when the predicate says the
*reference's* algorithm does not fit the device. It records:

- `baseline.outcome = "cannot_run"` with the derived requirement **and an empirical
  confirmation** — the score tensor is actually requested at three sizes and the driver's
  refusals go in the row;
- `signature_floor` — bytes, whether they fit, and the fact that this bound is
  device-specific unlike the one above;
- `capability` — sequences completed of sequences required, tokens computed, peak memory,
  which dispatch path ran, and a **full-batch attempt** run *last* (it poisons the
  allocator) that reports what happens when a grader simply calls `forward` once;
- `correctness.checks` — both oracles, each carrying its own `covers` / `does_not_cover`;
- `timing.speedup = None`, `status = "reference_infeasible"`.

`BenchLedger.record` gained an `extra=` argument that cannot overwrite an existing field.
Nothing about append-only, dirty-tree refusal, contention refusal or arm isolation changed.

**`bench/candidates/v33_streamed_long.py`** — parent `v26_causal_correct`. Restores the
batch-streaming dispatch that **fell out of the frontier's lineage at generation 17**: the
line is v26 <- v23 <- v18 <- v17 <- v13, and v17 branched from v13 rather than from v14,
which is where `choose()` lived. Nothing noticed for sixteen generations because on the
thirteen runnable configs the predicate always answers "resident". v33 imports v14's
predicate rather than restating it, and a test asserts the objects are identical so the two
cannot drift.

This is a **portability fix**, not a rescue. Without it the frontier plans a ~73 GiB
resident working set at config 14 and then v13's capture clones the 12.21 GiB input for a
static buffer — so it would fail this shape on an 80 GiB accelerator too, on arithmetic
that fits in under 4 GiB. It does not make config 14 runnable here, because §1.2's floor
is not a working-set problem.

---

## 5. What we may and may not claim in the tech report

**May claim, and should:**

- The reference implementation cannot execute config 14 on any hardware that exists. One
  head of one sequence needs 37.25 GiB; the full config needs 18.63 TiB. Derived from the
  reference's source and confirmed against the driver.
- We compute it, at the announced shape, in a few GiB.
- The answer is verified at the real sequence length **for query rows 0..4095** against
  the unmodified reference, using the causal-prefix theorem — real model, real 100,000-token
  input, the harness's own `compare_outputs` at the locked tolerance. Not proxy shapes.
  (Once the fp64 oracle completes, and only then, "every row including the last" becomes
  claimable. Until it does, say rows 0..4095.)
- The frontier's batch-streaming gap was found and closed.

**May not claim:**

- **A speedup. There is none, and there cannot be one.** A speedup is a ratio of two
  measured times; the denominator does not exist. Timing our own slower reimplementation
  of the baseline and dividing by it would be a number about us, not about the reference.
  `timing.speedup` is `None` and stays `None`.
- A pass/fail against the reference for query rows beyond P=4096. The causal-prefix
  theorem says nothing there, and the fp64 oracle — which would — has not completed at
  S=100000. Even when it does, what it gives is a pass against the oracle plus a measured
  and then extrapolated bound on the reference's own distance from exact (§2.3), never a
  direct `|candidate - reference|`.
- That config 14 is runnable end to end on this card by anything. It is not: §1.2.

**How it scores.** `weighted_score` gives an unmeasured config 1.0, and a
`reference_infeasible` row is not `status="ok"`, so config 14 contributes 1.0 — exactly as
if we had never looked. That understatement is deliberate and should be stated in the
report rather than quietly corrected in `matrix.py`. A grader who credits "the baseline
produces no output and this produces a verified one" will score it higher; that is the
grader's judgement to make, and putting a manufactured 3.0 into our own headline would be
the precise error [L8] and [L22] were written about. **Report the conservative number as
ours, and the feasibility result as the finding.**
