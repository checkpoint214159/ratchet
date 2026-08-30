# 46 — Config 6's 1.6x regression: the decision was taken at the harness's worst moment

**Date:** 2026-08-30. **Branch:** `cand/g38/stream-fallback` (from `cand/g37/recombined2`
at `42bacd7`). **Candidate:** `v38_stream_fallback`, parent `v37_recombined2`.

Section 4.5 of the technical report recorded config 6 as an **open defect** and the single
largest unresolved risk in the submission: the whole streaming lineage ran 1.6x slow on the
row that is 83% of the matrix's wall time, across two sweeps, two commits and two parents.

    v26_causal_correct    57.437 ms     baseline 441.251
    v33_streamed_long     92.178 ms     baseline 447.399     <- streaming
    v34_launch_bound      62.085 ms     baseline 476.630
    v36_gemm_gelu         59.076 ms     baseline 463.853
    v37_recombined2       90.974 ms     baseline 443.889     <- streaming

It is diagnosed, fixed and verified. `v37_recombined2` was the capability superset we
would otherwise submit and was not defensible on speed; `v38_stream_fallback` is.

---

## 1. The diagnosis, measured rather than argued

`bench/probes/g38_stream_fallback/probe_decision_moment.py` replays
`bench/run_matrix.py`'s `measure_one` ordering exactly and prints free memory and the
latched decision at each stage. On config 6 with `v37_recombined2`:

    [after baseline arm timed            ] free  3.69 GiB  reserved 11.00  allocated 0.62
    [both models resident, pre-forward   ] free  3.69 GiB  reserved 11.00  allocated 0.62
    [+ correctness input                 ] free  3.69 GiB  reserved 11.00  allocated 1.24
    [+ reference output (the real moment)] free  2.75 GiB  reserved 11.61  allocated 1.85
    DECISION AT FIRST FORWARD: streamed
      streamed: working set 3.66 GiB vs 0.35 x 2.75 GiB free; slice=2627
    [baselines freed, timing phase       ] free 13.27 GiB  reserved  1.40  allocated 0.69
    DECISION STILL LATCHED AS: streamed
    candidate 90.956 ms

which reproduces the ledger's 90.974 ms to 0.02%. `choose()` evaluated on the same shape
against the memory free during the **timing** phase returns `("resident", True)`.

**The predicate is not wrong about memory. It is asked once, at the worst moment there is,
and never asked again.** `_decide_stream` latches on the first forward; under `run_matrix`
the first forward is the correctness check, which by design runs with both models resident
and the reference output live, immediately after the baseline arm has been timed. The
graded harness keeps both models resident too, so "decide later" does not fix it by itself.

## 2. Two things were wrong with the predicate, and only one of them is the timing

**(a) The estimate is ~1.5x pessimistic and the budget multiplies it.**
`estimate_working_set_bytes` returns `activation * 6` = 3.66 GiB for config 6. The
lineage's **measured** peak on config 6 is 2656 MB = 2.47 GiB (ledger, `memory.peak_MB`,
v23 through v26), and that figure already includes the input, the output and the CUDA
graph's static buffers. `RESIDENT_BUDGET = 0.35` then demands 10.46 GiB free to use 2.47.

**(b) `mem_get_info` is the wrong denominator entirely, and no budget fixes that.**
It reports memory free **on the device**. At the moment of the decision this process's own
caching allocator had *reserved* 11.61 GiB and *allocated* 1.85 of it — nearly 10 GiB of
blocks the resident forward can have without a single `cudaMalloc`, which `mem_get_info`
counts as unavailable. The predicate understates what a resident forward may use by
roughly the amount the process has already cached, which under this harness is most of
the card.

Tuning `RESIDENT_BUDGET` until config 6 passes would fit one constant to one row of a
fourteen-row matrix and would still be asking a question whose denominator does not mean
what the caller thinks it means.

## 3. The fix

**Try resident. Stream only after an actual `torch.cuda.OutOfMemoryError`.**

Exact, and nothing to calibrate: a shape that fits runs resident because it fit; a shape
that does not fit streams because it did not. The catch is narrow on purpose —
`OutOfMemoryError` is the caching allocator's own signal, raised after it has already
flushed and retried, with the CUDA context intact. A bare `except Exception` would convert
every bug below into a silent switch to a path that still returns an answer ([L23], [L25]),
and a raw driver `CUDA error: out of memory` (`AcceleratorError`) poisons the context, so
"recovering" from it would produce numbers from a broken device.

One pre-check survives, and it is not an estimate: the **signature floor**,

    signature_floor_bytes(B, S, d, elem) = 2 * B * S * d * elem   >   total_memory

`forward(x) -> y` holds both tensors at once and no implementation removes either (a
mutated view of the input is the [L25] defect). Config 14's full batch is 24.41 GiB against
15.99 GiB — 1.53x over, on two tensors, with no coefficient. `total_memory` is a measured
device property, so the same predicate lets an 80 GiB card attempt what this one refuses;
no config id appears (CLAUDE.md rule 2). It is the same floor `bench/feasibility.py` states
as impossibility 2, and a test pins the two equal so they cannot drift ([L14]).

## 4. The fix's own defect, caught by a probe that was built to watch it fail

The first draft was **inert in exactly the case it exists for**.
`bench/probes/g38_stream_fallback/probe_real_oom.py` caps the caching allocator with
`set_per_process_memory_fraction`, so it refuses for real through its own code path, at a
budget between a resident forward's measured peak (841.9 MB) and what it holds at rest
(372.1 MB). The fallback fired — `stream_fallbacks: 1`, `stream_basis: oom_fallback`,
every kernel decision re-settled against the slice — and then **OOMed again**:

    cap 560.0 MB;  slice = 256  for a batch of 256

because the slice was still sized from `mem_get_info`, which under the cap cheerfully
reported 13.94 GiB free against a 534 MiB budget. **A slice equal to the batch is not a
smaller computation; it is the identical one.** So the argument that replaced the path
predicate now also applies to the slice:

* whenever the candidate commits to streaming, the slice is bounded strictly below the
  batch — the whole batch is *known* not to fit, that is why we are here;
* `_streamed_forward` **halves and retries** on an allocator refusal, terminating at one
  row, where the OOM is re-raised with its own traceback because the shape genuinely does
  not fit this device.

[L36] is the lesson and it landed on this author: the fallback passed an accuracy test, set
every observable flag correctly, and did nothing.

### 4a. And the release was not releasing — `gc.collect()` is load-bearing

The narrowing loop then exposed a second thing. A `CUDAGraph` hands its private memory pool
back when the Python object is **destroyed**, and `_graph = None` only drops one reference:
a capture that OOMed part-way leaves the object in a reference cycle through its own
traceback, so refcounting alone does not collect it and `empty_cache()` has nothing it is
permitted to release. Measured, on the same capped allocator, at the moment the fallback
tried to allocate its output tensor:

    without gc.collect()   257.88 MiB still held in private pools  ->  8 MiB of a
                                                                      188 MiB budget left
    with    gc.collect()    11.88 MiB                              ->  246 MiB recovered

`_release_shape_state` is therefore `_invalidate_shape_state` + `gc.collect()` +
`empty_cache()`, in that order.

### 4b. The floor the narrowing cannot cross, stated rather than papered over

At a 0.40-of-headroom cap the probe narrows 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1 and then
**re-raises**, every time on the same allocation: `out = torch.empty_like(x)`, 64 MiB.
That is correct. The output tensor is the full batch's, no slice shrinks it, and a budget
that cannot hold input + output cannot run the shape by any means — which is the signature
floor again, arriving from the other side. The candidate fails loudly with the allocator's
own traceback instead of looping or lying. At 0.65 of headroom the window opens and the
same code narrows once and answers inside the locked tolerance.

## 5. The four verifications

### (1) Config 6 runs resident and is back in family

ABBA-interleaved, all three arms resident in one process, cold round discarded, min of
four kept rounds (`bench/abba.py`, the protocol of finding 44). **Two independent runs**,
the second after the `gc.collect()` and slice-narrowing changes of section 4:

    config 6              run A median / min      run B median / min    path     max_abs
    v26_causal_correct     58067.5 / 57606.1      59110.4 / 57672.7    (none)   0.001447
    v37_recombined2        91430.9 / 90870.8      91991.0 / 91502.6   streamed  0.001499
    v38_stream_fallback    60034.1 / 57609.2      57938.9 / 57628.7   resident  0.001447

    config 8 -- in-run control, byte-identical code on v37 and v38
    v26_causal_correct      6601.7 /  6589.4       6624.3 /  6599.7
    v37_recombined2         6602.8 /  6588.4       6602.8 /  6589.4
    v38_stream_fallback     6607.9 /  6577.2       6603.8 /  6592.5   <- all within 0.4%

v38's minimum matches v26's to **3 and 44 microseconds on a 57.6 ms row** (0.005% and
0.08%) across the two runs, and its accuracy against the reference is v26's to six digits
— the two are executing the same path. v37 streams *in both runs*, so the defect is
reproduced in-process by an interleaved protocol, not only inferred from two old sweeps.
Config 8 puts the protocol floor at ±0.4%; v38's own two config-6 minima differ by 0.03%.

Ranking is by the candidate's own time against a fixed reference, never by a per-run
speedup ratio (finding 42's addendum).

### (2) Config 14 still streams, and its capability result is bitwise unchanged

`bench/probes/g38_stream_fallback/probe_config14_paths.py`, at config 14's true shapes:

| | v37_recombined2 | v38_stream_fallback |
|---|---|---|
| per-sequence (B=1, S=100000, d=1024) | `resident` | `resident`, basis `attempt` |
| max abs difference of the two outputs | — | **0.0** (bitwise identical) |
| full batch (B=32) decision | `streamed` | `streamed`, basis `signature_floor` |
| resident attempted at the full batch | — | **no** |

The causal-prefix oracle and the blocked fp64 certificate are computed on the
**per-sequence** output — every config-14 ledger row records `stream_path: resident,
slice=1` there — and that output is bitwise identical to v37's. So both oracles return
exactly what they returned for v33 and v37, without re-running 525 s of fp64.
`bench/feasibility.py` is untouched (it does not exist on this branch point; the
cross-check test skips here and runs after the merge).

### (3) The shape-latch reset survives

`SHAPE_LATCHED` is still derived by `shape_latched_over(v36_cls, v26)`, still covers v35's
declared five, and the fallback *uses* it — `_release_shape_state` is
`_invalidate_shape_state` plus `empty_cache`, which is exactly the transition a
half-finished resident forward needs. The two-shape probe (warm at batch 8, call at batch
1) still returns `(1, 128, 128)` where v26 returns eight rows. A new assertion pins the
other direction: `stream_slice`, `stream_path` and `stream_fallbacks` must **not** enter
the derived set, because the fallback sets them and the reset runs around them.

### (4) The mechanism is asserted, from measured device properties

`tests/bench/test_v38_stream_fallback.py`, 21 tests. Against this card's reported
`total_memory`, the floor pre-check refuses **exactly** `[14]` of the fourteen announced
configs — and admits config 14's own per-sequence shape, which is where its capability
result comes from. Each of the three routes into `stream_path` is asserted through
`stream_basis`, which `stream_path` alone cannot distinguish, and each has a control:

* the parent still streams at the same reported free memory ([L40] — the proof there was
  something to fix);
* the same shape under the *real* card is attempted, so the floor test cannot pass by
  refusing everything;
* a non-OOM `RuntimeError` propagates instead of being converted into a slower path;
* the free-memory predicate is poisoned in both modules that hold a reference to it, and a
  forward still completes — so nothing v38 inherits consults it either;
* a **real** allocator refusal (`set_per_process_memory_fraction`) narrows the slice and
  answers inside the locked tolerance.

## 6. Proposed lessons

Not appended to `docs/findings/00-learnings.md` — `ben` is ahead of this branch and the
numbering would collide. Offered for the integrator to take.

### L55 — A decision latched at the first forward is latched at the harness's worst moment

Correctness runs before timing, and it runs with the baseline model resident; that is the
harness's documented design and the grader's too. So the first call a candidate ever sees
is the one call taken under maximum memory pressure, on the coldest caches, before
anything has settled. Anything decided lazily is decided *there* and then latched into a
steady state that looks nothing like it. Either decide at the moment you will run, or
decide something that does not depend on the moment.

### L56 — `mem_get_info` is not "memory you can have"

It reports what is free on the **device**. The caching allocator's reserved-but-unallocated
pool is invisible to it and is usually the largest pool available to the very forward being
judged — here 10 GiB of the 13 the decision needed. Any predicate over free device memory
systematically under-provisions the process asking it, by an amount that grows with how
much work that process has already done. When the thing you want to know is "does this
allocation fit", the exact test is to make it and catch `OutOfMemoryError`; an estimate can
only be calibrated, and a calibration is a constant fitted to the rows you happened to run.

Corollary, learned the hard way in section 4: **the recovery path needs the same
discipline as the thing it recovers from.** A fallback sized from the same wrong number is
a fallback that does nothing, and it will pass every accuracy test while doing it.

## 7. Two things the regression run turned up that are not about v38

**`tests/bench/test_v37_recombined2.py` is order-dependent.** Running seven candidate test
files in one pytest process, three of its tests fail — `test_the_kernel_reduction_
survives_the_second_merge[2]` and `[12]`, and `test_config_3_still_takes_the_path_that_
produced_v35s_win`. Run alone, **all 43 pass, with these changes applied**; the combined
run logs `torch._dynamo hit config.recompile_limit (8)`, so by the time v37's file runs
Dynamo has stopped compiling and the kernel census it asserts on is of a different program.
Verified in both directions: the same three tests also pass alone on the *unmodified* tree,
and pass alone with these changes. Not a regression, and worth a `--forked` or a
`torch._dynamo.reset()` fixture rather than living as three intermittent reds.

**`test_lineage_topology.py::...[v23_single_tile_attn]` and `[v26_causal_correct]` fail at
this branch point** and are untouched by this work — they name candidates from generations
23 and 26 and fail identically before any of these commits. `ben` carries a later version
of that file (+58 lines); this branch does not.

## 8. Disposition

`v38_stream_fallback` is `v37_recombined2` plus a dispatch decision, with no new kernel and
therefore no new speed argument ([L33]). It is a null against v37 everywhere v37 was
already resident (config 8: 0.3%; config 14 per-sequence: bitwise), and it recovers
config 6 to its parent line.

**Not yet swept.** No ledger row is written from this branch: `bench/run_matrix.py` at this
branch point predates finding 42's interleaved second opinion, and finding 45 shows the
isolated arm misreports this lineage by 2-4x. The sweep belongs to the controller, on
`ben`'s harness, after the merge. What is claimed here is configs 6, 8 and 14, by the
protocols named against each.
