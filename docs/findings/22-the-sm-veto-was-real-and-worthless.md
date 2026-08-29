# 22 — The 68-SM veto was real, correctly diagnosed, and worth nothing

**Date:** 2026-08-30. **Candidate:** `v15_lifted_veto` (gen 15, parent v9b_reduce_overhead,
branch `cand/g15/lifted-veto`). **Node chosen by CMP + Thompson (seed 29 -> 2e855f81).**
**Ideas from two independent research agents that converged on the same mechanism.**

## The diagnosis, which was correct

`torch/_inductor/utils.py:1433`:

    min_sms = 16 if device.type == "xpu" else 68  # 3080

This card has 66 SMs. `is_big_gpu()` returns False; its single call site gates
`use_triton_template`, which gates whether Inductor may emit a Triton GEMM template; and
without a template node there is no epilogue fusion into a GEMM. All verified in our own
install, not taken on trust from the agents.

The lift works. In the real candidate at config 6's shape the profiler shows
`veto_lifted=True` and three template kernels — `triton_tem_fused_addmm_1`,
`triton_tem_fused_addmm_2`, `triton_tem_fused_addmm_gelu_4`. The GEMM epilogue really is
being fused where before it could not be.

## The isolated probe promised 1.58x

Standalone `x + gelu(addmm(b,x,w1)) @ w2`, fp16, D=128, 1.28M tokens, fresh cache:

    stock    5.576 ms   CUTLASS GEMMs + triton_poi_fused_addmm_gelu 1.006 ms
                        + memcpy128 1.061 ms
    patched  3.537 ms   triton_tem_fused_addmm_gelu x2

Both the standalone pointwise kernel and the memcpy vanished into the epilogue.

## In the real model it is worth zero, and slightly negative

    v15 vs its PARENT v9b (identical structure; the veto is the ONLY difference)
      cfg 6 (the bandwidth-bound one the win was predicted on)   +2.4%  WORSE
      geomean vs compiled          v9b 2.655   ->   v15 2.618    -1.4%

    Frontier is unchanged: v13_safe_capture 2.711x.

## Why the probe lied, which is the actual finding

The real profile at config 6's shape:

    flash_fwd_kernel                                    1.602 ms   attention
    triton_per_fused__to_copy_add_native_layer_norm_*   2.284 ms   ALREADY FUSED
    triton_tem_fused_addmm_* (the new templates)        1.865 ms

**The elementwise work the GEMM epilogue would have absorbed was already being absorbed
by Inductor's pointwise/reduction fusion, into the LayerNorm kernels.** The SM veto gates
`use_triton_template` only; `max_autotune_pointwise` and the pointwise fuser were never
vetoed at all. So lifting the veto MOVED work between kernels rather than eliminating it.

The isolated probe showed 1.58x precisely because it was isolated: a bare FFN with no
LayerNorm to fuse into leaves the epilogue nowhere else to go, so the GEMM template is the
only mechanism available and it looks decisive. Put the same GEMM back in a transformer
layer and a different fuser has already claimed the work.

## RETRACTION

I told the user this was "the likeliest single explanation for the generation 11-14
plateau." **That is false and I am withdrawing it.** The veto was real and had been
mis-read at L15, but removing it changes nothing measurable, so it cannot explain a
plateau. The plateau remains unexplained. Stating a cause before measuring the fix is the
same error as L28, where a number was committed before the measurement returned.

## L33 — A mechanism measured in isolation measures the isolation

An optimization's ceiling is set by what ELSE is competing for the same work, and a
microbenchmark deletes exactly that context. The probe was not wrong about the mechanism;
it was wrong about the mechanism's value, because it removed every alternative path the
work could take. **Before trusting a component-level win, check whether the work it saves
is already being saved by something else in the full system.**

This sharpens L32 ("measure the fix, not just the bug"): measuring the fix is not enough
if you measure it somewhere the bug is artificially dominant.

## L34 — Independent convergence is corroboration of the READING, not of the VALUE

Two research agents, given disjoint territories, independently found `min_sms = 68` and
independently proposed lifting it — one ranking it their top idea. That agreement made the
diagnosis look strong, and the diagnosis WAS strong. It said nothing about whether the fix
was worth anything, because both agents were reading the same source file and neither had
run the full model. Agreement between analysts sharing a method is not replication.

## What survives

  * The veto reading is correct and now documented; nobody needs to re-derive it.
  * `lift_sm_veto()` is device-conditioned and idempotent, and stays in the tree as a
    composable tool. It may still matter combined with something that removes the
    LayerNorm fusion competing for the same work — a hand-written fused-layer kernel, for
    instance, which is what several queued proposals actually propose.
  * v15 is kept as a measured stepping stone. It is NOT merged to `ben` and is not the
    frontier.
  * A separate hazard found while testing it: Dynamo's `cache_size_limit` is 8, shared
    per process, and once exhausted `torch.compile` falls back to eager SILENTLY — the
    profile shows pure ATen kernels. A graded run compiling 13 shapes in one process
    could hit this. Our harness forks per config, so we are safe by accident rather than
    by design. Worth stating in the tech report as a deployment caveat.
