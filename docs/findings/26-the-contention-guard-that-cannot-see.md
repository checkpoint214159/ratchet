# 26 — One GPU, several agents: the guard works, the detector is blind, and v15 is suspect

**Date:** 2026-08-30. **Added:** `bench/gpu_lock.py`, contention refusal in `run_matrix.py`.

## The hazard

The architecture now runs research agents, expander agents and a measuring controller at
once, all on one RTX 4070 Ti SUPER. **Two processes on one GPU do not produce two
independent measurements. They produce two wrong ones.**

Finding 05 is the precedent: a co-resident model inflated config 6's baseline 4.1x
(2037 ms against a true 446 ms) by forcing a host-memory spill. That was two models inside
ONE process, and the fix was to time the arms in isolation. Two PROCESSES is the same
failure with none of the defences — the allocator cannot see the other tenant, and the
timing loop cannot know it was descheduled.

## What was built, and what does not work

A **lock file** — cooperative, names the holding pid, reclaims a lock whose owner died.
This is the real mechanism and it is tested.

A **foreign-process check** via `nvidia-smi --query-compute-apps`, intended to catch a
subagent's throwaway probe that knows nothing about our lock. **It does not work reliably
on WSL2.** Measured directly, with a process holding a 16 MB CUDA tensor and confirmed
alive, two identical trials seven seconds apart:

    trial 1    nvidia-smi -> "893453, [N/A]"    DETECTED
    trial 2    nvidia-smi -> ""                 NOT DETECTED

Same command, same kind of holder, opposite answers. The query also always reports
`used_memory` as `[N/A]` under WSL.

**So a clean report from this check means nothing.** A positive result is still
trustworthy — if it names a process, that process is really there — but an empty one is
not evidence the GPU is free.

This is L36 arriving one level up, less than an hour after L36 was written. The lesson was
recorded as "a test can pass because its subject was never built"; here a *guard* passes
because its sensor saw nothing. Writing the lesson down did not stop me from building the
same shape of mistake into the very next thing, and only checking whether the detector
could see a process I *knew* was there caught it. **Verify that a check can fail before
trusting that it passed.**

## v15's measurement was contended, and finding 22 rests on it

Ledger timestamps are UTC; agent output mtimes are local (UTC+8):

    v15 sweep       15:57 - 16:00 UTC     <- research agent C still running, and BENCHMARKING
    agent C done    16:01 UTC
    v16 sweep       16:14 - 16:17 UTC     clean
    v17 sweep       16:22 - 16:50 UTC     clean

Agent C spent that window writing and benchmarking three Triton kernels on this GPU. The
v15 sweep overlapped it.

**Finding 22 concluded from that sweep that lifting Inductor's 68-SM veto "was real,
correctly diagnosed, and worth nothing" — and on that basis I RETRACTED a claim and closed
a direction.** Its evidence was v15 measuring 2.618x against v9b's 2.655x, and config 6
coming out +2.4% worse. Those are exactly the margins a co-resident benchmark can
manufacture.

The mechanism argument in finding 22 is unaffected: Inductor's pointwise fuser really was
already absorbing the elementwise work, and that was read from profiles, not from the
sweep. But the *quantitative* claim that the veto buys nothing is not currently supported
by a clean measurement.

**Action: re-measure v15 under the lock, on an idle GPU, before finding 22's conclusion is
used to justify anything else.** Until then it is marked provisional. The original rows are
NOT deleted — the ledger is append-only and a contended row is data about a contended run.

## L38 — Verify that a check can FAIL before trusting that it passed

A guard is only evidence if it is capable of firing. Test it against a condition you have
deliberately created, not against the quiet case you hope for. The nvidia-smi check looked
perfectly healthy in every run where nothing was wrong, which is precisely the run where a
broken detector is indistinguishable from a working one.

Corollary for this project's architecture: **every tool that measures must take the lock**,
because the only reliable exclusion here is the cooperative kind. Subagents that probe must
be told to take it too, or told not to measure at all.
