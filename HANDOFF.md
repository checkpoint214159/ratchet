# Ratchet — handoff

You are a Claude Code instance running in WSL2 on a machine with an NVIDIA GPU. This
directory is a **specification package**, not a working repo. Your job is to turn it into
one, then run it.

A ratchet only turns one way. That is the invariant this whole system is built around:
**a measurement, once taken, is never discarded, and the best-known kernel for a shape
never regresses.** Everything else is negotiable.

---

## Read in this order

| # | File | What it gives you |
|---|---|---|
| 1 | `docs/00-mission.md` | What we are building, why, the competition constraints, and the tiering |
| 2 | `docs/04-failure-modes.md` | **Read before writing any measurement code.** Every way a kernel benchmark lies |
| 3 | `docs/01-architecture.md` | The three-zone design and the invariants that hold it together |
| 4 | `docs/02-milestones.md` | The ordered build plan with acceptance gates |
| 5 | `specs/01-measurement-core.md` … `specs/06-scout.md` | One spec per component |
| 6 | `docs/03-research-dossier.md` | Papers and reference implementations, with what to take from each |

`seed/` contains real, working code for the parts that are easiest to get subtly wrong.
Copy it in, do not rewrite it from scratch, and read `seed/ratchet/oracle/` closely —
the comments explain *why* each line is there.

---

## First actions, in order

**1. Report the hardware before anything else.** Everything in this package is
conditional on it. Run:

```bash
python -c "
import torch, triton, sys
p = torch.cuda.get_device_properties(0)
print('torch', torch.__version__, '| triton', triton.__version__, '| py', sys.version.split()[0])
print(p.name, f'sm_{p.major}{p.minor}', p.multi_processor_count, 'SMs')
print('smem/block optin', p.shared_memory_per_block_optin, '| L2', p.L2_cache_size)
"
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.memory --format=csv
```

Write the result into `docs/00-mission.md` under **Target hardware** and stop guessing
from there on. The MMA family (`mma.sync` on sm_80/86/89/120, `wgmma` on sm_90a,
`tcgen05` on sm_100a) and the shared-memory budget fork most later decisions.

**2. Bootstrap the repo.** `bootstrap.sh` has the exact command sequence, including the
Beryl install. Read its header comment first — Beryl is v0.1.0 with real caveats and
there is a documented fallback.

**3. Build Tier 0 (`docs/02-milestones.md`, M0–M3) before anything else.** Tier 0 is the
oracle, the ledger, and a single hand-written baseline kernel that passes. Do not start
the search loop until Tier 0's acceptance gate is green. A search loop on top of an
untrustworthy oracle produces confident nonsense, and you will not be able to tell.

---

## The five rules you may not break

These are restated in `CLAUDE.md`, which is the file that will actually be in your
context on every turn. They are here too because they matter more than anything else in
this package.

1. **`ratchet/oracle/` is immutable.** You may read it, call it, and report bugs in it.
   You may not edit it as part of an optimization step. If a kernel only passes after
   you change the oracle, the kernel is wrong. `scripts/check-oracle.sh` enforces this
   with a checksum manifest; if it fails, stop and ask the user.

2. **Correctness tolerances are locked constants.** `rel < 0.02`, `abs < 0.002`, defined
   once in `oracle/correctness.py`. Never widen them to make something pass. Never make
   them a search parameter.

3. **The ledger is append-only.** Measurements are facts about hardware. Model
   predictions, critic scores, and rankings are opinions and may be recomputed freely.
   Never conflate the two, never delete the former.

4. **Correctness is a gate, not a term in the objective.** A candidate that fails
   correctness scores nothing at all. It does not score "a bit less."

5. **Benchmark shapes and correctness shapes are disjoint sets.** If you tune against
   the shapes you validate on, you have measured your own tail.

---

## What "done" looks like for each tier

**Tier 0 — trustworthy measurement.** You can point at any number the system produces
and defend how it was obtained. Locked clocks, flushed L2, adaptive stopping, process
isolation, a baseline that is `torch.compile(mode="max-autotune")` with TF32 on, not
eager FP32.

**Tier 1 — a working search loop.** Kernel variants are proposed, gated, measured and
recorded automatically. The dispatch table is populated from measurement rather than
from constants. There is a report you could hand to a judge.

**Tier 2 — the co-evolving loop.** A learned critic prunes candidates before they cost
GPU time, promoted only against held-out real measurements. An adversarial input pool
that grows from near-misses. A scout that reads other implementations and proposes
architectural moves rather than parameter tweaks.

Tier 0 is a day. Tier 1 is the competition deliverable. Tier 2 is the part that is
actually novel, and it is worth nothing on top of a broken Tier 0.

---

## When you are stuck

Ask the user. Specifically ask rather than guessing when:

- The GPU is not what the plan assumed and a whole branch is now irrelevant.
- A reference implementation you need is not installable in this environment.
- You are about to widen a tolerance, weaken a test, or special-case a benchmark shape.
  These are the three moves that look like progress and are not.
- Beryl's install does something unexpected. It is v0.1.0 and the fallback is fine.
