# Finding 05 — Two measurement artifacts we produced, and caught

Recorded 2026-08-29. Both were bugs in *our own harness*, not in the candidates. Both
were caught by instrumentation this project already had, which is the argument for
building the instrumentation first.

## 1. Co-resident models inflated the baseline by 4.1x

The first clean-looking v2 run reported **12.15x on config 6**. The baseline underneath
it read 2037 ms, where an earlier run of the same config had read 459 ms. A 4.4x
disagreement between two runs of unchanged reference code is not a result, it is a
symptom.

Probed directly, holding everything else fixed:

| what is resident | baseline median | peak alloc | reserved |
|---|---|---|---|
| baseline alone | **446.1 ms** | 11.16 GB | 11.81 GB |
| baseline + candidate | **1851.2 ms** | 12.50 GB | 18.42 GB |

18.42 GB reserved on a card with 15.99 GB of VRAM. Under WSL the Windows driver does not
fail that allocation, it **spills to host memory over PCIe** — so the baseline arm was
quietly streaming its working set across the bus while the candidate arm was not. The
speedup was measuring our own memory pressure.

**Fix:** each arm is timed while it is the only model on the device. The baseline is
built and timed first, the candidate is built for the correctness pass, every baseline is
freed, and only then is the candidate timed.

**What that cost, stated because it is a real trade:** cross-arm interleaving, which was
the defence against thermal drift on a card whose clocks cannot be locked. Interleaving
is worth a few percent; the pressure artifact was worth 410%. Rows now carry
`interleaved: false` and `arms_isolated: true` so the trade is visible rather than
inferred.

Corrected figure for config 6: **3.21x** (v2), not 12.15x.

## 2. The ledger dirtied the tree it was recording

`bench/results.jsonl` is a tracked file. So the act of recording a measurement made the
working tree dirty, and the *next* run in the same session captured `dirty=True` — which
correctly excluded it from its own clade statistics. 13 of 14 v2 rows, then all 14 v3
rows, were disqualified by their own evidence.

The dirty rule is right and stays: a sha that does not describe the code that ran is a
false provenance claim. But **appended data is not changed source**, so `is_dirty()` now
ignores the ledger path specifically.

A related bug found alongside it: provenance was captured *per row* rather than per run,
so editing any file mid-run silently re-stamped later rows with a different sha. One
provenance is now captured at run start and stamped across the whole run.

## The general lesson

Both bugs produced *plausible* numbers. Nothing crashed, correctness passed everywhere,
and the reported speedups were in a believable range — 12.15x on the config with the most
headroom is exactly what a hopeful reader wants to see. The only thing that exposed either
was **cross-checking a number against an independent measurement of the same thing** and
refusing to accept a 4.4x disagreement.

That is the whole argument for the oracle-and-ledger discipline: not that it makes the
kernels faster, but that it makes the numbers survivable.
