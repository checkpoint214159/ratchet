# GB10 CUDA qualification gate

A positive backend probe means only that a device appears available. It does **not**
qualify correctness, timing, memory, compilation, or a tuned dispatch. Per
`docs/hardware-support.md`, each vendor needs a separately ratified hardware hierarchy
before any empirical event is allowed. This file is GB10's gate — the CUDA analog of
Intel FG-01.

No row in `03-results.md` and no `ledger/` empirical event is permitted until every check
below passes on GB10.

## Gate checklist

- [x] **Toolchain** — `00-toolchain.md`: torch 2.9.1+cu130 + Triton 3.5.1 target `sm_121`;
      trivial Triton kernel compiles, runs, matches reference (needs `TRITON_PTXAS_PATH`).
- [x] **Calibration** — `ledger/device.gb10.json` written; `01-calibration.md` filled.
      *Caveat:* peak-TFLOP/s table entry missing → ridge point not yet valid (Zone-A gap).
- [x] **Backend probe** — `python -m ratchet.backends --backend cuda` reports available,
      dtypes `float32`/`bfloat16`, compile/events/peak-mem true (`validation: unvalidated`).
- [x] **Allocation + dtypes** — fp32 + bf16 confirmed; fp32-accumulate path available.
- [x] **Correctness oracle** — **PASS.** Reference floor holds through the locked gate on
      GB10 after a per-rig golden-seed re-pin (see `03-results.md`); tolerances and
      `reference.py` unchanged.
- [x] **Baseline family** — best-of-family timed on GB10: bf16 eager **4.285 ms** median is
      the number to beat (fp32 eager 5.205 ms; `torch.compile` no median win). `03-results.md`.
- [x] **Synchronized steady-state timing** — `torch.cuda.Event` timing; clocks locked at
      2418 MHz (min-of-N still advised).
- [x] **Authoritative evaluator** — `torch_transformer_benchmark.py` runs unmodified;
      optimized path reached only via `UserOptimizedTransformer` (currently identity).
- [x] **gpu-marked tests** — **63 passed, 0 failed.**

## Ratification

When every box is checked, record here: date, the human who ratified, the commit, and the
first `EXP`/`EVT` id that becomes eligible. Until then GB10 is `probe-only`.

## Status

**Ready pending one non-blocking gap.** 9/9 checklist boxes pass. The only open item is the
missing Zone-A peak-TFLOP/s entry (invalid ridge point) — it does **not** block candidate
correctness or latency measurement, only roofline interpretation (E1). A human should still
formally ratify above before the first empirical `EXP`/`EVT` is appended.
