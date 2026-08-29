# docs/findings/

Durable notes from working sessions. One file per finding, numbered in the order they
were established.

The rule that makes these worth keeping: **a finding records what was measured and how,
not what was believed.** Where a number appears, it was produced on the machine named in
the note. Where something is inferred rather than measured, it says so.

| # | Finding | Established |
|---|---|---|
| [01](01-competition-matrix.md) | The announced shape matrix and what it implies | 2026-08-29 |
| [02](02-allowed-techniques.md) | What the rules permit, decided against measurement | 2026-08-29 |
| [03](03-baseline-measurements.md) | Baseline and candidate across the real matrix | 2026-08-29 |
| [04](04-the-flash-attention-that-never-was.md) | v1 never reached FlashAttention; the fix is worth 3.11x -> 5.58x | 2026-08-29 |
| [05](05-two-measurement-artifacts.md) | Two harness bugs that produced plausible wrong numbers | 2026-08-29 |

Related, and deliberately elsewhere:

- `bench/matrix.py` — the matrix as executable data; findings cite it, never restate it.
- `bench/results.jsonl` — the measurement rows behind any number quoted here.
- `research/` — the fail-closed evidence archive. Nothing in `findings/` is ratified
  evidence under that hierarchy; see `bench/README.md` for the boundary.
