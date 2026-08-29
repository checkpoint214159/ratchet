> **HISTORICAL — NVIDIA/Triton competition scaffold. Not current, not an instruction.**
> Ratchet is now hardware-agnostic and hardware-gated (see `docs/hardware-support.md`).
> Single-GPU details below (e.g. `sm_120`, `wgmma`, A100/H100 tile budgets) are retained
> as historical design material, never as a general finding or a fixed target. See
> `docs/handoff-readme.md`.

# Spec 02 — Ledger

**Zone C. Append-only.** Seed code in `seed/ratchet/ledger.py`.

## Principle

A measurement is a fact about hardware at a moment in time. A prediction, a ranking, or a
critic score is an opinion. The ledger stores facts. Everything else is a derived view,
rebuilt by a pure function, and may be deleted at will.

This distinction is what lets the Tier 2 critic evolve without ever invalidating a GPU
measurement — the departure from RQGM's selective erasure that makes the design affordable.

## Storage

`ledger/measurements.jsonl` — one JSON object per line, UTF-8, newline-terminated, opened
`"a"`, flushed and `os.fsync`'d after every write. Never rewritten, never sorted in place,
never pruned. If it gets large, roll to `measurements.NNN.jsonl` and keep every part.

`ledger/artifacts/<candidate_id>/` — kernel source, compiled PTX/SASS if captured, the
`ncu` report, stdout/stderr. Content-addressed by the candidate's source hash so identical
candidates are stored once.

`ledger/device.json` — the calibration cache.

`ledger/intents/` — the scout's queue (see spec 06).

## Row schema

```jsonc
{
  "schema": 1,
  "ts": "2026-08-29T14:22:01.443Z",
  "run_id": "r0007",
  "candidate_id": "c0f3a91e",          // sha256 of normalized source, first 8
  "parent_id": "c0a71b40",             // null for a seed; drives clade metaproductivity
  "origin": "proposer" | "scout" | "search" | "seed" | "baseline",
  "intent_id": "i0031",                // null unless scout-originated

  "kernel": {
    "family": "flash_v2_tiled",
    "source_path": "ledger/artifacts/c0f3a91e/kernel.py",
    "source_sha256": "…",
    "config": {"BLOCK_M":128,"BLOCK_N":64,"num_warps":4,"num_stages":3}
  },

  "shape": {"B":32,"N":512,"H":16,"D":64,"causal":false,"dtype":"bfloat16"},

  "env": {
    "device_name": "…", "cc": "sm_120", "driver": "…",
    "torch": "…", "triton": "…", "cuda": "…",
    "clocks_locked": true, "locked_sm_clock_mhz": 1200
  },

  "status": "ok" | "compile_error" | "incorrect" | "timeout" | "oom" | "crash",

  "correctness": {
    "passed": true,
    "max_abs_err": 3.1e-4, "max_rel_err": 8.2e-4,
    "per_distribution": {"standard":true,"scaled_up":true,"scaled_down":true,"negated":true},
    "nonfinite_ok": true, "deterministic": true, "generalizes": true,
    "diagnostic": null                 // first-N mismatches with indices when failed
  },

  "timing": {
    "method": "cuda_event",
    "runs": 214, "mean_ns": 41230.0, "std_ns": 380.1, "sem_ns": 26.0,
    "min_ns": 40610.0, "p50_ns": 41190.0,
    "l2_flushed": true, "flush_bytes": 268435456,
    "warmup_ms": 25, "budget_ms": 100,
    "cross_check": {"method":"do_bench","mean_ns":41880.0,"agree_pct":1.6},
    "warning": null                    // e.g. "launch_dominated"
  },

  "memory": {"peak_alloc_bytes": 41943040, "peak_reserved_bytes": 67108864},

  "baseline": {
    "best_name": "sdpa_cudnn",
    "best_mean_ns": 58900.0,
    "all": {"eager_tf32": 121000.0, "compile_max_autotune": 63100.0,
            "sdpa_flash": 59900.0, "sdpa_cudnn": 58900.0,
            "sdpa_mem_efficient": 71200.0, "sdpa_math": 210000.0}
  },

  "derived": {                          // convenience only; recomputable from the above
    "speedup_vs_best_baseline": 1.428,
    "achieved_tflops": 53.3,
    "arithmetic_intensity": 256.0,
    "roofline_fraction": 0.61
  },

  "profile": {                          // optional
    "sm_throughput_pct": 71.2, "dram_throughput_pct": 22.4,
    "achieved_occupancy_pct": 18.5, "tensor_pipe_pct": 64.0,
    "top_stall": "long_scoreboard",
    "interpretation": "compute bound, tensor pipe is the busy one, no action on memory"
  },

  "critic": {                           // Tier 2; an OPINION, freely recomputable
    "epoch": 3, "p_compiles": 0.91, "p_correct": 0.74, "pred_speedup_band": [1.1, 1.6],
    "gated": false
  }
}
```

Rules:
- Unknown fields are `null`, never omitted, so consumers do not branch on presence.
- `derived` and `critic` are the only recomputable blocks. Everything else is a fact.
- `schema` bumps on any breaking change; readers handle every version they have seen.

## Derived views (pure functions of the ledger)

- `best_known(shape, device) -> Measurement | None` — highest speedup among
  `status == "ok"`, with a confidence interval.
- `promotion_candidates()` — pairs where a challenger's CI does **not** overlap the
  incumbent's on the same shape, device and toolchain. Only these may update the dispatch
  table.
- `clade_stats(candidate_id)` — pooled success/failure over the entire descendant subtree,
  for Thompson-sampled parent selection. A node's own score is a biased estimator of its
  value as an ancestor; this is the correction.
- `failure_corpus()` — every `compile_error` and `incorrect` row. The critic's most
  valuable training signal, and the reason failures are recorded rather than skipped.
- `critic_training_split(epoch)` — held out **by `candidate_id`**, never by row, so a
  candidate's own measurements cannot appear on both sides.

## Acceptance

1. Kill the writer mid-append; the ledger still parses (last partial line discarded and
   counted).
2. Every derived view rebuilds from scratch to byte-identical output.
3. Two candidates with identical normalized source share one artifact directory.
4. A grep for any code path that opens `measurements.jsonl` in mode `"w"` or `"r+"`
   returns nothing. Add this grep to `scripts/check-oracle.sh`.
