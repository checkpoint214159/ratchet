# Spec 06 — The scout

**Zone B.** A periodic subagent that keeps the loop's sights broad, so the search does not
converge onto local refinements of whatever it started with.

## The problem it solves

An LLM proposer conditioned only on its own history explores a narrowing neighbourhood.
The large wins in this domain come from **architectural moves borrowed from someone else's
kernel** — split-K over the KV axis, materializing S in shared memory, a persistent grid,
an epilogue fusion. The scout's job is to keep injecting those.

## What it produces

**Intents, not kernels and not configs.** An intent is a hypothesis about a technique,
scoped to a regime, with a citation.

```jsonc
// ledger/intents/i0031.json
{
  "id": "i0031",
  "created": "2026-08-29T09:12:00Z",
  "technique": "split-K over the KV axis with a separate merge kernel",
  "regime": "occupancy-bound: B*H*ceil(N/BLOCK_M) < 2 * sm_count",
  "rationale": "At B=1,H_kv=2,N=512 the grid is 8 CTAs. Every SM but eight is idle. Splitting the KV axis multiplies the grid by the split factor at the cost of a cheap merge.",
  "source": {
    "repo": "meta-pytorch/mslk",
    "path": "mslk/attention/fmha/triton_splitk.py",
    "symbol": "FwOp.get_split_k",
    "note": "Analytic heuristic: choose splits so total CTAs approx a parallelism target, then halve until each split's chunk is large enough to amortize. Early exit: if Mq > 1 and B*G*H > 64, return 1."
  },
  "reduction_math": {
    "repo": "flashinfer-ai/flashinfer",
    "path": "flashinfer/triton/kernels/cascade.py",
    "note": "merge_states_kernel — do not rewrite the log-sum-exp merge, port it"
  },
  "expected_effect": "grid multiplied by split factor; wins where occupancy < ~30%",
  "predicted_regimes": ["small_batch_low_head", "decode"],
  "status": "open" | "attempted" | "won" | "lost",
  "candidates": []
}
```

The proposer draws from `intents/` where `status == "open"`, and every candidate records
its `intent_id` so the ledger can trace a win back to the implementation that inspired it.

## The reading list

`docs/03-research-dossier.md`, section C, in that order. The scout re-reads a subset each
cycle and is explicitly asked for what the current best kernel is **not** doing.

Prioritise by regime gap: if the report shows the occupancy-bound regime losing to cuDNN
by 30%, send the scout at split-K and Flash-Decoding, not at everything.

## Cadence

Run the scout when any of:
- A regime has had no promotion in K rounds (the loop is dry there).
- A new regime appears in the shape matrix.
- The report shows a regime where a vendor baseline is winning by more than 20%.
- Every N hours regardless, as a diversity floor.

## Prompt shape

The scout gets: the current dispatch table with per-regime margins, the best kernel's
source for the weakest regime, that regime's roofline position, the list of already-tried
intents with outcomes, and the reading list.

It is asked for **three intents ranked by expected value**, each with a specific file and
symbol citation. An intent without a citation is rejected — the point is to import
knowledge, not to generate plausible-sounding technique names.

Explicitly constrain: *do not propose parameter changes; those belong to the parametric
search. Propose a change to what the kernel does.*

## Guarding against plausible nonsense

The failure mode is confident, well-cited-looking intents describing techniques that do
not exist or do not apply to this hardware. Three checks before an intent goes `open`:

1. **The citation must resolve.** Fetch the file, confirm the symbol exists. An intent
   citing a path that 404s is discarded. Note that paths move — vLLM's attention kernel
   moved to `vllm/v1/attention/ops/` and most write-ups still cite the dead path.
2. **The technique must be applicable to `dev`.** `wgmma` does not exist on sm_120;
   `tcgen05` is sm_100a only; TMA needs sm_90+. An intent proposing warp specialization on
   an Ada part is discarded with a note.
3. **The regime predicate must be expressible** in terms of the calibration. If you cannot
   write the condition under which this should win as a function of device properties and
   shape, the intent is too vague to test.

## Acceptance

1. At least one scout-originated intent produces a candidate that beats the best-known on
   some regime, traceable through `intent_id` in the ledger.
2. Every `open` intent has a resolving citation and an expressible regime predicate.
3. Intents inapplicable to the target hardware are rejected with a recorded reason, not
   silently attempted.
4. The report can state, per regime, which external implementation the winning idea came
   from. That provenance is a strong Innovation and Impact claim and costs nothing to
   maintain if the ledger carries `intent_id` from the start.
