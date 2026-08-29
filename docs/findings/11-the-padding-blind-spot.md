# Finding 11 — Every number in the ledger was measured on a path the grader may not take

Recorded 2026-08-29. Candidate: `bench/candidates/v8_padfast.py`, branch
`cand/g8/right-pad-redundant-mask`.

## The blind spot

Every measurement from v1 through v7 used `padding_ratio=0.0`. That is the **only** value
at which those candidates take their fast path: they elide the all-True mask so
FlashAttention qualifies. With any padding they fall into fp32 SDPA with an explicit
`attn_mask` — which is exactly the defect finding 04 diagnosed in v1, still alive in a
branch nobody had ever run.

Measured cost on v6 at `padding_ratio=0.5`:

| config | padding 0.0 | padding 0.5 | retained |
|---|---|---|---|
| 1 | 3.68x | 1.88x | 51% |
| 5 | 3.69x | 1.87x | 51% |
| 13 | 24.06x | 6.62x | **28%** |

The benchmark exposes `--padding-ratio` and the problem statement says test cases will
cover varied shapes. If the graders use anything above zero, the headline geomean is off
the path that actually executes.

## The fix rests on a proof, not a hunch

The reference applies three things: a causal mask, a key mask on invalid positions, and a
final zeroing of invalid output rows. `generate_random_case` builds the mask as
`positions < lengths[:, None]` — a contiguous **valid prefix**, padding only on the right.

Under those two facts the key mask is redundant:

- A **valid** query at position `i` (`i < length`) is already restricted by causality to
  keys `j <= i`. Since `i < length`, every such `j` is also `< length`, hence valid. The
  key mask removes nothing.
- An **invalid** query (`i >= length`) has its output row zeroed afterward regardless. It
  cannot NaN either: causality admits keys `j <= i`, and keys `0..length-1` are valid with
  `length >= 1`, so at least one key survives the softmax.

So dropping the key mask changes no surviving output element — and dropping it is exactly
what lets q/k/v stay fp16 with no `attn_mask`, the conditions FlashAttention requires.

## Guarded, because the proof has a precondition

The argument holds only for a contiguous right-padded mask. A mask with holes, or
left-padding, breaks it: a valid query could then look back at an invalid key.
`prefix_padded()` verifies the mask really is `arange < lengths` at prime time and falls
back to the slow path otherwise. Twelve tests pin the guard — including a holed mask, a
left-padded mask, and a batch where only one row is malformed — and check the numbers
against the reference at padding 0.0, 0.3, 0.5 and 0.9.

## Result

| | padding 0.0 | padding 0.5 |
|---|---|---|
| v6 | 6.712x | ~2.86x (3-config probe) |
| **v8** | **6.730x** | **5.853x** |

Strictly better: unchanged on the fast path (within the 3% noise floor), and it recovers
most of the padded path — config 13 goes 6.62x -> 21.36x, config 5 1.87x -> 3.22x.

## The transferable lesson

This was not found by optimizing. It was found by asking which assumption had never been
tested, and the answer was a default flag value that every single run had inherited.
**A benchmark parameter left at its default across an entire campaign is an untested
assumption wearing the costume of a measurement.**
