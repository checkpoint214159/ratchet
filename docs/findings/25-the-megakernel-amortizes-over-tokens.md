# 25 — The first hand-written kernel wins where there is work and loses where there is not

**Date:** 2026-08-30. **Candidate:** `v16_ffn_megakernel` (gen 16, parent v9b, CMP draw seed 30).
**Idea:** proposal C-02, from the GPU MODE research agent.

## Result

    v16 geomean vs compiled   2.493x      (v13 frontier 2.711x, v9b parent 2.655x)

**v16 is not the frontier.** But the per-config structure is the finding, not the geomean:

    cfg  6  (M=1.28M)   -7.6% vs v13     the largest config in the matrix
    cfg  1  (M=8192)    -6.7%
    cfg  7  (M=8192)    -5.7%
    cfg 13  (M=65536)   -2.5%
    cfg  4  (M=2048)    +0.8%            break-even
    cfg 12  (M=2048)    +1.4%
    cfg  3  (M=512)    +49.4%            loss
    cfg  2  (M=128)   +113.0%            collapse

## Why: the kernel amortizes weight loading over the token axis

Every program loads **both** weight matrices — 64 KB at d_model=128 — into shared memory,
then streams activation tiles past them. Weight traffic per token is `2*D*F*elem / M`:

    M = 128      512 bytes/token of weights   vs ~768 bytes/token of activation
    M = 1.28M    0.05 bytes/token

At config 2 the kernel spends more traffic on weights than on the data, and launches 2 CTAs
onto 66 SMs. There is nothing to amortize over. This is not a compilation problem —
`dynamo.explain` shows `graph_count=1, graph_breaks=0` for both v13 and v16, so the custom
Triton call is captured cleanly.

## L33 held, quantitatively

The docstring predicted this before the sweep ran: op-level the kernel is **4.59x on config
6's FFN**, but the FFN is ~15-21% of layer time, so the end-to-end win must be ~1.06x. It
measured **1.076x**. The isolated number was not wrong, it was *diluted by exactly the
predicted factor* — which is the first time L33 has been used as a forward prediction rather
than a retrospective excuse, and it worked.

## What is actually good here

  * A hand-written kernel beat the compiler on the biggest config in the matrix. That is the
    first time anything in this project has left the PyTorch-composition plateau and gained
    something the compiler could not.
  * It is **more accurate** than the path it replaces (1.13e-04 vs 2.35e-04 against the fp32
    reference), because `h` never rounds to fp16 between the GEMMs. It returns tolerance
    margin instead of spending it.
  * The smem predicate correctly declined config 8 (d_model=1024 needs 4.25 MB) and the
    fallback stayed correct.

## The obvious next move, and why it is not cheating

Dispatch: use the fused kernel where weight loading amortizes, fall back to the frontier's
path where it does not. The predicate must be a function of shape and measured device
properties — never a config id (rule 2). The honest form is a ratio, not a token count:

    weight_bytes / activation_bytes_per_token <= FRACTION * M

i.e. require weight traffic to be a small fraction of activation traffic. That expression
contains no benchmark knowledge; it would evaluate correctly on a shape and a card nobody
here has seen, which is the test v14_dispatch was built to satisfy.

## L37 — A kernel that hoists an invariant has a minimum problem size, and it is derivable

The megakernel's whole advantage is loading weights once and reusing them. That advantage is
proportional to how much work reuses them, so its speedup is a function of M with a
computable crossover — not a property of the kernel to be discovered by trying every config.
**Any optimization that hoists something out of a loop should have its break-even size
derived and turned into the dispatch predicate, before it is measured everywhere.**

We had the number needed to predict this before running the sweep: weight bytes, activation
bytes per token, and measured bandwidth. The sweep confirmed a crossover we could have
stated in advance, and the four configs below it cost a full 112-second measurement to learn
something arithmetic.

---

## Addendum, same day — v17 confirms the crossover

`v17_dispatched_megakernel` (gen 17, parent v13, a recombination merge) applies the g16
kernel only above the amortization threshold. Measured:

    cfg  6   70.558 -> 65.034 ms   -7.8%   FUSED
    cfg  7    0.125 ->  0.114 ms   -9.0%   FUSED
    cfg 13    3.362 ->  3.272 ms   -2.7%   FUSED
    all ten other configs                  v13 path, all within noise

    geomean vs compiled   v13 2.711x  ->  v17 2.758x   (+1.7%)
    TOTAL wall time       v13 82.6 ms ->  v17 77.0 ms  (-6.9%)

**The prediction stated in v17's docstring before the sweep ran was "roughly 2.745x, and
that is inside the noise floor". It measured 2.758x.** Two forward predictions in two
generations (L33's dilution factor, then this) have now landed, which is a different
epistemic position from the first fourteen generations, where every number was a surprise.

### How to report this honestly

  * The **geomean gain is +1.7%, inside the +/-7% noise floor.** It is not a geomean win
    and must not be quoted as one.
  * The **per-config wins on 6 and 7 (-7.8%, -9.0%) are at or beyond the noise boundary**
    and are supported by a mechanism with a derived crossover, not by a lucky draw.
  * The **total-wall-time reduction of 6.9%** is the number with real content, because
    config 6 alone is 85% of the matrix's wall time. Which of these three framings counts
    depends on an objective the organizers have not published, and the report must give
    all three rather than pick the flattering one.

v17 is the new frontier and is merged. It is the first frontier advance since generation
12, and the first one in this project's history that came from a kernel we wrote rather
than from arranging kernels somebody else wrote.
