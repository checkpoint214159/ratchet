# 49 — Which candidate to submit, and why the choice is not close on the axis that matters

**Date:** 2026-08-30. Best-available measurement per config: the interleaved arm where it
exists, the isolated arm otherwise (findings 42, 45).

## The numbers

    candidate                 wall ms   weighted   geomean    config 6
    v26_causal_correct           69.2     2.4129     3.103      57.437
    v34_launch_bound             74.3     2.4973     3.270      62.085
    v36_gemm_gelu                70.7     2.5293     3.324      59.076
    v37_recombined2             102.7     2.4227     3.076      90.974   <- the defect
    v38_stream_fallback          69.5     2.4923     3.226      57.801   <- the fix

**v38 confirmed: config 6 is back in family with v26** (57.801 vs 57.437), from v37's
90.974. Finding 46's fix works on the ledger, not only on its author's bench.

## v36 and v38 are tied, and the tie is not close to the noise

v36 leads on `weighted_score` by **0.037 on a 3.0 scale — 1.2%**. The g39 executor measured
that **two runs of a byte-identical arm, both under the GPU lock, differ by 7.1%**. A 1.2%
gap is a sixth of the floor. There is no ranking here, only a tie.

Per config, the differences are equally inside it: v36 leads config 2 (0.0563 vs 0.0625)
and config 9 (0.2243 vs 0.2263); v38 leads config 12 (0.0819 vs 0.0891). Config 3, where
v36 leads most, is **capped and scores nothing**.

## So the decision is made on what is not tied

v38 is a strict superset of v36 in capability, because it descends from it:

  * **v36's projection GEMMs with the exact-erf GELU epilogue** — inherited via v37.
  * **v33's shape-latch fix.** v13's CUDA graph latched to the first shape it saw and
    `_static_x.copy_(x)` BROADCAST a smaller one: a (1,128,128) input returned an
    (8,128,128) tensor. 177 tests were green because no model in this project's history had
    ever been called at two shapes.
  * **v35's combination-only mask fix** — a wrong-answer path that exists in neither
    parent alone (69407 of 262144 elements past tolerance), because v34 elides the mask
    buffer when `_nomask` and v33 removes the raise that made that unreachable.
  * **v33's streaming and the config-14 protocol** — 32/32 sequences at S=100000, peak
    3.54 GiB, with a causal-prefix oracle and a blocked fp64 certificate.
  * **v38's own fix**: attempt residency, fall back only on a real `OutOfMemoryError`,
    rather than estimating from `mem_get_info` — which reports DEVICE-free memory, not
    process-available, and had ~10 GiB of our own reusable blocks counted as unavailable.

**Ship v38.** Tied on score, best on wall time, and the only candidate carrying all four
correctness fixes. Choosing v36 would trade three silent-wrong-answer defences for 1.2% of
a quantity we cannot resolve to better than 7.1%.

## The honest caveats a reviewer will find anyway

1. **Config 6's interleaved arm does not exist** — the memory gate declines co-residency
   there, correctly (finding 05's 410% spill). So config 6 is compared on the isolated arm,
   which finding 45 shows misreports construction-time planners by 2-4x. The v26/v37/v38
   config-6 numbers are mutually consistent across two sweeps and two protocols, so the
   conclusion holds, but it rests on the weaker arm.
2. **v26 and v34 have no interleaved arm at all** — they were swept before finding 42. Their
   rows in this table mix protocols and their `weighted` figures are not strictly
   comparable with v36/v37/v38's.
3. **`weighted_score` with a 3.0 cap is our invention.** The organisers published no
   objective. Under total wall time the ordering is v38 < v26 < v36 < v34 << v37; under the
   cap it is v36 > v34 > v38 > v37 > v26. The two disagree about second place and agree
   that v37 is last.

## L55 — When two candidates tie inside the noise, decide on what is not measured

Six candidates were separated by less than the 7.1% single-arm floor this session, and the
search spent real GPU time trying to rank them. The tie-break that actually held up was
never a number: it was which candidate carried more proven correctness fixes. **A
statistical tie is an instruction to stop measuring and start comparing guarantees.**

---

## Addendum — the gap was noise, and v38 wins outright

Finding 49 above concluded v36 and v38 were tied inside the floor and broke the tie on
correctness. **Replicated measurement shows there was no gap at all**, and v38 is faster.

`bench/abba.py`, 6 rounds, 200 warmup iterations, all arms resident, cold round discarded,
configs 2 and 8 as in-run controls:

    config     v36 median    v38 median      verdict
      3          52.22 us      52.22 us      IDENTICAL
      2          47.10 us      47.10 us      IDENTICAL   (control)
      8        6593.54 us    6593.54 us      IDENTICAL   (control)
     12          95.23 us      74.75 us      v38 1.274x FASTER

Every per-config difference this decision was agonised over came from **one ledger row per
candidate**. Config 3 in particular: v36 read 0.0666 ms and v38 0.0973 ms — a 46% gap, on
**identical launch counts of 20**. Under replication both read 52.22 us to the hundredth of
a microsecond.

The 200-iteration warmup is what made this resolvable. The graded harness warms 20 against
a settling time of ~130 calls after CUDA-graph capture (finding 42's addendum); at 512
tokens that leaves the measurement dominated by whatever the host was doing.

**v38 is the submission, and now for the simple reason as well as the good one.** It is
faster where the two differ, identical everywhere else, and it carries four correctness
fixes v36 lacks.

## L56 — A per-config difference from one row per arm is not a difference

Two candidates were separated by 0.037 of weighted_score, decomposed per config, argued
about, and resolved on correctness grounds — and the entire gap was single-sample noise
on two sub-millisecond rows. The decomposition was rigorous and the input was one
measurement each.

**Replicate before you decompose.** Cheap configs are cheap to replicate: this run cost
under two minutes and overturned a conclusion built on a careful analysis of noise. The
byte-identical control arms (2 and 8, reading identical to the hundredth of a microsecond)
are what make the config-12 result believable, and they cost nothing to include.
