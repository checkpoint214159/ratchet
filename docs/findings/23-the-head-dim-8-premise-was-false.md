# 23 — The head_dim=8 premise was false, and it had steered the project for a week

**Date:** 2026-08-30. **Corrected:** `bench/matrix.py`. **Found by:** the GPU MODE research
agent, verified independently here before the correction was made.

## The claim that was wrong

`bench/matrix.py`, written when the matrix was transcribed on 2026-08-29, asserted:

> the two head_dim=8 rows (#7, #11) are the awkward ones: cuDNN and FlashAttention
> typically support {32, 64, 128, 256} and may silently fall back to a slow path there.
> That is a dispatch branch, and possibly the one place a hand-written Triton kernel
> genuinely earns its keep.

That comment lives in this project's single source of truth. It was cited into the
proposal rubric (spec 07 scores B4 at +5 for the head_dim=8 regime "where vendor fast
paths may refuse"), into three research-agent briefs, and into every discussion of where
to look next. It was never checked.

## Measured

    head_dim     flash   mem_eff     cudnn      math
           8        ok        ok        ok        ok
          16        ok        ok        ok        ok
          32        ok        ok        ok        ok
          64        ok        ok        ok        ok
         128        ok        ok        ok        ok
         256        ok        ok   REFUSES        ok

Every backend accepts head_dim=8. The one refusal in the whole matrix is cuDNN at
head_dim=**256** — the opposite end of the range, and it lands on **config 8**
(d_model 1024, 4 heads), a config nobody was watching for backend reasons.

## The region is still the prize, for a different reason

sm_89's tensor-core instruction is `m16n8k16`, so Triton's `tl.dot` requires K >= 16.
At head_dim=8 the contraction dimension is half the instruction's width, so the vendor
kernel's tiling is mismatched to the hardware — not refused, just wrong-shaped. Padding D
to 16 *inside* a kernel costs nothing. Padding it in HBM is numerically exact but measured
**1.2-2.7x slower**, which closes that variant without spending a generation discovering
it.

## L35 — A premise written into the source of truth is never re-derived, so it must be measured when written

The claim was plausible, load-bearing, and cost about 40 seconds to check. It survived a
week and propagated into a rubric dimension and three agent briefs because it sat in a
file whose whole purpose is to be cited rather than questioned — the same file whose
docstring says "cite it, never restate the numbers."

The rule this project already applies to measurements now applies to premises: **an
uncited claim in `matrix.py` is as dangerous as an uncited number in the ledger, and more
durable, because everything downstream treats it as settled.** Every remaining assertion
in that file that is not a transcription of the problem statement should be either
measured or marked UNVERIFIED.

Related: L34 warned that agreement between analysts sharing a method is not replication.
This is the mirror case — a single unexamined assertion, shared by everyone downstream,
produced the same false confidence with no agreement at all.
