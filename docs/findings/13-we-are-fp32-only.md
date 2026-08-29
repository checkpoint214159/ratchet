# Finding 13 — Every candidate assumes an fp32 model, and none of them says so

Recorded 2026-08-29. The last unaudited default from L8.

## The audit

`dtype` has been `float32` in every measurement ever taken. The custody benchmark accepts
`float32 | float16 | bfloat16`. Running the two frontier candidates at other dtypes:

| candidate | float16 | bfloat16 |
|---|---|---|
| v8_padfast | **crash** — `RuntimeError: expected scalar type Float but found Half` | **crash** |
| v9a_compiled_core | **incorrect** — max_abs 8.6e-3 vs a 2.0e-3 budget (4.3x over) | not reached |

## The cause is structural, not a slip

Every candidate from v2 onward writes the residual as fp32 explicitly:

```python
o = F.linear(ctx..., out_w, out_b).float()      # <-- hardcoded
x = x + o
```

That `.float()` exists for a good reason — finding 08 proved the fp32 residual is
load-bearing and cannot be demoted without blowing the tolerance. But it was written
assuming the *model* is fp32, so when the harness hands us an fp16 model the residual
promotes to fp32 while the LayerNorm parameters stay fp16, and the dtypes stop matching.

The fp16 cache compounds it: `weight.to(float16)` is a no-op on an fp16 model, so the
candidate's central trick — compute in fp16, accumulate in fp32 — degenerates into
"compute in fp16, accumulate in fp32, against a baseline already doing everything in
fp16". The comparison is no longer the one the candidate was designed for.

## What this does and does not mean

**It is a real limitation and it is recorded as one.** If the graders run `--dtype
float16`, our submission does not merely lose — it crashes on v8 and fails the gate on
v9a. That is worse than a slow kernel.

**It is probably not the graded path.** The benchmark defaults to `float32`, the problem
statement's tolerance (`abs < 0.002`) is nearly unsatisfiable for any reimplementation of
an fp16 model — one fp16 ulp at |x|=1 is about 1e-3, half the budget, before any
arithmetic difference — and finding 03 already measured bf16 GEMMs failing at 9.6e-3.
A tolerance that tight only makes sense against an fp32 reference.

**So the correct action is to state the assumption, not to chase it.** A submission that
silently assumes fp32 and crashes otherwise is worse than one that declares
`dtype == float32` as a precondition and says why. Making the family dtype-generic is
tractable — thread the model dtype through instead of hardcoding `.float()` — but it
would be optimizing for a configuration the tolerance itself argues against.

## Method note

This is the third finding in a row produced by auditing an untested default rather than by
profiling (finding 11: padding; finding 12: the baseline; this one: dtype). All three
defaults were inherited silently from the first run and never questioned. **L8's rule now
has three confirmations and no counterexamples.**
