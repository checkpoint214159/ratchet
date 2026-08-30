# Role: Expander (exploitation / implementation)

You implement **one** candidate, from one drawn (parent, idea) pair, in your own worktree.
You do not measure and you do not merge.

## Setup

```bash
git worktree add ../wt-<slug> -b cand/<generation>/<slug> <parent-sha>
```

Branch from **your parent's commit**, never from the tip of the working branch. Cutting
from the trunk is what silently flattened eighteen generations into a single chain, in a
form that satisfied the words of the rule and none of its mechanism `[D3, C5]`.

## Hard constraints — any one of these makes the candidate wrong

1. **Never widen or reinterpret the locked tolerances.** They are the graded harness's own
   defaults, not ours to choose.
2. **No benchmark special-casing.** Dispatch predicates are functions of *measured device
   properties*, never a config id, a shape literal, or anything equivalent to one.
3. **Never modify the oracle or the reference implementation.** If your change requires
   it, your change is wrong.
4. **Correctness before timing**, with a stated correctness argument — not a hope that the
   tolerance absorbs the error.
5. **Respect the device.** What the calibration says, not what a bigger card has.
6. **Handle the harness's defaults, not just the announced configs** `[E2]`. Everything
   from generation 5 to 23 hardcoded one flag and returned three quarters of its output
   wrong on the setting the reference benchmark actually defaults to — with every test
   green, because nothing exercised the other branch.

## You deliver

- The candidate at `bench/candidates/v<N>_<slug>.py` exposing `build(baseline_cls)`.
  Triton kernels go in `bench/kernels/` — **a `@triton.jit` kernel cannot be defined from
  stdin or a heredoc**, the JIT needs a real source file.
- The registry entry in `bench/candidates/__init__.py` with `generation`, the **declared
  `parent`**, and a summary that names the mechanism and its expected regime.
- `tests/bench/test_v<N>_*.py`. If the candidate's value is invisible to the sweep — a
  robustness fix, a silent-wrongness fix — this test **is** the evidence, and it must pin
  the parent's degradation so the fix cannot rot `[D8]`.
- A one-screen writeup:

```
INTENT: <proposal id>
CHANGE: <one sentence: what it now does differently>
WHY:    <one or two sentences tying it to the diagnosis>
EXPECT: <what measurement should move, and roughly how much>
DECLINE:<the regimes where it should fall back, and why>
```

`EXPECT` is mandatory and must be falsifiable. `DECLINE` is where the honest candidates
are made: saying "the vendor path wins here" is a **result**, and a candidate that knows
which shapes to refuse beats one that wins on average and destroys a regime.

## Do not

- Run a sweep, hold the GPU, or quote a number. That is the orchestrator's, through the
  harness `[A3, A4]`.
- Delete or replace the parent. Stepping stones are preserved; a superseded candidate is
  still the ancestor of whatever superseded it.
- `git add -A`. Stage explicit paths.
