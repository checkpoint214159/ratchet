# Onboarding

Start here. Fifteen minutes to be useful, and the order matters.

## What this is

Ratchet optimizes a transformer inference workload and **keeps every experiment**, so a
number can always be traced to the code that produced it. It runs as an agent loop: an
orchestrator that owns the GPU and the ledger, exploration agents that propose, and
implementation agents that build — described in `docs/loop/architecture.md`.

**Current state** (2026-08-30): frontier `v26_causal_correct` at **3.103x geomean** over
the `torch.compile` baseline, 13 of 14 announced configs passing (config 14 is where the
*reference* OOMs), on an RTX 4070 Ti SUPER (sm_89). 550 measured rows, 28 candidate
branches, 36 findings, 38 running learnings.

## Read in this order

1. **[docs/loop/method.md](docs/loop/method.md)** — 24 rules distilled from everything
   measured so far. Most of them cost a day each to learn. **Do not skip this.**
2. **[docs/loop/architecture.md](docs/loop/architecture.md)** — the orchestrator and its
   subagents, and why one GPU forces that shape.
3. **[docs/loop/runbook.md](docs/loop/runbook.md)** — one turn of the loop, as commands.
4. `docs/00-mission.md` — the hardware truth table. Every headroom argument cites it.
5. `bench/README.md` — why git *is* the evolutionary tree, and the ledger's row shape.
6. `docs/findings/00-learnings.md` — the loop's long-term memory. Newest last.

Then, when you need them: `specs/07-proposal-rubric.md` (how ideas are scored into a
sampling distribution) and `docs/findings/README.md` (the index of what has been
established, so you do not re-derive it).

## Handing this to your agent

Point it at **[docs/loop/roles/](docs/loop/roles/)** — orchestrator, researcher, expander,
verifier — plus `docs/loop/method.md`. Those five files are the whole contract; they are
written to be pasted.

`CLAUDE.md`, `AGENTS.md`, `.github/copilot-instructions.md`, `.cursor/` and `.codex/` are
**generated shims** from `.beryl/agent/tool-instruction-template.md`. Do not hand-edit
them; edit the template and run `.beryl/agent/scripts/sync-agent-env.sh`.

**Do not keep findings in your agent's private memory.** Anything that should change what
the loop does next belongs in `docs/findings/00-learnings.md`, where the next person's
agent can read it. Private memory is where knowledge goes to be lost.

## Two lanes, and the boundary between them

They look similar and are not interchangeable.

| | `research/` | `bench/` |
|---|---|---|
| what | the fail-closed evidence archive | working measurements on real hardware |
| gate | per-vendor qualification (`FG-01`); today nothing can be marked `QUALIFIED`, so it admits only `NO_RUN` and `SYNTHETIC` | the harness's own correctness gate |
| why | authored on a machine with no GPU; it must never fabricate a number | one machine *does* have a CUDA device and there is a deadline |
| output | `research/paper/latest.pdf` | `bench/results.jsonl` + `docs/findings/` |
| env | `.venv` (CPU-only by default) | system python 3.10, torch 2.8.0+cu128, triton 3.4.0 |

**Nothing in `bench/` is promoted into `research/archive/` implicitly.** Promotion needs
the NVIDIA qualification hierarchy written and ratified. Until then the two lanes stay
adjacent and honest about which is which. The loop documented in `docs/loop/` is the
`bench/` lane.

## Where things live

```
docs/loop/            the method, the roles, the runbook   ← this package
docs/findings/        NN-*.md, one per established fact; 00-learnings.md is the memory
docs/00-mission.md    the calibrated hardware table
specs/                the loop's design contracts (03 search, 04 dispatch, 07 rubric)
bench/matrix.py       the 14 announced configs as executable data — cite, never restate
bench/results.jsonl   append-only measurements, keyed to commit sha. Never edited.
bench/ledger.py       scoreboard, clade metaproductivity, Thompson parent sampling
bench/screen.py       stage-1 kill, 30 s, advisory log only
bench/run_matrix.py   stage-2 confirm, the only sanctioned path to a ledger row
bench/gpu_lock.py     exclusive GPU access; every measuring tool must take it
bench/candidates/     every candidate ever measured, kept, with declared lineage
bench/kernels/        Triton kernels (they need real source files — no heredocs)
tests/bench/          lineage/ledger/screen invariants + per-candidate contracts
ratchet/oracle/       Zone A, immutable, manifest-checked by scripts/check-oracle.sh
research/             the other lane — see above
dashboard/            a viewer over the ledger and the tree
```

## Setup on a CUDA machine

```bash
python3 -m ratchet.oracle.device      # calibrate — writes ledger/device.json (~10 s GPU)
./scripts/check-oracle.sh             # must pass before and after every session
python3 bench/matrix.py               # the announced configs + derived feasibility
python3 bench/ledger.py               # the current frontier
python3 -m pytest tests/bench/ -q     # the loop's own invariants
```

**Your card is not this card.** The numbers in `docs/00-mission.md` are an RTX 4070 Ti
SUPER; re-calibrate and re-measure your own noise floor before trusting any margin. Ours
went from an assumed 3% to a measured ±7%, which changed which candidates counted as wins.

## Things that will bite you

- **Clocks are not lockable under WSL2** (`nvidia-smi -lgc` fails) → minimum-of-N timing,
  arms interleaved. `ncu` is also unavailable (`ERR_NVGPUCTRPERM`), so profiling is by
  contrast, not by counters.
- **`@triton.jit` cannot be defined in stdin or a heredoc** — the JIT needs a real source
  file (`OSError: could not get source code`). Always write kernels to files.
- **`do_bench(warmup=25, rep=100)` takes milliseconds of budget, not iteration counts.**
  `do_bench_cudagraph` does not flush L2 and amortizes launch — never compare the two
  directly.
- **Python is 3.10**, not the 3.11+ some contracts assume. Avoid 3.11-only syntax.
- **`git add -A` has swept a subagent's in-progress files into a commit here.** Stage
  explicit paths.
- **Never rebase, squash, amend or force-push a candidate branch.** It silently reparents
  the tree and invalidates every statistic derived from it.
- `gh` is not installed on the CUDA machine.
