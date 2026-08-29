# Ratchet

Ratchet is a research-driven transformer optimization environment. It preserves every
validated experiment in an append-only catalogue and turns the important evidence into a
concise LaTeX research paper. The current checked-in state is **literature-only**: Intel
XPU is unavailable (`ENV-0001`), so `EXP-0001` is a verified no-run event—not a kernel,
benchmark, correctness, or speedup result.

## Fast, reproducible autoresearch setup

Requirements: Linux or WSL2, Python 3.10+, Git, and [Tectonic](https://tectonic-typesetting.github.io/)
with its local bundle available. A GPU is **not** required for this verification path.

```bash
git clone https://github.com/checkpoint214159/ratchet.git && cd ratchet
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
chmod +x scripts/verify-autoresearch.sh
./scripts/verify-autoresearch.sh
```

That single verification command is the supported low-friction entry point. It verifies
the append-only archive, rebuilds `research/paper/latest.pdf` using cached/untrusted
Tectonic, and runs every deterministic Beryl check. It fails fast with a useful error if
the virtual environment or Tectonic prerequisites are missing. It never generates a
candidate, invokes a backend, or manufactures a benchmark result.

Open `research/paper/latest.pdf` after a successful run for the current research state:
the reviewed literature, `EVT-000001` no-run evidence, its traceable future hypothesis,
and the boundary against empirical claims.

## When Intel Arc/XPU is available

Do not force a different backend through the current controller. First run:

```bash
.venv/bin/python -m ratchet.backends --backend xpu
```

A successful availability probe is **not** performance qualification. It starts the
separate, user-ratified FG-01 hardware hierarchy that must validate allocation, SDPA,
compilation, synchronization, event timing, memory observation, the authoritative
correctness matrix, and the documented steady-state methodology before any empirical
experiment can enter the catalogue.

CUDA and ROCm/HIP adapters are deliberately isolated and have analogous future probes:
`--backend cuda` and `--backend hip`. They remain untuned fallbacks until separately
qualified evidence exists.

## Research workflow

1. Add a durable human idea, constraint, priority, redirect, or reviewed literature input
   through the append-only planning queue—not by editing historic events.
2. Define a finite, canonical, scoreless parametric or architectural search plan. Search
   stays planning-only behind FG-01.
3. On qualified hardware, isolate a candidate worktree, validate correctness first, then
   measure synchronized steady-state execution and append immutable evidence.
4. Regenerate the paper. The catalogue contains every result; the paper selectively
   reports traceable conclusions and important negative findings.

Read [`docs/research-process.md`](docs/research-process.md) for the full process,
[`docs/benchmarking.md`](docs/benchmarking.md) for the measurement contract,
[`docs/hardware-support.md`](docs/hardware-support.md) for vendor gates, and
[`docs/experiments.md`](docs/experiments.md) for provenance and recovery rules.

## Agent and Beryl orientation

`AGENTS.md` routes implementation work through the Beryl initial-build or feature
workflow. The durable current architecture, vocabulary, test policy, and decisions live
under [`.beryl/agent/`](.beryl/agent/). Run `./.beryl/scripts/check.sh` after any change;
the installed-project command intentionally does not use `--development`.

`HANDOFF.md` has been retired: its useful safety principles are preserved in this README,
Beryl's canonical project context, and the documents above. Do not run `bootstrap.sh`; it
is historical setup material and refuses to run inside the working repository.
