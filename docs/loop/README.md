# docs/loop/ — how to run the research loop

The transferable core of this project. Four documents and four role prompts; nothing here
is specific to one candidate or one card except where it says so.

| file | what it is | who reads it |
|---|---|---|
| [method.md](method.md) | **24 rules distilled from 550 measured rows, 34 findings and 43 learnings.** The single most valuable file in the repo. | everyone, first |
| [architecture.md](architecture.md) | The orchestrator + researcher/expander/verifier shape, and why one GPU forces it | everyone |
| [runbook.md](runbook.md) | One turn of the loop, as exact commands | whoever is driving |
| [roles/](roles/) | Ready-to-paste prompts: [orchestrator](roles/orchestrator.md), [researcher](roles/researcher.md), [expander](roles/expander.md), [verifier](roles/verifier.md) | dispatched agents |

## The 60-second version

- **A candidate is a git commit; lineage is git ancestry.** There is no second tree store.
  Reproducing a number is `git checkout <sha>`.
- **Measurement is the scarce resource, not ideation.** One GPU, ±7% noise floor. Screen
  for 30 s, confirm for 112 s, and only the confirm enters the ledger.
- **Parents are drawn by clade, not by score** — a mediocre candidate that spawns good
  children is a good parent. This only works if you really branch from the sampled parent.
- **Correctness before timing, tolerances locked, failures recorded.** An OOM is a result.
- **The orchestrator is the only thing that measures.** Two processes on one GPU produce
  two wrong numbers.
- **Every structural claim needs an executable check.** The ones that only lived in prose
  have all, eventually, been found false.

## If you are setting this up on your own machine

1. Read `method.md`. Do not skip to the code — most of these rules cost a day each.
2. Calibrate: `python3 -m ratchet.oracle.device` writes `ledger/device.json`. Every
   headroom argument cites that table; the numbers in `docs/00-mission.md` are **this**
   card and are not yours.
3. `./scripts/check-oracle.sh` must pass before and after every session.
4. Your noise floor is not ours. Measure it from replicates on your own hardware before
   you trust any margin — ours went from an assumed 3% to a measured ±7%.
