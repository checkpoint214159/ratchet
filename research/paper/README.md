# Paper pipeline

Hand-authored inputs are `main.tex`, `sections/`, and `bibliography.bib`. `generated/` and
`latest.pdf` are generated output; regenerate them only with:

```bash
.venv/bin/python -m ratchet.reporting build-paper
```

The selection logic accepts only `papers_read.md` keys that resolve exactly in
`bibliography.bib`. It reads catalogue facts through the verified public archive projection.
The current projection contains one no-run event, `EVT-000001` / `EXP-0001`, and zero
empirical events. Generated sources disclose its environment, stop reason, literature
links, and FG-01-gated next hypothesis; no hand-authored or generated prose may represent
it as a benchmark, correctness, timing, memory, or speedup result.

The evidence-boundary figure and machine-readable `generated/catalogue.json` are rebuilt
from the same selection. Tectonic runs in `--untrusted --only-cached` mode, so a paper
build neither executes trusted TeX nor fetches TeX dependencies.
