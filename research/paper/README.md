# Paper pipeline

Hand-authored inputs are `main.tex`, `sections/`, and `bibliography.bib`.
`generated/` and `latest.pdf` are generated output: regenerate them only with
`python -m ratchet.reporting build-paper`.

The selection logic accepts only keys listed in `papers_read.md` that resolve exactly
in `bibliography.bib`, and reads catalogue facts only through the verified public archive
projection. With an empty experiment catalogue, every TeX source included by `main.tex`
is checked for every empirical or comparative token; generated no-result disclosures are
the only controlled exception. The generated evidence-boundary figure derives its nine
reviewed sources and zero event count from that same verified selection. Bibliography
control words are restricted to `\url`. Tectonic runs in `--untrusted --only-cached` mode,
so a paper build neither executes trusted TeX nor fetches TeX dependencies.
