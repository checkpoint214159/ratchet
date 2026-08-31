"""Emit clade metaproductivity (CMP) as JSON, for the dashboard's evolution tree.

CMP is what the loop's parent selection actually runs on: Thompson sampling draws
Beta(1 + wins, 1 + failures) per candidate over the POOLED stats of its declared-lineage
subtree (bench/ledger.py::sample_candidate). The dashboard shows those posteriors, so
the numbers here must be the ledger's own, not a re-derivation: `pooled` comes straight
from bench.ledger.clade_stats_by_candidate.

`own` (this candidate's rows only, before pooling over descendants) is not exposed by
bench/ledger.py, so its ~15-line stage is mirrored here VERBATIM against the same
helpers (compiled_baseline_ms, declared_lineage, CLADE_NOISE). If the criterion in
clade_stats_by_candidate changes, change this mirror with it -- the guard is that
pooling this file's `own` over candidate_descendants must reproduce `pooled` exactly,
and that identity is asserted below on every run rather than trusted.

Read-only: BenchLedger is opened on the ledger path only to read; nothing is written.

Usage: python3 dashboard/server/cmp.py [results_path]
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> None:
    from bench.ledger import (
        BASELINE,
        BASELINE_COMPILED,
        CLADE_NOISE,
        BenchLedger,
        candidate_descendants,
        clade_stats_by_candidate,
        compiled_baseline_ms,
        declared_lineage,
    )

    path = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "bench" / "results.jsonl")
    ledger = BenchLedger(path)

    pooled = clade_stats_by_candidate(ledger)

    # -- the `own` stage, mirrored from clade_stats_by_candidate ------------------
    compiled = compiled_baseline_ms(ledger)
    lineage = declared_lineage()
    best: dict[str, dict[int, float]] = {}
    for r in ledger.clean_rows():
        if r.get("candidate") in (BASELINE, BASELINE_COMPILED) or r.get("status") != "ok":
            continue
        if "padding_ratio=0.0" not in (r.get("notes") or ""):
            continue
        ms = (r.get("timing") or {}).get("candidate_ms")
        if not ms or not (r.get("correctness") or {}).get("passed"):
            continue
        cfg = best.setdefault(r["candidate"], {})
        if r["config_id"] not in cfg or ms < cfg[r["config_id"]]:
            cfg[r["config_id"]] = ms

    own: dict[str, tuple[int, int]] = {}
    for name, per_cfg in best.items():
        parent = lineage.get(name)
        w = f = 0
        for cid, ms in per_cfg.items():
            cb = compiled.get(cid)
            pms = best.get(parent, {}).get(cid) if parent else None
            if cb and ms < cb and (pms is None or ms < pms * (1.0 - CLADE_NOISE)):
                w += 1
            else:
                f += 1
        own[name] = (w, f)

    # The mirror is only trustworthy if pooling it reproduces the ledger's pooled
    # stats. Cheap to check, catastrophic to drift on -- so it is checked every run.
    for name in own:
        w = f = 0
        for d in candidate_descendants(name):
            dw, df = own.get(d, (0, 0))
            w, f = w + dw, f + df
        if (w, f) != pooled.get(name):
            raise AssertionError(
                f"cmp.py own-stage mirror has drifted from bench.ledger for {name}: "
                f"pooled from own = {(w, f)}, ledger says {pooled.get(name)}")

    print(json.dumps({
        "clade_noise": CLADE_NOISE,
        # A success is a pad-0.0 row that beats the compiled baseline AND improves on
        # the declared parent's best for that config by more than CLADE_NOISE.
        "criterion": (
            f"pad-0.0 rows only; win = beats compiled baseline AND declared parent's "
            f"best by >{CLADE_NOISE:.0%}; sampler draws Beta(1+W, 1+F) over pooled"),
        "by_candidate": {
            name: {"own": list(own.get(name, (0, 0))), "pooled": list(pooled[name])}
            for name in pooled
        },
    }))


if __name__ == "__main__":
    main()
