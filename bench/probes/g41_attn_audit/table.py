"""Fold the two independent runs of `probe_three_arms.py` into one table.

Replication is the whole point (finding 49: two candidates read 46% apart on config 3 and
matched to the hundredth of a microsecond at 200 warmup). Every number in finding 51's
table comes out of here, so the write-up cannot quote a run that was not repeated.

    python3 bench/probes/g41_attn_audit/table.py run1.json run2.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from bench.matrix import BY_ID


def best(rows, form):
    ok = [r for r in rows if r["form"] == form and r["status"] == "ok"]
    return min(ok, key=lambda r: r["hot_ms"]) if ok else None


DECISIVE = 0.10          # v23's, inherited unchanged


def main() -> int:
    runs = [json.loads(Path(p).read_text()) for p in sys.argv[1:]]
    hdr = (f"{'cfg':>3} {'B':>6} {'H':>3} {'hd':>4} {'S':>5}  {'runs today':<22}"
           f"{'incumbent':>10}{'single':>9}{'looped':>9}{'sdpa':>10}"
           f"{'sdpa vs inc':>12}  {'fastest':<22} clears 10%?")
    print(hdr)
    print("-" * len(hdr))
    for cid in sorted(int(k) for k in runs[0]["configs"]):
        cfg = BY_ID[cid]
        for i, run in enumerate(runs):
            d = run["configs"][str(cid)]
            m, rows = d["meta"], d["rows"]
            bs = best(rows, "single_tile")
            bl = best(rows, "looped")
            sd = best(rows, "sdpa")
            inc = m["incumbent_ms"] * 1e3
            s = bs["hot_ms"] * 1e3 if bs else float("nan")
            lp = bl["hot_ms"] * 1e3 if bl else float("nan")
            sp = sd["hot_ms"] * 1e3
            cands = {"single_tile": s, "looped": lp, "sdpa": sp}
            wv, win = min((v, k) for k, v in cands.items() if v == v)
            tiles = {"single_tile": bs and tuple(bs["tile"]),
                     "looped": bl and tuple(bl["tile"]), "sdpa": ()}
            win = f"{win}{tiles[win] if tiles[win] else ''}"
            # The decision-relevant fact is not "who is fastest" but "does the fastest
            # beat WHAT THE MODEL RUNS by more than the timer can resolve".
            if win == m["incumbent"]:
                clears = "IS the incumbent"
            elif wv < inc * (1.0 - DECISIVE):
                clears = f"YES -- {inc/wv:.3f}x, would displace"
            else:
                clears = "no -- incumbent holds"
            head = (f"{cid:>3} {cfg.batch_size:>6} {cfg.heads:>3} {cfg.head_dim:>4} "
                    f"{cfg.seq_len:>5}  {m['incumbent']:<22}" if i == 0
                    else " " * 46)
            print(f"{head}{inc:>10.3f}{s:>9.3f}{lp:>9.3f}{sp:>10.3f}"
                  f"{inc/sp:>12.3f}  {win:<22} {clears}")
    print("\n`sdpa vs inc` = incumbent_time / sdpa_time, i.e. THE VENDOR'S SPEEDUP over "
          "what the model runs today.\n              >1 means sdpa+repack is faster and "
          "the predicate is letting a losing kernel fire.\ntwo rows per config = two "
          "independent runs. `nan` = that form declines this shape.\n`clears 10%?` "
          "applies v23's inherited DECISIVE margin to the fastest arm against the "
          "incumbent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
