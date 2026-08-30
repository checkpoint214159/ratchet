"""Turn the append-only ledger into the submission's results view.

Reads ledger/bench_results.jsonl and prints: the best correct point per config (the measured
dispatch table), the geomean speedup vs torch.compile, and a failure summary (what was tried
and rejected -- the loop's negative evidence). CPU only.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[2] / "ledger" / "bench_results.jsonl"

rows = [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()]
best = {}
fails = defaultdict(list)
for r in rows:
    cid = r["config_id"]
    if r.get("status") == "ok":
        sp = r["timing"]["speedup_vs_compile"]
        if cid not in best or sp > best[cid]["sp"]:
            best[cid] = {"sp": sp, "point": r["point"], "ms": r["timing"]["ms"],
                         "config": r.get("config", {})}
    else:
        fails[cid].append((r["point"], r.get("status")))

print(f"ledger rows: {len(rows)}  |  configs measured: {len(best)}\n")
print("=== measured dispatch table (best correct point per config, vs torch.compile) ===")
sps = []
for cid in sorted(best):
    b = best[cid]
    c = b["config"]
    sps.append(b["sp"])
    print(f"cfg{cid:<2} B{c.get('batch')} d{c.get('d_model')} H{c.get('heads')} "
          f"S{c.get('seq')} ffn{c.get('ffn')} | {b['point']} | {b['sp']:.2f}x  ({b['ms']:.3f} ms)")
if sps:
    geo = math.exp(sum(math.log(s) for s in sps) / len(sps))
    print(f"\ngeomean vs torch.compile = {geo:.3f}x over {len(sps)} configs")

print("\n=== rejected points (negative evidence, kept in the ledger) ===")
for cid in sorted(fails):
    uniq = sorted({f"{p['dtype']}/{'graph' if p['use_graph'] else 'nograph'}:{s}"
                   for p, s in fails[cid]})
    print(f"cfg{cid}: {', '.join(uniq)}")
