"""The search loop: propose -> gate -> measure -> record -> select, over repo kernels.

A single-process port of the ben branch's parametric loop (bench/loop.py). It does NOT
invent kernels -- it searches the knobs the dispatch exposes (use_graph x compute dtype),
measures each point against torch.compile with a correctness gate, and records every
evaluation (including failures) to an append-only ledger with git provenance. The best
correct point per config, promoted only if it beats the incumbent by more than the noise
floor, becomes the measured dispatch table.

The multi-agent orchestration in ben (orchestrator owns the GPU lock, expanders write
kernels, verifiers break claims) is out of scope here; this is the measurement+select core
that a single agent runs. One GPU => everything is serialized in one process, which is the
same guarantee the lock gives.

    python tests/manual/search_loop.py --ids 2 11 13     # screen a representative subset
    python tests/manual/search_loop.py                   # full matrix
"""
import argparse
import importlib.util
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_guard import exclusivity_record, require_exclusive

from ratchet.kernels.dispatch import MATRIX
from ratchet.kernels.graphed import graphed_forward

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "ledger" / "bench_results.jsonl"
BENCH = REPO / "benchmarks/reference/torch_transformer_benchmark.py"
spec = importlib.util.spec_from_file_location("tt_bench", BENCH)
mod = importlib.util.module_from_spec(spec)
sys.modules["tt_bench"] = mod
spec.loader.exec_module(mod)

# The search space: real knobs the dispatch exposes. Every value is plausible on SOME shape
# -- that is the point of searching rather than hardcoding.
SPACE = {
    "use_graph": [True, False],
    "dtype": ["float16", "bfloat16"],
}
NOISE_FLOOR = 0.03          # promote only on a margin beyond run-to-run spread
_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
    except Exception:
        return ""


def _provenance():
    return {"sha": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "gpu": exclusivity_record()}


def _append(row):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def enumerate_space():
    keys = sorted(SPACE)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(SPACE[k] for k in keys))]


def evaluate(point, cfg, base, x, ref, t_compile, prov):
    """Measure one point on one config. Records a ledger row; returns speedup or None."""
    row = {"ts": datetime.now(timezone.utc).isoformat(), "config_id": cfg.id,
           "candidate": "graphed", "point": point, "provenance": prov,
           "config": {"batch": cfg.batch_size, "d_model": cfg.d_model, "heads": cfg.heads,
                      "seq": cfg.seq_len, "layers": cfg.layers, "ffn": cfg.ffn_dim}}
    try:
        tc = mod.TransformerConfig(batch_size=cfg.batch_size, seq_len=cfg.seq_len,
                                   d_model=cfg.d_model, num_heads=cfg.heads,
                                   ffn_dim=cfg.ffn_dim, num_layers=cfg.layers, causal=True)
        cand = mod.UserOptimizedTransformer(tc).to("cuda").eval()
        mod.copy_model_weights(base, cand)
        cand._lp = _DT[point["dtype"]]
        cand._use_graph = point["use_graph"]
        cand.forward = graphed_forward.__get__(cand, type(cand))
        with torch.no_grad():
            got = cand(x, None).float()
        abs_err = (got - ref).abs()
        ok = bool(((abs_err <= 2e-3) | (abs_err <= 2e-2 * ref.abs())).all())
        if not ok:
            row.update(status="incorrect",
                       correctness={"max_abs": float(abs_err.max()), "passed": False})
            _append(row)
            return None
        t = triton.testing.do_bench(lambda: cand(x, None), warmup=25, rep=100)
        sp = t_compile / t
        row.update(status="ok", correctness={"passed": True},
                   timing={"ms": t, "speedup_vs_compile": sp})
        _append(row)
        return sp
    except Exception as e:
        row.update(status="infeasible", notes=f"{type(e).__name__}: {str(e)[:120]}")
        _append(row)
        return None


def run(config_ids):
    prov = _provenance()
    table = {}
    for cfg in MATRIX:
        if cfg.id == 14 or (config_ids and cfg.id not in config_ids):
            continue
        tc = mod.TransformerConfig(batch_size=cfg.batch_size, seq_len=cfg.seq_len,
                                   d_model=cfg.d_model, num_heads=cfg.heads,
                                   ffn_dim=cfg.ffn_dim, num_layers=cfg.layers, causal=True)
        base = mod.BaselineTransformer(tc).to("cuda").eval()
        x = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model, device="cuda")
        with torch.no_grad():
            ref = base(x, None).float()
        compiled = torch.compile(base, mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            for _ in range(3):
                compiled(x, None)
        t_compile = triton.testing.do_bench(lambda: compiled(x, None), warmup=25, rep=100)

        best_sp, best_pt = 0.0, None
        for point in enumerate_space():
            sp = evaluate(point, cfg, base, x, ref, t_compile, prov)
            tag = f"{sp:.2f}x" if sp else "FAIL"
            print(f"  cfg{cfg.id:<2} {point} -> {tag}")
            # promotion needs a margin over the incumbent, else it is measuring noise
            if sp and sp > best_sp * (1 + NOISE_FLOOR):
                best_sp, best_pt = sp, point
        table[cfg.id] = {"point": best_pt, "speedup": best_sp}
        print(f"cfg{cfg.id}: BEST {best_pt} @ {best_sp:.2f}x vs compile")

    print("\n=== measured dispatch table (best correct point per config) ===")
    sps = [v["speedup"] for v in table.values() if v["speedup"] > 0]
    for cid, v in sorted(table.items()):
        print(f"  cfg{cid}: {v['point']}  {v['speedup']:.2f}x")
    if sps:
        geo = torch.tensor(sps).log().mean().exp().item()
        print(f"geomean vs torch.compile = {geo:.3f}x | ledger: {LEDGER}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    ap.add_argument("--allow-contention", action="store_true",
                    help="measure even if another process holds the GPU (rows stay marked)")
    args = ap.parse_args()
    require_exclusive(args.allow_contention)
    run(set(args.ids))
