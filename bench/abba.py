"""ABBA-interleaved timing of N candidates in ONE process, with the cold round discarded.

WHY THIS EXISTS AND WHAT IT REPLACES
------------------------------------
Finding 42 established that none of the three protocols in play measures the same thing:
`run_matrix` times arm A, then COMPILES and autotunes arm B, then times arm B, on a card
whose clocks are not lockable -- and that inverted two signs. The graded harness
interleaves, but its BASELINE arm (byte-identical reference code) spreads 33-39% on the
sub-millisecond rows where all the remaining score lives, because it warms up 20 times
against a settling time of ~130 calls after CUDA-graph capture.

What DID work, and what this file is: **every arm resident in one process, round-robin
interleaved with the order reversed on alternate rounds, the cold round discarded, and
the minimum of the remaining rounds kept.** It was used for v36 (+0.082 of
weighted_score, no regression on any config) with configs 8 and 13 running byte-identical
code as an in-run control that put the floor at +/-0.4%.

WHAT IT IS NOT
--------------
It is not the ledger and it does not write to it. `bench/results.jsonl` is produced by
`bench/run_matrix.py`, one config per subprocess, and that stays the record. This is the
instrument for RANKING two candidates, which finding 42 showed neither of the other two
protocols can do. [L41]: a probe may propose; it may never conclude.

RANK BY THE CANDIDATE'S OWN TIME, NEVER BY A PER-RUN SPEEDUP. The reported speedup
inherits a noisy denominator (finding 42's addendum). This file therefore never computes
one; it prints per-arm times and their ratios to each other.

CORRECTNESS BEFORE TIMING. Every arm is checked against a fresh reference at the locked
tolerance before any round runs, and an arm that fails is timed anyway but reported as
FAILED so its number can never be quoted as a win.

    python3 bench/abba.py --ids 3 9 12 --arms v34_launch_bound v36_gemm_gelu v37_recombined2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

RTOL, ATOL = 0.02, 0.002          # the harness's own defaults. Never widened.


def _reference(tag: str):
    spec = importlib.util.spec_from_file_location(
        tag, REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def _timed(model, x, m, n: int) -> list[float]:
    """Per-call milliseconds, measured EXACTLY the way the graded harness measures.

    `benchmark_once` in the reference allocates all 2N events up front, records them
    around N un-synchronized calls and synchronizes ONCE at the end. Syncing per call
    instead would insert a host round trip into every iteration -- which on the
    sub-millisecond, CPU-bound rows this file exists to rank is a large fraction of the
    quantity being measured. Matching the scored loop is the whole point (finding 42).
    """
    import torch
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    with torch.inference_mode():
        torch.cuda.synchronize()
        for i in range(n):
            starts[i].record()
            model(x, m)
            ends[i].record()
        torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def run_config(config_id: int, arms: list[str], rounds: int, iters: int,
               warmup: int) -> dict:
    import torch
    sys.path.insert(0, str(REPO))
    from bench.candidates import REGISTRY
    from bench.matrix import BY_ID

    cfg = BY_ID[config_id]
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    ref = _reference(f"ref_abba_{config_id}")
    tcfg = ref.TransformerConfig(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, d_model=cfg.d_model,
        num_heads=cfg.heads, ffn_dim=cfg.ffn_dim, num_layers=cfg.layers,
        causal=cfg.causal)
    tcfg.validate()

    torch.manual_seed(1234)
    base = ref.BaselineTransformer(tcfg)
    models = {}
    for name in arms:
        mdl = REGISTRY[name].build(ref.BaselineTransformer)(tcfg)
        ref.copy_model_weights(base, mdl)
        models[name] = mdl.to(device=dev, dtype=torch.float32).eval()
    base = base.to(device=dev, dtype=torch.float32).eval()
    x, m = ref.generate_random_case(tcfg, dev, torch.float32, seed=1234,
                                    padding_ratio=0.0, input_scale=1.0)

    # ------------------------------------------------ correctness, before any timing
    notes, ok = {}, {}
    with torch.inference_mode():
        want = base(x, m)
        for name, mdl in models.items():
            res = ref.compare_outputs(want, mdl(x, m), rtol=RTOL, atol=ATOL)
            ok[name] = bool(res.passed)
            notes[name] = {
                "passed": bool(res.passed),
                "max_abs": float(res.max_abs_error),
                "launch_reason": getattr(mdl, "launch_reason", None),
                "gemm_reason": getattr(mdl, "gemm_reason", None),
                "gemm_sites": list(getattr(mdl, "gemm_sites", ()) or ()),
                "gemm_engaged": getattr(mdl, "gemm_engaged", None),
                "stream_path": getattr(mdl, "stream_path", None),
                # WHAT THE ARM ACTUALLY RAN, recorded next to what it measured. [L36]:
                # a mechanism has to be shown to engage, and until g42 this file could
                # rank two attention plans without recording which two they were -- so a
                # row where both arms happened to pick the same tile was indistinguishable
                # from a row where they differed. Cheap, and it turns "these are controls"
                # from an assumption into a field.
                "attn_form": getattr(mdl, "attn_form", None),
                "attn_tile": list(getattr(mdl, "attn_tile", ()) or ()),
                "attn_reason": getattr(mdl, "attn_reason", None),
            }

    # ------------------------------------------------------------- settle, then time
    for name in arms:
        _timed(models[name], x, m, warmup)

    per_round: dict[str, list[dict]] = {n: [] for n in arms}
    for r in range(rounds):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for name in order:
            ts = _timed(models[name], x, m, iters)
            per_round[name].append({"round": r,
                                    "median": statistics.median(ts),
                                    "min": min(ts)})
    # Round 0 is DISCARDED: after graph capture it takes ~130 calls before the numbers
    # mean anything (finding 42's addendum measured 932.9 us against a settled 250.9).
    kept = {n: [row for row in rows if row["round"] > 0] for n, rows in per_round.items()}
    result = {
        "config_id": config_id, "rounds": rounds, "iters": iters, "warmup": warmup,
        "arms": arms,
        "median_of_min_rounds": {n: min(r["median"] for r in kept[n]) for n in arms},
        "min_of_min_rounds": {n: min(r["min"] for r in kept[n]) for n in arms},
        "per_round": per_round, "correctness": notes,
        "all_correct": all(ok.values()),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--out", default=None)
    ap.add_argument("--_child", type=int, default=None,
                    help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._child is not None:
        print("<<<JSON>>>" + json.dumps(
            run_config(a._child, a.arms, a.rounds, a.iters, a.warmup)))
        return 0

    # One config per subprocess -- the isolation `run_matrix` exists for. The GPU lock is
    # the CALLER's job: this file measures, it does not arbitrate.
    results = []
    for cid in a.ids:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--ids", str(cid),
               "--arms", *a.arms, "--rounds", str(a.rounds), "--iters", str(a.iters),
               "--warmup", str(a.warmup), "--_child", str(cid)]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
        tag = [l for l in p.stdout.splitlines() if l.startswith("<<<JSON>>>")]
        if not tag:
            print(f"config {cid}: FAILED\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
            continue
        r = json.loads(tag[0][len("<<<JSON>>>"):])
        results.append(r)
        ref_arm = a.arms[0]
        print(f"\n=== config {cid} "
              f"{'(all arms correct)' if r['all_correct'] else '(AN ARM FAILED)'} ===")
        print(f"{'arm':<22} {'median us':>10} {'min us':>10} "
              f"{'vs ' + ref_arm:>12}  sites")
        for n in a.arms:
            med = r["median_of_min_rounds"][n] * 1e3
            mn = r["min_of_min_rounds"][n] * 1e3
            rel = r["median_of_min_rounds"][ref_arm] / r["median_of_min_rounds"][n]
            c = r["correctness"][n]
            flag = "" if c["passed"] else "  <-- ACCURACY FAILED"
            print(f"{n:<22} {med:>10.2f} {mn:>10.2f} {rel:>12.4f}x  "
                  f"{','.join(c['gemm_sites']) or '-'}{flag}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
