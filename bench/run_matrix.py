"""Measure a candidate against the announced matrix and record it to the ledger.

Measurement lives IN-TREE rather than in a scratch script so that the commit sha the
ledger records actually describes the code that ran. A measurement whose provenance is a
throwaway file is not reproducible, and an irreproducible number is not evidence.

Correctness runs BEFORE timing, in the same process, and a candidate that fails is never
timed. The tolerances come from the custody benchmark's own CLI defaults; they are not
ours to choose.

    python3 bench/run_matrix.py --candidate v2_fp16_flash            # all configs
    python3 bench/run_matrix.py --candidate v2_fp16_flash --ids 6 13 # a subset
    python3 bench/run_matrix.py --candidate v2_fp16_flash --dry-run  # no ledger write

One config per subprocess: a candidate can OOM, hang, or take the CUDA context down with
it, and none of those may cost us the rest of the run. An OOM is a RESULT -- it is
recorded with `status="oom"`, because "the reference cannot run this shape" is one of the
more valuable things the matrix can tell us.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py"

# The custody benchmark's own defaults. Never widen these to make something pass.
RTOL, ATOL = 0.02, 0.002
ACCURACY_TRIALS = 5
SEED = 1234


def load_reference():
    """Import the pinned benchmark without importing it as a package or touching it."""
    spec = importlib.util.spec_from_file_location("ref_bench", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench"] = module
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# The measured work -- runs inside the child process
# ======================================================================================

def measure_one(config_id: int, candidate_name: str, samples: int = 300,
                padding: float = 0.0) -> dict:
    import torch

    sys.path.insert(0, str(REPO))
    from bench.matrix import BY_ID
    from bench.candidates import REGISTRY

    ref = load_reference()
    cfg = BY_ID[config_id]
    device = torch.device("cuda")
    dtype = torch.float32

    torch.set_float32_matmul_precision("high")     # TF32 on, for BOTH arms
    torch.backends.cuda.matmul.allow_tf32 = True

    tcfg = ref.TransformerConfig(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, d_model=cfg.d_model,
        num_heads=cfg.heads, ffn_dim=cfg.ffn_dim, num_layers=cfg.layers,
        causal=cfg.causal,
    )
    tcfg.validate()

    out: dict = {"config_id": config_id, "candidate": candidate_name,
                 "padding_ratio": padding}

    def make_input(seed):
        return ref.generate_random_case(tcfg, device, dtype, seed=seed,
                                        padding_ratio=padding, input_scale=1.0)

    def median_ms(model, x, mask, n):
        with torch.inference_mode():
            for _ in range(20):
                model(x, mask)
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            for i in range(n):
                starts[i].record()
                model(x, mask)
                ends[i].record()
            torch.cuda.synchronize()
        vals = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
        return vals[len(vals) // 2]

    # ORDER MATTERS, and it is the fix for a measurement bug we made and caught:
    # holding both models resident inflated config 6's baseline 4.1x (446 -> 1851 ms)
    # because 18.4 GB of reserved memory on a 16 GB card spills to host over PCIe.
    # So each arm is timed while it is the ONLY model on the device.
    #
    # The cost is losing cross-arm interleaving, which is our defence against thermal
    # drift on a card whose clocks cannot be locked. That trade is deliberate: the
    # memory-pressure distortion is 410%, drift between two adjacent measurements is a
    # few percent, and `interleaved: False` is recorded so nobody has to guess.
    torch.manual_seed(SEED)
    baseline = ref.BaselineTransformer(tcfg).to(device=device, dtype=dtype).eval()

    xt, mt = make_input(SEED + 100000)
    probe = median_ms(baseline, xt, mt, 3)
    n = max(11, min(samples, int(2000.0 / max(probe, 0.05))))

    torch.cuda.reset_peak_memory_stats()
    base_ms = min(median_ms(baseline, xt, mt, n), median_ms(baseline, xt, mt, n))
    base_peak = torch.cuda.max_memory_allocated() / 1e6

    # Now build the candidate and check correctness -- both models resident, which is
    # unavoidable here, but correctness is not timed so pressure cannot distort it.
    torch.manual_seed(SEED)
    fresh_baseline = ref.BaselineTransformer(tcfg)
    cand_cls = REGISTRY[candidate_name].build(ref.BaselineTransformer)
    candidate = cand_cls(tcfg)
    ref.copy_model_weights(fresh_baseline, candidate)   # CPU, fp32, before .to()
    fresh_baseline = fresh_baseline.to(device=device, dtype=dtype).eval()
    candidate = candidate.to(device=device, dtype=dtype).eval()

    trials, worst_abs, worst_rel, failed = [], 0.0, 0.0, 0
    with torch.inference_mode():
        for t in range(ACCURACY_TRIALS):
            x, mask = make_input(SEED + t)
            expected = fresh_baseline(x, mask)
            got = candidate(x, mask)
            res = ref.compare_outputs(expected, got, rtol=RTOL, atol=ATOL)
            trials.append({"passed": bool(res.passed),
                           "max_abs": float(res.max_abs_error),
                           "max_rel": float(res.max_relative_error),
                           "failed": int(res.failed_elements)})
            worst_abs = max(worst_abs, float(res.max_abs_error))
            worst_rel = max(worst_rel, float(res.max_relative_error))
            failed += int(res.failed_elements)
            del x, mask, expected, got
    passed = all(t["passed"] for t in trials)
    out["correctness"] = {"passed": passed, "max_abs": worst_abs, "max_rel": worst_rel,
                          "failed_elements": failed, "trials": len(trials)}
    if not passed:
        out["status"] = "incorrect"
        return out

    # Free every baseline before timing the candidate, for the same reason as above.
    del baseline, fresh_baseline
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    cand_ms = min(median_ms(candidate, xt, mt, n), median_ms(candidate, xt, mt, n))
    cand_peak = torch.cuda.max_memory_allocated() / 1e6

    out["status"] = "ok"
    out["timing"] = {
        "baseline_ms": base_ms, "candidate_ms": cand_ms,
        "speedup": base_ms / cand_ms, "method": "cuda_event",
        "samples": n, "reduction": "median", "interleaved": False,
        "arms_isolated": True, "clocks_locked": False,
    }
    out["memory"] = {"peak_MB": cand_peak, "baseline_peak_MB": base_peak}
    return out


# ======================================================================================
# Parent process
# ======================================================================================

def run_child(config_id: int, candidate: str, padding: float = 0.0) -> dict:
    """One config in its own process. An OOM or a crash is a result, not an exception."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child",
         "--id", str(config_id), "--candidate", candidate,
         "--padding", str(padding)],
        capture_output=True, text=True, cwd=str(REPO), timeout=3600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    tail = (proc.stderr or "")[-600:]
    status = "oom" if "OutOfMemoryError" in proc.stderr else "crash"
    return {"config_id": config_id, "candidate": candidate, "status": status,
            "correctness": None, "timing": None, "memory": None, "notes": tail.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--padding", type=float, default=0.0,
                    help="padding_ratio; every measurement before 2026-08-29 used 0.0, "
                         "which is the ONLY value where the all-True mask fast path exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json-out", action="store_true",
                    help="emit one __ROW__ json line per config, for the search loop")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--id", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        try:
            print("__RESULT__" + json.dumps(measure_one(args.id, args.candidate,
                                                        args.samples, args.padding)))
        except Exception:
            traceback.print_exc()
            return 1
        return 0

    sys.path.insert(0, str(REPO))
    from bench.matrix import MATRIX, BY_ID
    from bench.ledger import BenchLedger, provenance
    import torch

    ids = args.ids or [c.id for c in MATRIX]
    prov = provenance(str(REPO))
    if prov["dirty"]:
        print("WARNING: tree is dirty -- rows will be marked dirty and excluded from "
              "clade statistics. Commit first if these numbers are meant to count.")

    props = torch.cuda.get_device_properties(0)
    env = {"device": props.name, "cc": f"sm_{props.major}{props.minor}",
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "clocks_locked": False, "platform": "WSL2"}
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None

    # A search point is (commit, candidate, params). Without the params in the row,
    # every point the loop evaluates collapses onto one scoreboard key and the search
    # becomes unauditable after the fact.
    import os as _os
    tuned_params = {k: v for k, v in sorted(_os.environ.items())
                    if k.startswith("RATCHET_")}
    led = BenchLedger(repo=str(REPO))
    # Provenance is captured ONCE, here, and stamped on every row of this run. Calling
    # provenance() per row means a file edited while the run is in flight silently
    # changes the sha partway through -- the rows stop describing one tree state.
    run_prov = dict(prov)
    print(f"{'#':>3} {'status':<10} {'baseline':>10} {'cand':>10} {'speedup':>8}  max_abs")
    speedups = {}
    for cid in ids:
        r = run_child(cid, args.candidate, args.padding)
        t, c = r.get("timing") or {}, r.get("correctness") or {}
        sp = t.get("speedup")
        if sp:
            speedups[cid] = sp
        print(f"{cid:>3} {r['status']:<10} "
              f"{t.get('baseline_ms', float('nan')):>10.3f} "
              f"{t.get('candidate_ms', float('nan')):>10.3f} "
              f"{(sp or float('nan')):>8.2f}  {c.get('max_abs', '-')}")
        if not args.dry_run:
            row = led.record(config_id=cid, status=r["status"], candidate=args.candidate,
                             timing=r.get("timing"), correctness=r.get("correctness"),
                             memory=r.get("memory"), env=env,
                             config=BY_ID[cid].to_dict(),
                             notes=(f"padding_ratio={args.padding} " + r.get("notes", "") + (
                                 f" params={json.dumps(tuned_params)}" if tuned_params else "")
                             ).strip(),
                             provenance_override=run_prov)
        if args.json_out:
            print("__ROW__" + json.dumps({"config_id": cid, "status": r["status"],
                                          "timing": r.get("timing")}))

    if speedups:
        from bench.matrix import weighted_score
        import math
        geo = math.exp(sum(math.log(v) for v in speedups.values()) / len(speedups))
        print(f"\n{len(speedups)} config(s) measured | geomean {geo:.3f}x | "
              f"weighted score {weighted_score(speedups):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
