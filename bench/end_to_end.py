"""Run the custody benchmark ITSELF, with a candidate patched into its seam.

Everything in bench/results.jsonl was produced by our own timing loop
(bench/run_matrix.py). That loop deliberately differs from the benchmark's:

  * ours times each arm while it is the ONLY model resident, because holding both
    inflated config 6's baseline 4.1x through host-memory spill (finding 05);
  * the benchmark's `benchmark_models` warms up BOTH models first and keeps both
    resident for the whole run.

So our numbers and the graded numbers are produced by different protocols, and nobody has
ever checked they agree. This runs the benchmark's own `main()` unmodified -- same
warmup, same ABBA/BAAB blocks, same accuracy trials -- with `UserOptimizedTransformer`
monkeypatched to a candidate. The file on disk is never touched; it stays SHA-256 pinned.

    python3 bench/end_to_end.py --candidate v9a_compiled_core --ids 1 6
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "ref_e2e", REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_e2e"] = m
    spec.loader.exec_module(m)
    return m


def run_one(config_id: int, candidate: str, extra: list[str]) -> dict:
    sys.path.insert(0, str(REPO))
    from bench.matrix import BY_ID
    from bench.candidates import REGISTRY

    ref = load_reference()
    cfg = BY_ID[config_id]

    # Patch the seam, not the file. The benchmark constructs
    # UserOptimizedTransformer(config) itself, so the class must accept that signature --
    # which every candidate does, since they all subclass BaselineTransformer.
    ref.UserOptimizedTransformer = REGISTRY[candidate].build(ref.BaselineTransformer)

    argv = ["torch_transformer_benchmark.py", *cfg.cli_args(), *extra]
    old, sys.argv = sys.argv, argv
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            code = ref.main()
    except Exception as exc:                       # a crash is a result
        return {"config_id": config_id, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        sys.argv = old

    out = buf.getvalue()
    def grab(pattern, cast=float):
        m = re.search(pattern, out)
        return cast(m.group(1)) if m else None

    return {
        "config_id": config_id, "ok": code == 0, "exit_code": code,
        "speedup": grab(r"[Ss]peedup[^0-9]*([0-9.]+)"),
        "baseline_ms": grab(r"[Bb]aseline[^0-9]*median[^0-9]*([0-9.]+)"),
        "optimized_ms": grab(r"[Oo]ptimi[sz]ed[^0-9]*median[^0-9]*([0-9.]+)"),
        "accuracy_pass": "PASS" in out and "FAIL" not in out,
        "tail": "\n".join(out.strip().splitlines()[-6:]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    args = ap.parse_args()
    for cid in args.ids:
        r = run_one(cid, args.candidate, args.extra)
        if r["ok"]:
            print(f"cfg{cid:<3} benchmark reports: speedup={r['speedup']} "
                  f"baseline={r['baseline_ms']} optimized={r['optimized_ms']} "
                  f"accuracy_pass={r['accuracy_pass']}")
        else:
            print(f"cfg{cid:<3} FAILED: {r.get('error') or r.get('tail')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
