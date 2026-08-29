"""Config 14 (B=32, S=100000, d=1024, H=16, L=2): can anything run it, and is it correct?

The reference implementation cannot. It OOMs inside the benchmark's OWN input generator
before forward() is ever called, and its materialized attention would need 18.63 TB for a
single layer against 16 GB of VRAM.

That makes config 14 a CAPABILITY question rather than a speed one, and it needs a
different method from the rest of the matrix, because the usual method is unavailable:
there is no baseline output to compare against, so correctness cannot be checked at the
real shape by any means.

So this probe splits the claim in two, and reports them separately:

  1. CORRECTNESS AT PROXY SHAPES. Same d_model, heads and layers as config 14, at
     sequence lengths where the reference still fits. If the candidate is correct at
     S=1024 and S=4096 with this width and depth, the arithmetic is right; sequence
     length changes how much work is done, not what is computed.

  2. FEASIBILITY AT THE REAL SHAPE. Whether a forward completes at S=100000, how long it
     takes, and what it peaks at. Reported as "it runs", never as "it is correct" --
     that second claim is not available at this shape and is not made.

The input is streamed from pinned host memory one sequence at a time. The full fp32 input
is 12.21 GiB and the benchmark's generator allocates it twice; keeping it host-side is
what makes the shape reachable at all, and it is an honest description of the constraint:
the harness's own input construction, not the model, is the binding limit on a 16 GB card.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RTOL, ATOL, SEED = 0.02, 0.002, 1234


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "ref_bench", REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench"] = m
    spec.loader.exec_module(m)
    return m


def proxy_correctness(ref, cand_name, seq_lens):
    """Config 14's width and depth at survivable sequence lengths."""
    import torch
    from bench.candidates import REGISTRY
    out = []
    for S in seq_lens:
        tcfg = ref.TransformerConfig(batch_size=1, seq_len=S, d_model=1024, num_heads=16,
                                     ffn_dim=1024, num_layers=2, causal=True)
        torch.manual_seed(SEED)
        base = ref.BaselineTransformer(tcfg)
        cand = REGISTRY[cand_name].build(ref.BaselineTransformer)(tcfg)
        ref.copy_model_weights(base, cand)
        base = base.to("cuda", torch.float32).eval()
        cand = cand.to("cuda", torch.float32).eval()
        worst_abs = worst_rel = 0.0
        failed = 0
        with torch.inference_mode():
            for t in range(3):
                x, m = ref.generate_random_case(tcfg, torch.device("cuda"),
                                                torch.float32, SEED + t, 0.0, 1.0)
                r = ref.compare_outputs(base(x, m), cand(x, m), rtol=RTOL, atol=ATOL)
                worst_abs = max(worst_abs, float(r.max_abs_error))
                worst_rel = max(worst_rel, float(r.max_relative_error))
                failed += int(r.failed_elements)
                del x, m
        out.append({"seq_len": S, "passed": failed == 0, "max_abs": worst_abs,
                    "max_rel": worst_rel, "failed_elements": failed})
        print(f"  proxy S={S:<6} passed={failed == 0}  max_abs={worst_abs:.3e}  failed={failed}")
        del base, cand
        torch.cuda.empty_cache()
    return out


def real_shape(ref, cand_name, batch=32, seq=100000):
    """Does a forward complete at the announced shape, and at what cost?"""
    import torch
    from bench.candidates import REGISTRY
    tcfg = ref.TransformerConfig(batch_size=1, seq_len=seq, d_model=1024, num_heads=16,
                                 ffn_dim=1024, num_layers=2, causal=True)
    torch.manual_seed(SEED)
    base = ref.BaselineTransformer(tcfg)
    cand = REGISTRY[cand_name].build(ref.BaselineTransformer)(tcfg)
    ref.copy_model_weights(base, cand)
    cand = cand.to("cuda", torch.float32).eval()
    cand.use_graph = False          # one 0.38 GiB static buffer per sequence is not worth it
    del base

    # Host-resident input, streamed one sequence at a time.
    host = torch.randn(1, seq, 1024, dtype=torch.float32, pin_memory=True)
    mask = torch.ones(1, seq, dtype=torch.bool, device="cuda")
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        x = host.to("cuda", non_blocking=True)
        cand(x, mask)                                  # warm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            x = host.to("cuda", non_blocking=True)
            y = cand(x, mask)
            del y
        torch.cuda.synchronize()
        per_seq = (time.perf_counter() - t0) / 3

    peak = torch.cuda.max_memory_allocated() / 1e9
    return {"ran": True, "per_sequence_s": per_seq, "batch": batch,
            "extrapolated_full_batch_s": per_seq * batch, "peak_GB": peak,
            "graph": False, "note": "input streamed from pinned host memory; "
                                    "correctness NOT verifiable at this shape"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="v6_fp16_gelu")
    ap.add_argument("--proxy-seqs", type=int, nargs="*", default=[1024, 4096])
    args = ap.parse_args()
    sys.path.insert(0, str(REPO))
    ref = load_reference()

    print(f"config 14 probe :: candidate={args.candidate}")
    print("[1] correctness at proxy shapes (d=1024, H=16, L=2, causal)")
    proxies = proxy_correctness(ref, args.candidate, args.proxy_seqs)

    print("[2] feasibility at the real shape (S=100000)")
    try:
        real = real_shape(ref, args.candidate)
        print(f"  RAN: {real['per_sequence_s']:.2f}s per sequence, "
              f"{real['extrapolated_full_batch_s']:.1f}s extrapolated for B=32, "
              f"peak {real['peak_GB']:.2f} GB")
    except Exception as exc:
        real = {"ran": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"  DID NOT RUN: {real['error']}")

    print("__RESULT__" + json.dumps({"proxies": proxies, "real": real}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
