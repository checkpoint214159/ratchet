"""Per-config kernel decomposition: prove the win is a kernel improvement, not a dtype trick.

For each announced config, isolate the attention call and measure my flash kernel vs the
baseline's explicit attention at MATCHED precision (fp32) -- so the ratio is purely the
kernel/algorithm (streaming online-softmax + exact causal-skip), with no dtype advantage --
then again at fp16 to show the additional tensor-core factor. Grounds the end-to-end
numbers to the problem statement's "implement a GPU kernel for the layer".

Measurement notes, because the first version of this script reported a number that later
runs could not reproduce (cfg13 read 7.17x once, then 6.30-6.54x on four re-runs):

  * `do_bench(warmup, rep)` takes MILLISECONDS, not iterations. Triton divides them by the
    estimated per-call time, so the defaults (25/100) gave the cfg13 fp32 baseline -- ~59ms
    per call -- exactly ONE unwarmed sample. Sample counts are now derived from a measured
    estimate and floored, so every config gets a real distribution.
  * `return_mode` defaults to "mean". A mean over a handful of samples chases stragglers;
    the competition harness reports a median, so we do too.
  * A ratio is nearly immune to clock ramp (measured: 0.3% between a 208MHz-idle GPU and a
    boosted one) because both sides scale together. Cross-RUN spread was the real problem,
    so the ratio is repeated `--repeats` times and reported as median plus observed range.
"""
import argparse
import statistics
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpu_guard import require_exclusive

from ratchet.kernels.dispatch import MATRIX
from ratchet.kernels.flash_attention import flash_attention

MIN_WARMUP_ITERS = 10
MIN_BENCH_ITERS = 30


def baseline_attn(q, k, v, causal, D):
    s = (q @ k.transpose(-2, -1)) * (D ** -0.5)
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1),
                          float("-inf"))
    return torch.softmax(s.float(), -1).to(q.dtype) @ v


def _estimate_ms(fn):
    """One timed call after a short warm-up, to size the real run."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return max(start.elapsed_time(end), 1e-3)


def bench(fn):
    """Median runtime with sample counts derived from the function's own cost."""
    try:
        est = _estimate_ms(fn)
        return triton.testing.do_bench(
            fn,
            warmup=max(25, int(est * MIN_WARMUP_ITERS)),
            rep=max(100, int(est * MIN_BENCH_ITERS)),
            return_mode="median",
        )
    except Exception:
        return None


def ratios(cfg):
    """One trial -> (pure-kernel fp32, fp16, flash_fp16 vs baseline_fp32)."""
    B, H, N, D = cfg.batch_size, cfg.heads, cfg.seq_len, cfg.head_dim
    q32 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
    q16 = q32.to(torch.float16)
    tb32 = bench(lambda: baseline_attn(q32, q32, q32, True, D))
    tf32 = bench(lambda: flash_attention(q32, q32, q32, causal=True))
    tb16 = bench(lambda: baseline_attn(q16, q16, q16, True, D))
    tf16 = bench(lambda: flash_attention(q16, q16, q16, causal=True))
    nan = float("nan")
    return (tb32 / tf32 if tb32 and tf32 else nan,
            tb16 / tf16 if tb16 and tf16 else nan,
            tb32 / tf16 if tb32 and tf16 else nan)


def fmt(trials):
    """Median with the observed spread, so the reader sees the uncertainty."""
    vals = [v for v in trials if v == v]  # drop NaN
    if not vals:
        return "    n/a       "
    med = statistics.median(vals)
    if len(vals) == 1:
        return f"{med:6.2f}x        "
    lo, hi = min(vals), max(vals)
    return f"{med:6.2f}x +-{100 * (hi - lo) / 2 / med:4.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", type=int, nargs="*", help="config ids (default: all runnable)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent trials per config; reported as median +- half-range")
    ap.add_argument("--allow-contention", action="store_true")
    args = ap.parse_args()

    require_exclusive(args.allow_contention)
    only = set(args.ids)

    print(f"repeats={args.repeats}, median of do_bench(median), "
          f">={MIN_WARMUP_ITERS} warmup / >={MIN_BENCH_ITERS} timed iterations per trial")
    print("cfg | shape[B,H,N,D]      | flash vs base fp32 | fp16               | "
          "flash_fp16 vs base_fp32")
    for cfg in MATRIX:
        if cfg.id == 14 or (only and cfg.id not in only):
            continue
        B, H, N, D = cfg.batch_size, cfg.heads, cfg.seq_len, cfg.head_dim
        try:
            trials = [ratios(cfg) for _ in range(args.repeats)]
            pk, f16, cross = zip(*trials)
            shape = f"[{B},{H},{N},{D}]"
            print(f"cfg{cfg.id:<2} {shape:<20} | {fmt(pk)} | {fmt(f16)} | {fmt(cross)}")
        except Exception as e:
            print(f"cfg{cfg.id}: FAILED {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
