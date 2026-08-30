"""Drift-robust A/B timing: baseline vs candidate seam, interleaved, min-of-N.

GB10's SM clock cannot be hard-locked without root, so absolute latencies drift with DVFS.
The ratio, however, is stable if both sides are measured back-to-back under the same clock
state. Each round times baseline and candidate adjacently (CUDA-event min-of-N with an L2
flush); we report the median of per-round ratios plus a do_bench cross-check.
"""
import importlib.util
import os
import statistics
import sys
from pathlib import Path

import torch

from ratchet.kernels.explore import (
    forward_cublas_tf32,
    forward_fused_ffn,
    forward_sdpa_fp32,
)
from ratchet.kernels.transformer_layer import (
    optimized_forward,
    optimized_forward_full,
    optimized_forward_qkv,
    optimized_forward_tf32,
)
from ratchet.oracle import calibrate
from ratchet.oracle.timing import get_timer

BENCH = Path("benchmarks/reference/torch_transformer_benchmark.py").resolve()
spec = importlib.util.spec_from_file_location("tt_bench", BENCH)
mod = importlib.util.module_from_spec(spec)
sys.modules["tt_bench"] = mod
spec.loader.exec_module(mod)

SEAM = os.environ.get("RATCHET_SEAM", "tf32")
forward = {"flash": optimized_forward, "tf32": optimized_forward_tf32,
           "qkv": optimized_forward_qkv, "full": optimized_forward_full,
           "cublastf32": forward_cublas_tf32, "sdpa": forward_sdpa_fp32,
           "fusedffn": forward_fused_ffn}[SEAM]

dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
    os.environ.get("RATCHET_DTYPE", "float32")]
B = int(os.environ.get("RATCHET_BS", "8"))
Nseq = int(os.environ.get("RATCHET_SEQ", "128"))
causal = os.environ.get("RATCHET_CAUSAL", "0") == "1"

cfg = mod.TransformerConfig(batch_size=B, seq_len=Nseq, d_model=512, num_heads=8,
                            ffn_dim=2048, num_layers=6, causal=causal)
dev = torch.device("cuda")
baseline = mod.BaselineTransformer(cfg).to(dev).to(dtype).eval()
optimized = mod.UserOptimizedTransformer(cfg).to(dev).to(dtype).eval()
mod.UserOptimizedTransformer.forward = forward
mod.copy_model_weights(baseline, optimized)

x = torch.randn(B, Nseq, 512, device=dev, dtype=dtype)
prof = calibrate(cache_path="ledger/device.gb10.json")
# L2 flush models a cold-cache serve; no-flush matches the authoritative evaluator's warm
# back-to-back timing (the scored methodology). RATCHET_FLUSH=0 selects warm.
flush = prof.l2_flush_bytes if os.environ.get("RATCHET_FLUSH", "1") == "1" else 0

def fb():
    with torch.no_grad():
        baseline(x, None)

def fo():
    with torch.no_grad():
        optimized(x, None)

# correctness sanity (OR-rule, elementwise) before trusting any timing
with torch.no_grad():
    rb, ro = baseline(x, None).float(), optimized(x, None).float()
abs_err = (rb - ro).abs()
ok = ((abs_err <= 0.002) | (abs_err <= 0.02 * rb.abs())).all().item()

timer = get_timer("do_bench")
ratios, tb, to = [], [], []
# warm up once so autotune / JIT compile is not counted in the first round
fb()
fo()
torch.cuda.synchronize()
for _ in range(int(os.environ.get("RATCHET_ROUNDS", "3"))):
    sb, _ = timer(fb, flush_bytes=flush)
    so, _ = timer(fo, flush_bytes=flush)
    tb.append(sb.min_ns)
    to.append(so.min_ns)
    ratios.append(sb.min_ns / so.min_ns)

print(f"seam={SEAM} dtype={dtype} B={B} seq={Nseq} causal={causal} correct={ok}")
print(f"baseline  min={min(tb)/1e6:.3f} ms")
print(f"optimized min={min(to)/1e6:.3f} ms")
print(f"speedup (median of per-round ratios) = {statistics.median(ratios):.3f}x  "
      f"[min {min(ratios):.3f}, max {max(ratios):.3f}]")
