"""E2c: run the AUTHORITATIVE evaluator with the flash kernel injected at the seam.

The benchmark file is imported unmodified (SHA preserved). We monkeypatch only
UserOptimizedTransformer.forward -- the designated seam -- then hand control to the
evaluator's own main(), so accuracy rule and timing are the authoritative ones.
"""
import importlib.util
import os
import sys
from pathlib import Path

from ratchet.kernels.transformer_layer import (
    optimized_forward,
    optimized_forward_full,
    optimized_forward_qkv,
    optimized_forward_tf32,
)

BENCH = Path("benchmarks/reference/torch_transformer_benchmark.py").resolve()
spec = importlib.util.spec_from_file_location("tt_bench", BENCH)
mod = importlib.util.module_from_spec(spec)
sys.modules["tt_bench"] = mod
spec.loader.exec_module(mod)

seam = os.environ.get("RATCHET_SEAM", "flash")
mod.UserOptimizedTransformer.forward = {
    "flash": optimized_forward, "tf32": optimized_forward_tf32,
    "qkv": optimized_forward_qkv, "full": optimized_forward_full,
}[seam]

# default config; pass through any extra CLI args (e.g. --causal, --dtype)
sys.argv = ["torch_transformer_benchmark.py", "--device", "cuda"] + sys.argv[1:]
raise SystemExit(mod.main())
