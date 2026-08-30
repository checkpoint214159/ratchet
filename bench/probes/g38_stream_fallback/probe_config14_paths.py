"""Config 14 under v38: does the per-sequence path still run RESIDENT and answer
identically to v37, and does the full batch still refuse without attempting?

Config 14 is where v33's streaming layer earns its keep, and requirement (2) of the g38
task is that the capability result must survive the fix. That result is produced by
`run_matrix.capability_one`, which calls the model ONE SEQUENCE AT A TIME (B=1) -- every
config-14 ledger row records `stream_path: resident, slice=1` there -- and then makes one
full-batch attempt at the end.

This probe checks both shapes at their true sizes:

  A. B=1, S=100000, d=1024, H=16, L=2. The shape the causal-prefix oracle and the blocked
     fp64 certificate are computed on. v38 must take the resident path, as v37 does, and
     must return the SAME TENSOR -- if the outputs are bitwise equal then every oracle
     downstream of them returns exactly what it returned for v33/v37, and no oracle needs
     to be re-run to know that.

  B. B=32, S=100000, d=1024. The single call a grading harness makes. Only the DECISION
     is exercised (12.21 GiB of input, no forward): the forward cannot succeed on this
     card for reasons no implementation removes -- 24.42 GiB of input and output against
     15.99 GiB -- which is exactly what the signature-floor pre-check states.

[L41]: a probe may propose; it may never conclude. Nothing here is a timing measurement.
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from bench.candidates import REGISTRY  # noqa: E402
from bench.candidates.v14_dispatch import signature_floor_bytes  # noqa: E402
from bench.matrix import BY_ID  # noqa: E402


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "ref_bench", REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench"] = m
    spec.loader.exec_module(m)
    return m


SEED = 1234
ref = load_reference()
c = BY_ID[14]
dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

tcfg = ref.TransformerConfig(batch_size=c.batch_size, seq_len=c.seq_len,
                             d_model=c.d_model, num_heads=c.heads, ffn_dim=c.ffn_dim,
                             num_layers=c.layers, causal=c.causal)
per_seq = ref.TransformerConfig(batch_size=1, seq_len=c.seq_len, d_model=c.d_model,
                                num_heads=c.heads, ffn_dim=c.ffn_dim,
                                num_layers=c.layers, causal=c.causal)

out = {"config": 14, "shape_full": [c.batch_size, c.seq_len, c.d_model]}
_free, total = torch.cuda.mem_get_info(dev)
out["device_total_GiB"] = total / 2**30
out["floor_full_GiB"] = signature_floor_bytes(
    c.batch_size, c.seq_len, c.d_model, 4) / 2**30
out["floor_per_sequence_GiB"] = signature_floor_bytes(1, c.seq_len, c.d_model, 4) / 2**30

# ---------------------------------------------------------------- A: the oracle shape
ys = {}
for name in ("v37_recombined2", "v38_stream_fallback"):
    torch._dynamo.reset()
    torch.manual_seed(SEED)
    base = ref.BaselineTransformer(tcfg)
    cand = REGISTRY[name].build(ref.BaselineTransformer)(tcfg)
    ref.copy_model_weights(base, cand)
    del base
    cand = cand.to(device=dev, dtype=torch.float32).eval()
    x, m = ref.generate_random_case(per_seq, dev, torch.float32, SEED, 0.0, 1.0)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        y = cand(x, m)
        torch.cuda.synchronize()
    ys[name] = y.detach().clone()
    out[name] = {
        "per_sequence_path": getattr(cand, "stream_path", None),
        "per_sequence_basis": getattr(cand, "stream_basis", None),
        "per_sequence_fallbacks": getattr(cand, "stream_fallbacks", None),
        "per_sequence_reason": getattr(cand, "stream_reason", None),
        "peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "attn_reason": getattr(cand, "attn_reason", None),
    }
    del cand, x, m, y
    torch.cuda.empty_cache()

d = (ys["v37_recombined2"] - ys["v38_stream_fallback"]).abs().max().item()
out["per_sequence_max_abs_v37_vs_v38"] = d
out["per_sequence_bitwise_identical"] = (d == 0.0)
del ys
torch.cuda.empty_cache()

# --------------------------------------------------- B: the full-batch DECISION only
torch.manual_seed(SEED)
base = ref.BaselineTransformer(tcfg)
cand = REGISTRY["v38_stream_fallback"].build(ref.BaselineTransformer)(tcfg)
ref.copy_model_weights(base, cand)
del base
cand = cand.to(device=dev, dtype=torch.float32).eval()
try:
    xb = torch.empty((c.batch_size, c.seq_len, c.d_model), device=dev,
                     dtype=torch.float32)
    cand._decide_stream(xb)
    out["full_batch_decision"] = {
        "path": cand.stream_path, "basis": cand.stream_basis,
        "attempted_resident": cand.stream_attempted_resident,
        "slice": cand.stream_slice, "reason": cand.stream_reason,
    }
    del xb
except Exception as exc:
    out["full_batch_decision"] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}

print("<<<JSON>>>" + json.dumps(out, indent=1))
