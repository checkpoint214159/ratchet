"""Does the fallback survive a REAL allocator refusal, not an injected exception?

`tests/bench/test_v38_stream_fallback.py` raises `torch.cuda.OutOfMemoryError` into the
resident path by hand. That exercises v38's try/except and its recovery, but it does not
exercise the state the ALLOCATOR is actually in when it refuses -- fragmented, with a
half-finished forward's intermediates still cached -- and that state is the reason a
"just catch OOM" design can fail in practice.

`torch.cuda.set_per_process_memory_fraction` makes the caching allocator refuse for real,
through its own code path, at a budget we choose. The budget here is set BETWEEN the
resident forward's measured peak and what a slice needs, so that residency must fail and
streaming must then succeed under the same cap.

[L38]: a check nobody has watched fail is not a check. [L41]: this proposes, it does not
conclude -- there is no timing here.
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from bench.candidates import REGISTRY  # noqa: E402


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "ref_bench", REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench"] = m
    spec.loader.exec_module(m)
    return m


B, S, D, H, L = 256, 512, 128, 4, 2
FRACTION_OF_HEADROOM = float(sys.argv[1]) if len(sys.argv) > 1 else 0.40

ref = load_reference()
dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")
cfg = ref.TransformerConfig(batch_size=B, seq_len=S, d_model=D, num_heads=H,
                            ffn_dim=D, num_layers=L, causal=True)
out = {"shape": [B, S, D], "fraction_of_headroom": FRACTION_OF_HEADROOM}


def build(name):
    torch.manual_seed(1234)
    base = ref.BaselineTransformer(cfg)
    cand = REGISTRY[name].build(ref.BaselineTransformer)(cfg)
    ref.copy_model_weights(base, cand)
    return (base.to(dev, torch.float32).eval(), cand.to(dev, torch.float32).eval())

# ---- what a RESIDENT forward costs, measured -----------------------------------------
base, warm = build("v38_stream_fallback")
x = torch.randn(B, S, D, device=dev)
m = torch.ones(B, S, dtype=torch.bool, device=dev)
with torch.inference_mode():
    expected = base(x, m)
del base
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    warm(x, m)
    torch.cuda.synchronize()
resident_peak = torch.cuda.max_memory_allocated()
out["resident_path_of_the_warm_model"] = warm.stream_path
del warm
torch.cuda.empty_cache()
resting = torch.cuda.memory_allocated()
_free, total = torch.cuda.mem_get_info(dev)
out.update(resident_peak_MB=resident_peak / 1e6, resting_MB=resting / 1e6,
           total_MB=total / 1e6)

# ---- the candidate, built BEFORE the cap so construction is not what fails ------------
_b2, cand = build("v38_stream_fallback")
torch.cuda.empty_cache()
cap = resting + (resident_peak - resting) * FRACTION_OF_HEADROOM
out["cap_MB"] = cap / 1e6
torch.cuda.set_per_process_memory_fraction(cap / total, 0)
err = None
try:
    with torch.inference_mode():
        y = cand(x, m)
        torch.cuda.synchronize()
except Exception as exc:
    err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    y = None
finally:
    torch.cuda.set_per_process_memory_fraction(1.0, 0)

out.update(
    error=err,
    stream_path=getattr(cand, "stream_path", None),
    stream_basis=getattr(cand, "stream_basis", None),
    stream_fallbacks=getattr(cand, "stream_fallbacks", None),
    stream_slice=getattr(cand, "stream_slice", None),
    stream_reason=getattr(cand, "stream_reason", None),
    settled={a: getattr(cand, a, None) for a in
             ("attn_reason", "fused_ffn_reason", "launch_reason", "gemm_reason")},
)
if y is not None:
    out["shape_ok"] = tuple(y.shape) == tuple(x.shape)
    out["max_abs_vs_reference"] = (y - expected).abs().max().item()
    out["inside_locked_atol_2e-3"] = out["max_abs_vs_reference"] < 2e-3

print("<<<JSON>>>" + json.dumps(out, indent=1))
