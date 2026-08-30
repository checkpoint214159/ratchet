"""Diagnosis probe: WHICH path does the streaming lineage latch on config 6, and when?

Replicates bench/run_matrix.py measure_one's ORDERING exactly (baseline timed first,
then both models resident for correctness, then baselines freed and the candidate
timed) and reports free memory + the latched decision at each stage.

[L41] A probe may propose; it may never conclude. This file is not the ledger and its
timing numbers are the isolated protocol's, which finding 45 shows misreports candidates
that plan at construction. Read the MEMORY numbers and the PATH; rank nothing by it.
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
from bench.matrix import BY_ID  # noqa: E402
from bench.candidates import REGISTRY  # noqa: E402


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "ref_bench", REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench"] = m
    spec.loader.exec_module(m)
    return m


SEED = 1234
cid = int(sys.argv[1]) if len(sys.argv) > 1 else 6
name = sys.argv[2] if len(sys.argv) > 2 else "v37_recombined2"

ref = load_reference()
cfg = BY_ID[cid]
dev = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
tcfg = ref.TransformerConfig(batch_size=cfg.batch_size, seq_len=cfg.seq_len,
                             d_model=cfg.d_model, num_heads=cfg.heads,
                             ffn_dim=cfg.ffn_dim, num_layers=cfg.layers,
                             causal=cfg.causal)
tcfg.validate()


def free_bytes(tag):
    f, _t = torch.cuda.mem_get_info(dev)
    print(f"  [{tag:<36}] free {f / 2**30:7.2f} GiB "
          f" reserved {torch.cuda.memory_reserved() / 2**30:6.2f}"
          f"  allocated {torch.cuda.memory_allocated() / 2**30:6.2f}")
    return f


def make_input(seed):
    return ref.generate_random_case(tcfg, dev, torch.float32, seed=seed,
                                    padding_ratio=0.0, input_scale=1.0)


def median_ms(model, x, mask, n):
    with torch.inference_mode():
        for _ in range(20):
            model(x, mask)
        torch.cuda.synchronize()
        s = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        e = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            s[i].record()
            model(x, mask)
            e[i].record()
        torch.cuda.synchronize()
    v = sorted(a.elapsed_time(b) for a, b in zip(s, e))
    return v[len(v) // 2]


print(f"config {cid}  {name}  act={cfg.activation_bytes() / 2**30:.3f} GiB")
free_bytes("process start")
torch.manual_seed(SEED)
baseline = ref.BaselineTransformer(tcfg).to(device=dev, dtype=torch.float32).eval()
xt, mt = make_input(SEED + 100000)
free_bytes("baseline + timing input built")
probe = median_ms(baseline, xt, mt, 3)
n = max(11, min(300, int(2000.0 / max(probe, 0.05))))
base_ms = min(median_ms(baseline, xt, mt, n), median_ms(baseline, xt, mt, n))
free_bytes("after baseline arm timed")
print(f"  baseline {base_ms:.3f} ms, n={n}")

torch.manual_seed(SEED)
fresh_baseline = ref.BaselineTransformer(tcfg)
cand_cls = REGISTRY[name].build(ref.BaselineTransformer)
candidate = cand_cls(tcfg)
ref.copy_model_weights(fresh_baseline, candidate)
fresh_baseline = fresh_baseline.to(device=dev, dtype=torch.float32).eval()
candidate = candidate.to(device=dev, dtype=torch.float32).eval()
free_bytes("both models resident, pre-forward")

with torch.inference_mode():
    x, mask = make_input(SEED)
    free_bytes("+ correctness input")
    expected = fresh_baseline(x, mask)
    free_at_decision = free_bytes("+ reference output (the real moment)")
    got = candidate(x, mask)
print("  DECISION AT FIRST FORWARD:", getattr(candidate, "stream_path", None))
print("  reason:", getattr(candidate, "stream_reason", None))
del x, mask, expected, got

del baseline, fresh_baseline
torch.cuda.empty_cache()
free_at_timing = free_bytes("baselines freed, timing phase")
print("  DECISION STILL LATCHED AS:", getattr(candidate, "stream_path", None))
cand_ms = min(median_ms(candidate, xt, mt, n), median_ms(candidate, xt, mt, n))
print(f"  candidate {cand_ms:.3f} ms  (isolated-protocol ratio {base_ms / cand_ms:.3f})")

from bench.candidates.v14_dispatch import (  # noqa: E402
    RESIDENT_BUDGET, choose, estimate_working_set_bytes)

need = estimate_working_set_bytes(cfg.batch_size, cfg.seq_len, cfg.d_model, cfg.heads,
                                  cfg.layers, 4)
print(json.dumps({
    "config": cid, "candidate": name,
    "act_GiB": cfg.activation_bytes() / 2**30,
    "estimate_GiB": need / 2**30,
    "free_at_decision_GiB": free_at_decision / 2**30,
    "free_at_timing_GiB": free_at_timing / 2**30,
    "free_needed_for_resident_GiB": need / RESIDENT_BUDGET / 2**30,
    "choose_at_decision": choose(cfg.batch_size, cfg.seq_len, cfg.d_model, cfg.heads,
                                 cfg.layers, 4, free_at_decision),
    "choose_at_timing": choose(cfg.batch_size, cfg.seq_len, cfg.d_model, cfg.heads,
                               cfg.layers, 4, free_at_timing),
    "latched": getattr(candidate, "stream_path", None),
    "measured_peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
    "baseline_ms": base_ms, "candidate_ms": cand_ms,
}, indent=1))
