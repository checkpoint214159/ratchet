"""Which attention form does v40 select on each config, and where does it differ from v38?

Two jobs, both prerequisites for a fair ABBA run:

1. **[L36] Assert the mechanism engaged.** A candidate that selects `single_tile`
   everywhere is v38 with extra build time, and would still pass every correctness test.

2. **Find a byte-identical in-run control.** Finding 49's addendum shows `bench/abba.py`
   resolves two arms to the hundredth of a microsecond when they run identical code, and
   that control is what makes the rest of the run believable. v40 is NOT automatically
   identical to v38 where the looped form declines: `attn_choice` also sweeps
   `sdpa+repack`, which v23's `autotune_tile` never did, so a shape can change hands
   without the looped kernel being involved at all. The control has to be a config where
   both models are MEASURED to choose the same thing, not one where they are assumed to.

Also reports the wall-clock cost of the sweep at construction, because it is real: 54
legal looped tiles is 54 Triton compiles at prime time, and a build-time cost that large
should be visible rather than discovered.

INDICATIVE ONLY [L41]. Take the GPU lock.
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch

from bench.matrix import MATRIX

ARMS = ("v38_stream_fallback", "v40_looped_attn")
# Config 14 needs the streaming protocol and minutes per forward; it is excluded from
# every ABBA run in this project for the same reason (finding 40).
SKIP = {14}


def _reference(tag: str):
    spec = importlib.util.spec_from_file_location(
        tag, REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def one(config_id: int) -> dict:
    from bench.candidates import REGISTRY
    from bench.matrix import BY_ID

    cfg = BY_ID[config_id]
    ref = _reference(f"ref_form_{config_id}")
    tcfg = ref.TransformerConfig(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, d_model=cfg.d_model,
        num_heads=cfg.heads, ffn_dim=cfg.ffn_dim, num_layers=cfg.layers,
        causal=cfg.causal)
    tcfg.validate()
    dev = torch.device("cuda")
    torch.manual_seed(1234)
    base = ref.BaselineTransformer(tcfg)
    x, m = ref.generate_random_case(tcfg, dev, torch.float32, seed=1234,
                                    padding_ratio=0.0, input_scale=1.0)
    base_d = base.to(device=dev, dtype=torch.float32).eval()
    with torch.inference_mode():
        want = base_d(x, m)

    out = {"config_id": config_id}
    for arm in ARMS:
        mdl = REGISTRY[arm].build(ref.BaselineTransformer)(tcfg)
        ref.copy_model_weights(base_d, mdl)
        mdl = mdl.to(device=dev, dtype=torch.float32).eval()
        t0 = time.perf_counter()
        with torch.inference_mode():
            got = mdl(x, m)                  # the forward that runs `_decide_attn`
        torch.cuda.synchronize()
        build_s = time.perf_counter() - t0
        res = ref.compare_outputs(want, got, rtol=0.02, atol=0.002)
        out[arm] = {
            "form": getattr(mdl, "attn_form", "single_tile" if getattr(
                mdl, "attn_used", False) else "sdpa"),
            "used": getattr(mdl, "attn_used", None),
            "tile": getattr(mdl, "attn_tile", None),
            "reason": getattr(mdl, "attn_reason", None),
            "passed": bool(res.passed),
            "max_abs": float(res.max_abs_error),
            "first_forward_s": build_s,
        }
        del mdl
        torch.cuda.empty_cache()
    return out


def main() -> int:
    from bench.gpu_lock import gpu_lock

    with gpu_lock("g40 which-form", timeout_s=14400):
        rows = []
        for c in MATRIX:
            if c.id in SKIP:
                continue
            try:
                rows.append(one(c.id))
            except Exception as exc:
                print(f"config {c.id}: FAILED {type(exc).__name__}: {exc}")

        print(f"\n{'cfg':>4} {'B':>6} {'H':>3} {'hd':>4} {'S':>5}  "
              f"{'v38 form':<12}{'v38 tile':<14}{'v40 form':<12}{'v40 tile':<18}"
              f"{'same?':<7}{'1st fwd s':>10}{'ok':>4}")
        from bench.matrix import BY_ID
        identical, differ = [], []
        for r in rows:
            c = BY_ID[r["config_id"]]
            a, b = r[ARMS[0]], r[ARMS[1]]
            same = (a["form"] == b["form"] and a["tile"] == b["tile"])
            (identical if same else differ).append(r["config_id"])
            print(f"{c.id:>4} {c.batch_size:>6} {c.heads:>3} "
                  f"{c.d_model//c.heads:>4} {c.seq_len:>5}  "
                  f"{a['form']:<12}{str(a['tile']):<14}{b['form']:<12}"
                  f"{str(b['tile']):<18}{'YES' if same else 'no':<7}"
                  f"{b['first_forward_s']:>10.1f}"
                  f"{'  ok' if a['passed'] and b['passed'] else ' FAIL':>4}")

        print(f"\nBYTE-IDENTICAL configs (candidates for the in-run ABBA control): "
              f"{identical}")
        print(f"CONFIGS WHERE v40 DIFFERS: {differ}")
        for r in rows:
            if r["config_id"] in differ:
                print(f"\n  config {r['config_id']} v40: "
                      f"{r[ARMS[1]]['reason']}")
                print(f"  config {r['config_id']} v38: "
                      f"{r[ARMS[0]]['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
