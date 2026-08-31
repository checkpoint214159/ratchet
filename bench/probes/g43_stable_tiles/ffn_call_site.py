"""Which FFN kernel, at which tile, does the FRONTIER actually run on each config?

Job 2's arithmetic said `_ffn_block` (v16's `BLOCK_M = 64, NUM_WARPS = 8`) fires where
`amortizes` holds and `_ffn_block_normed` (v34's derived `launch_bm` and swept
`launch_warps`) fires where `one_wave` holds. [L36] says to check that rather than
believe it: a sweep of constants that the model does not run is a sweep of nothing.

Prints, per config, the two decisions and the four numbers they resolve to.

    python3 bench/probes/g43_stable_tiles/ffn_call_site.py --ids 1 2 3 4 6 7 12 13
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> int:
    import torch
    from bench.candidates import REGISTRY
    from bench.matrix import BY_ID

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--arm", default="v43_replicated_tile")
    a = ap.parse_args()

    p = REPO / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_callsite", p)
    ref = importlib.util.module_from_spec(spec)
    sys.modules["ref_callsite"] = ref
    spec.loader.exec_module(ref)
    dev = torch.device("cuda")

    print(f"{'cfg':>4} {'tokens':>9} {'ffn kernel':<20} {'bm':>4} {'warps':>6}  why")
    for cid in a.ids:
        c = BY_ID[cid]
        cfg = ref.TransformerConfig(
            batch_size=c.batch_size, seq_len=c.seq_len, d_model=c.d_model,
            num_heads=c.heads, ffn_dim=c.ffn_dim, num_layers=c.layers, causal=c.causal)
        cfg.validate()
        torch.manual_seed(1234)
        base = ref.BaselineTransformer(cfg)
        m = REGISTRY[a.arm].build(ref.BaselineTransformer)(cfg)
        ref.copy_model_weights(base, m)
        m = m.to(device=dev, dtype=torch.float32).eval()
        del base
        x, mask = ref.generate_random_case(cfg, dev, torch.float32, seed=1234,
                                           padding_ratio=0.0, input_scale=1.0)
        try:
            with torch.inference_mode():
                m(x, mask)
        except Exception as exc:
            print(f"{cid:>4} {c.tokens:>9} {'FAILED':<20} {type(exc).__name__}: {exc}")
            del m, x, mask
            torch.cuda.empty_cache()
            continue

        if getattr(m, "launch_fused_used", False):
            kern, bm, w = ("_ffn_block_normed", m.launch_bm, m.launch_warps)
            why = getattr(m, "launch_reason", "")
        elif getattr(m, "fused_ffn_used", False) and getattr(m, "_nomask", False):
            kern, bm, w = ("_ffn_block", m.BLOCK_M, m.NUM_WARPS)
            why = getattr(m, "fused_ffn_reason", "")
        else:
            kern, bm, w = ("none (unfused path)", 0, 0)
            why = (f"launch: {getattr(m, 'launch_reason', '')} | "
                   f"ffn: {getattr(m, 'fused_ffn_reason', '')}")
        print(f"{cid:>4} {c.tokens:>9} {kern:<20} {bm:>4} {w:>6}  {why[:100]}")
        del m, x, mask
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
