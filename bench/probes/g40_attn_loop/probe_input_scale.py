"""Is the input_scale 0.1 accuracy failure NEW, or does the parent do it too?

`tests/bench/test_v40_looped_attn.py::test_input_scale_tail_is_unchanged` fails on config
10 at `input_scale=0.1`: 298 of 1048576 elements past the locked tolerance, max_abs
3.851e-03, worst element -0.1307 (reference) against -0.1269 (candidate) -- 2.95% relative,
which is outside the 2e-2 rtol as well as the 2e-3 atol.

**That number means nothing on its own.** CLAUDE.md's standing hazard says the strict
both-bounds tolerance is nearly saturated by bf16 itself (a floor of 1.95e-03 against a
locked 2e-03) and that honest kernels are expected to trip it on large shapes. Finding 19
is the same tail from the other side. So the only question that matters is the DIFFERENCE:
does `v38_stream_fallback` -- the shipping candidate, on the identical input, with the
identical seed -- fail in the same place?

This probe answers exactly that and nothing else. It does not interpret the tolerance, and
it does not touch it. If v38 fails too, the test's premise is wrong and the test should
compare candidates rather than assert an absolute. If v38 passes and v40 fails, the looped
kernel is less accurate than the form it replaces and the candidate must be withdrawn --
correctness before timing, no exceptions.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch

from bench.matrix import BY_ID

ARMS = ("v38_stream_fallback", "v40_looped_attn")
SCALES = (0.1, 0.5, 1.0, 2.0, 10.0)
SEEDS = (1234, 4321, 7)


def _reference(tag: str):
    spec = importlib.util.spec_from_file_location(
        tag, REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    from bench.candidates import REGISTRY
    from bench.gpu_lock import gpu_lock

    config_id = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    c = BY_ID[config_id]
    ref = _reference("ref_scale")
    cfg = ref.TransformerConfig(
        batch_size=c.batch_size, seq_len=c.seq_len, d_model=c.d_model,
        num_heads=c.heads, ffn_dim=c.ffn_dim, num_layers=c.layers, causal=c.causal)
    cfg.validate()
    dev = torch.device("cuda")

    with gpu_lock("g40 input-scale tail", timeout_s=7200):
        torch.manual_seed(4321)
        base = ref.BaselineTransformer(cfg)
        models = {}
        for arm in ARMS:
            m = REGISTRY[arm].build(ref.BaselineTransformer)(cfg)
            ref.copy_model_weights(base, m)
            models[arm] = m.to(device=dev, dtype=torch.float32).eval()
        base = base.to(device=dev, dtype=torch.float32).eval()

        print(f"config {config_id}   B={c.batch_size} H={c.heads} "
              f"hd={c.d_model//c.heads} S={c.seq_len}\n")
        print(f"{'seed':>6}{'scale':>7}   "
              f"{'v38 max_abs':>13}{'failed':>9}{'ok':>4}   "
              f"{'v40 max_abs':>13}{'failed':>9}{'ok':>4}   verdict")
        verdicts = []
        for seed in SEEDS:
            for sc in SCALES:
                x, mask = ref.generate_random_case(cfg, dev, torch.float32, seed=seed,
                                                   padding_ratio=0.0, input_scale=sc)
                with torch.inference_mode():
                    want = base(x, mask)
                    res = {a: ref.compare_outputs(want, models[a](x, mask),
                                                  rtol=0.02, atol=0.002)
                           for a in ARMS}
                a, b = res[ARMS[0]], res[ARMS[1]]
                if a.passed and b.passed:
                    v = "both pass"
                elif not a.passed and not b.passed:
                    v = "BOTH FAIL -- inherited, not new"
                elif a.passed and not b.passed:
                    v = "*** v40 ONLY -- A REGRESSION ***"
                else:
                    v = "v38 only fails; v40 better"
                verdicts.append(v)
                print(f"{seed:>6}{sc:>7}   "
                      f"{a.max_abs_error:>13.3e}{a.failed_elements:>9}"
                      f"{'  ok' if a.passed else 'FAIL':>4}   "
                      f"{b.max_abs_error:>13.3e}{b.failed_elements:>9}"
                      f"{'  ok' if b.passed else 'FAIL':>4}   {v}")
        print()
        for arm in ARMS:
            print(f"  {arm}: attn_form="
                  f"{getattr(models[arm], 'attn_form', 'single_tile')} "
                  f"tile={getattr(models[arm], 'attn_tile', None)}")
        regressions = sum("REGRESSION" in v for v in verdicts)
        print(f"\nrows where v40 fails and v38 does not: {regressions} of {len(verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
