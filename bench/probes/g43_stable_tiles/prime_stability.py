"""How many DIFFERENT plans does a candidate's tuner select, asked the same question?

WHAT IS BEING MEASURED, AND WHY IT IS NOT A TIMING
---------------------------------------------------
Generation 42 replaced an instrument that could not resolve its arms with one that can,
and finding 53 measured the price: on the shape whose two leading tiles are ~2% apart the
sweep went from one tile in 6 runs to THREE tiles in 6. Generation 43's claim is about
that number and only that number. A speed A/B cannot see it -- a plan that varies run to
run shows up as variance in the timing, which is exactly the thing every protocol in this
repo is built to suppress.

So this file measures the OUTPUT OF THE SELECTION RULE, not the time it selects. The
observable is `(attn_form, attn_tile)` after `_decide_attn`, and the statistic is how
many distinct values it takes over repeated asks.

TWO MODES, BECAUSE THE REGIME IS THE WHOLE PROBLEM (finding 53's own confession)
---------------------------------------------------------------------------------
Finding 53 diagnosed a tuner for using a timer whose regime did not match its call site,
and then validated the fix with a probe whose regime did not match the tuner's: on B=4
the model's own prime-time sweep reported one tile beating another 1.460x where the
standalone probe measured the same two arms at 0.98x. A 1.49x gap on the same arms under
the same timer. So this file does not have a regime; it has two, and says which is which.

  --mode fresh      ONE priming per PROCESS. This is the call site exactly: a model is
                    constructed, `_decide_attn` runs once before compilation and graph
                    capture, and the plan is fixed for the life of the process. It is
                    what `bench/abba.py` records and what the graded harness would see.
                    Authoritative, and expensive -- one CUDA context per priming.

  --mode resident   N primings per process, arms INTERLEAVED, on models that are already
                    built, compiled and captured. Cheap enough for large N, and strictly
                    harsher than the call site: the allocator and L2 have accumulated
                    state that a fresh process does not have. A rule that is stable here
                    is stable at prime time; the converse does not follow, which is why
                    `fresh` is the mode that decides anything.

Both modes interleave the arms and give every arm the same number of asks, for the same
reason `bench/abba.py` reverses its round order: an asymmetry in when the arms are asked
is an asymmetry in what they are asked.

    python3 bench/probes/g43_stable_tiles/prime_stability.py --mode fresh \
        --ids 2 3 --arms v42_hot_tuned_tile v43_replicated_tile --primings 8

[L41]: a probe may propose; it may never conclude. What this one can conclude about is
the tuner's own output, which is a discrete fact and not a measurement of the card.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _reference(tag: str):
    spec = importlib.util.spec_from_file_location(
        tag, REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def _plan(model) -> list:
    return [getattr(model, "attn_form", None),
            list(getattr(model, "attn_tile", ()) or ()),
            getattr(model, "attn_reason", None)]


def _setup(config_id: int, arms: list[str]):
    import torch
    sys.path.insert(0, str(REPO))
    from bench.candidates import REGISTRY
    from bench.matrix import BY_ID

    cfg = BY_ID[config_id]
    ref = _reference(f"ref_g43_{config_id}")
    tcfg = ref.TransformerConfig(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, d_model=cfg.d_model,
        num_heads=cfg.heads, ffn_dim=cfg.ffn_dim, num_layers=cfg.layers,
        causal=cfg.causal)
    tcfg.validate()
    dev = __import__("torch").device("cuda")

    torch.manual_seed(1234)
    base = ref.BaselineTransformer(tcfg)
    models = {}
    for name in arms:
        mdl = REGISTRY[name].build(ref.BaselineTransformer)(tcfg)
        ref.copy_model_weights(base, mdl)
        models[name] = mdl.to(device=dev, dtype=torch.float32).eval()
    del base
    x, m = ref.generate_random_case(tcfg, dev, torch.float32, seed=1234,
                                    padding_ratio=0.0, input_scale=1.0)
    return models, x, m


def run_resident(config_id: int, arms: list[str], primings: int) -> dict:
    """N asks per arm, interleaved, on models that are already built and captured."""
    import torch
    models, x, m = _setup(config_id, arms)

    first = {}
    with torch.inference_mode():
        for name in arms:                     # the REAL prime, one per arm
            models[name](x, m)
            first[name] = _plan(models[name])

    seq: dict[str, list] = {n: [] for n in arms}
    for r in range(primings):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for name in order:
            models[name].attn_reason = "undecided"
            models[name]._decide_attn(x)
            seq[name].append(_plan(models[name]))
    return {"config_id": config_id, "mode": "resident", "primings": primings,
            "arms": arms, "first_prime": first, "sequence": seq}


def run_fresh_once(config_id: int, arms: list[str]) -> dict:
    """ONE process, every arm built and primed ONCE. `bench/abba.py`'s regime exactly.

    THE ARMS ARE ALL BUILT AND ALL PRIMED, and that is not a detail. Measured here: with
    ONE arm alone in a process, both v42 and v43 select the same plan in 8 of 8 processes
    on BOTH shapes -- including the shape finding 53 recorded v42 moving three ways on.
    The instability does not exist in a one-arm process. It appears when a second model
    is resident and has already been built, primed and run, which is what `abba.py` does
    and therefore what every ranking of these two candidates is taken in.

    So a one-arm-per-process probe would have reported "both arms stable, nothing to fix"
    and been wrong for exactly the reason finding 53 was wrong about its blast radius:
    the probe's regime was not the call site's. The arms' ORDER is alternated by the
    caller so that neither arm is always the one that primes into a busier machine.
    """
    import torch
    models, x, m = _setup(config_id, arms)
    plans = {}
    with torch.inference_mode():
        for name in arms:
            models[name](x, m)
            plans[name] = _plan(models[name])
    return {"config_id": config_id, "mode": "fresh", "order": list(arms),
            "plans": plans}


def _summarise(rows: list[dict]) -> None:
    by = {}
    for r in rows:
        if r["mode"] == "fresh":
            for name, p in r["plans"].items():
                by.setdefault((r["config_id"], name), []).append(
                    (p[0], tuple(p[1])))
        else:
            for name, seq in r["sequence"].items():
                by.setdefault((r["config_id"], name), []).extend(
                    (p[0], tuple(p[1])) for p in seq)
    print(f"\n{'cfg':>4} {'arm':<24} {'asks':>5} {'distinct':>9}  plans")
    for (cid, name), picks in sorted(by.items()):
        c = Counter(picks)
        shown = "  ".join(f"{f}{t}x{n}" for (f, t), n in c.most_common())
        flag = "" if len(c) == 1 else "   <-- UNSTABLE"
        print(f"{cid:>4} {name:<24} {len(picks):>5} {len(c):>9}  {shown}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--primings", type=int, default=8)
    ap.add_argument("--mode", choices=("fresh", "resident"), default="fresh")
    ap.add_argument("--out", default=None)
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._child == "resident":
        print("<<<JSON>>>" + json.dumps(
            run_resident(a.ids[0], a.arms, a.primings)))
        return 0
    if a._child == "fresh":
        print("<<<JSON>>>" + json.dumps(run_fresh_once(a.ids[0], a.arms)))
        return 0

    rows = []
    for cid in a.ids:
        if a.mode == "resident":
            jobs = [(["--ids", str(cid), "--arms", *a.arms,
                      "--primings", str(a.primings), "--_child", "resident"])]
        else:
            # One process per replicate, EVERY arm built and primed in it -- abba.py's
            # regime. The prime ORDER alternates so neither arm is always the one that
            # primes into a machine the other has already warmed.
            jobs = []
            for i in range(a.primings):
                order = a.arms if i % 2 == 0 else list(reversed(a.arms))
                jobs.append(["--ids", str(cid), "--arms", *order, "--_child", "fresh"])
        for argv in jobs:
            p = subprocess.run([sys.executable, str(Path(__file__).resolve())] + argv,
                               capture_output=True, text=True, cwd=str(REPO))
            tag = [l for l in p.stdout.splitlines() if l.startswith("<<<JSON>>>")]
            if not tag:
                print(f"cfg {cid} {' '.join(argv)}: FAILED\n"
                      f"{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
                continue
            rows.append(json.loads(tag[0][len("<<<JSON>>>"):]))
    _summarise(rows)
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
