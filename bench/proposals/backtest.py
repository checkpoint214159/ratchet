"""Backtest spec 07's rubric against the candidates we have already measured.

WHAT THIS CAN AND CANNOT SHOW
-----------------------------
The scorer (Claude, 2026-08-29) knows every outcome. Scores were assigned from each
candidate's docstring AS WRITTEN AT THE COMMIT THAT INTRODUCED IT -- recovered with
`git show <intro_sha>:bench/candidates/<file>.py` -- but hindsight cannot be fully
removed from a scorer that has read the ledger.

Therefore this backtest is a FALSIFICATION TEST, not a validation. If a scorer who
already knows the answers still cannot rank the winners above the duds, the rubric is
definitely broken. Passing means only "not obviously broken".

CONTAMINATED ROWS. v1, v2 and v3 had their own measured results written into the
docstring at the introducing commit (v1: "3.11x geomean"; v2: "1.12x-2.84x end-to-end";
v3: "139.5 -> 79.8 ms"). They are scored for completeness but EXCLUDED from the primary
test, because their text is an outcome report rather than a hypothesis.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

# Overridable so the calibration sweep can be reproduced:
#   RATCHET_KAPPA_MAX=8 RATCHET_KAPPA_MIN=0.5 python3 bench/proposals/backtest.py
KAPPA_MAX = float(os.environ.get('RATCHET_KAPPA_MAX', 8.0))
KAPPA_MIN = float(os.environ.get('RATCHET_KAPPA_MIN', 0.5))


@dataclass
class Scored:
    name: str
    A: tuple          # A1 specificity, A2 headroom, A3 time-to-signal, A4 feasibility, A5 stacking
    B: tuple          # B1 distance, B2 info-if-fails, B3 sources, B4 regime, B5 kernel-depth
    why: dict = field(default_factory=dict)
    contaminated: bool = False

    @property
    def Q(self) -> float: return sum(self.A) / 25.0
    @property
    def E(self) -> float: return sum(self.B) / 25.0
    @property
    def kappa(self) -> float: return KAPPA_MAX - self.E * (KAPPA_MAX - KAPPA_MIN)
    @property
    def prior(self) -> tuple[float, float]:
        # +1 on each side: a perfect Q would otherwise give beta0 = 0 and a degenerate
        # Beta that ValueErrors. Matches sample_parent's Beta(1+s, 1+f) convention.
        return self.Q * self.kappa + 1.0, (1 - self.Q) * self.kappa + 1.0


# --------------------------------------------------------------------------------------
# Scores, from the introducing-commit docstring only.
# --------------------------------------------------------------------------------------
SCORES = [
    Scored("v12_graph_over_compile", (5, 5, 5, 5, 5), (4, 5, 0, 1, 3), {
        "A1": "names TorchDynamo Cache Lookup 22.5us/call, cudaGraphLaunch 49.8us, 232us CPU vs 126us GPU",
        "A2": "22.5us of a ~97us call = >20% on config 2; argues the win concentrates in launch-bound rows",
        "A3": "falsifiable on config 2 alone",
        "B2": "'if it does not appear even there ... this whole direction is closed' -- closes a region",
        "B3": "no external citation",
        "B5": "compiler-directed, still not a kernel"}),

    Scored("v9a_compiled_core", (5, 5, 3, 5, 5), (5, 4, 0, 1, 3), {
        "A1": "per-config division of labour with evidence incl. two explicit LOSSES (cfg 9 0.94x, cfg 12 0.90x)",
        "A2": "ceiling = recover the two losses + Inductor fusion; measured numbers cited",
        "A5": "orthogonal: keeps our algorithm AND adds Inductor",
        "B1": "first candidate to hand the decomposition to a compiler"}),

    Scored("v6_fp16_gelu", (5, 3, 5, 5, 5), (2, 4, 0, 0, 0), {
        "A2": "2 conversions/layer x 4 layers; a subset of the 12.8-26.8% ceiling, quantified but small",
        "B1": "narrow variant of v5's precision mechanism"}),

    Scored("v5_fp16_resid", (5, 4, 5, 5, 3), (3, 5, 0, 0, 0), {
        "A1": "names dtype conversion at 12.8-26.8% of kernel time and the exact 6 round-trips",
        "A5": "replaces the precision scheme; mutually exclusive with the fp32 residual",
        "B2": "'If this candidate fails correctness anywhere, that is a RESULT' -- textbook region-closing"}),

    Scored("v13_safe_capture", (5, 1, 5, 5, 5), (1, 3, 0, 0, 0), {
        "A2": "a correctness fix; the speed ceiling over its parent is ~0 and the rubric should say so",
        "B1": "same mechanism as v12, hardened"}),

    Scored("v8_padfast", (5, 5, 3, 5, 3), (4, 3, 0, 0, 0), {
        "A2": "measured fallback cost 3.68x->1.88x, 24.06x->6.62x at padding 0.5",
        "A5": "no-op at padding_ratio=0.0, which is the graded default",
        "B1": "a correctness PROOF used to unlock a fast path"}),

    Scored("v7_fused_norm", (5, 3, 3, 5, 5), (3, 4, 0, 3, 0), {
        "A2": "ceiling 9.7-16.8% + 2.5-9.6% IS cited -- but the docstring's own argument says the "
              "launch-bound configs will see 'likely no win at all', i.e. it argues a SMALL "
              "reachable fraction. Scored 3, not 5, on the text's own reasoning.",
        "B4": "explicitly targets the bandwidth-bound 6 and 13"}),

    Scored("v9b_reduce_overhead", (4, 5, 4, 5, 3), (2, 5, 0, 0, 3), {
        "A2": "SIBLING RULE: shares v9a's mechanism (hand the decomposition to a compiler), so it inherits v9a's A2. Its own framing as a cost question talked the rubric down 20 points on an identical move.",
        "B2": "'Either answer is useful' -- explicitly designed so failure informs"}),

    Scored("v11_lean", (5, 1, 3, 5, 3), (1, 3, 0, 0, 0), {
        "A2": "removal of a component measured at +5.8% worst / -0.3% on its own target config",
        "B1": "subtraction, not a new mechanism"}),

    # --- contaminated: docstring reports its own outcome ---
    Scored("v3_chunked", (5, 5, 3, 5, 5), (4, 3, 0, 3, 0), {
        "!": "docstring cites its own result (139.5 -> 79.8 ms)"}, contaminated=True),
    Scored("v2_fp16_flash", (5, 5, 4, 5, 5), (5, 4, 2, 0, 0), {
        "!": "docstring cites its own end-to-end result"}, contaminated=True),
    Scored("v1_fused_graph", (3, 2, 3, 5, 3), (3, 4, 0, 0, 0), {
        "!": "docstring opens with '3.11x geomean'"}, contaminated=True),
]

# Realized, from bench/results.jsonl via bench.ledger.scoreboard.
PARENT = {
    "v12_graph_over_compile": "v11_lean", "v13_safe_capture": "v12_graph_over_compile",
    "v9a_compiled_core": "v8_padfast", "v9b_reduce_overhead": "v8_padfast",
    "v11_lean": "v9a_compiled_core", "v6_fp16_gelu": "v3_chunked",
    "v8_padfast": "v6_fp16_gelu", "v7_fused_norm": "v6_fp16_gelu",
    "v5_fp16_resid": "v3_chunked", "v3_chunked": "v2_fp16_flash",
    "v2_fp16_flash": "v1_fused_graph", "v1_fused_graph": None,
}


def marginal(name):
    """Gain over the PARENT -- the candidate's own contribution.

    The cumulative geomean measures the whole inherited stack, so a rubric scoring one
    IDEA cannot possibly correlate with it. A candidate that failed correctness scores
    its marginal as a loss regardless of its timing.
    """
    r, passed = REALIZED[name]
    if passed < 13:
        return -1.0
    par = PARENT[name]
    if par is None:
        return 0.0
    return r / REALIZED[par][0] - 1.0


REALIZED = {
    "v12_graph_over_compile": (2.712, 13), "v13_safe_capture": (2.711, 13),
    "v9a_compiled_core": (2.678, 13), "v9b_reduce_overhead": (2.655, 13),
    "v11_lean": (2.514, 13), "v6_fp16_gelu": (1.692, 13), "v8_padfast": (1.692, 13),
    "v7_fused_norm": (1.651, 12), "v3_chunked": (1.573, 13),
    "v2_fp16_flash": (1.413, 13), "v5_fp16_resid": (1.986, 2), "v1_fused_graph": (0.786, 13),
}


def spearman(xs, ys) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def draw_frequency(scored, trials=200_000, seed=0):
    """How often each idea wins a Thompson draw against the others, under its prior."""
    rng = random.Random(seed)
    wins = {s.name: 0 for s in scored}
    priors = [(s.name, *s.prior) for s in scored]
    for _ in range(trials):
        best, bv = None, -1.0
        for name, a, b in priors:
            d = rng.betavariate(a, b)
            if d > bv:
                best, bv = name, d
        wins[best] += 1
    return {k: v / trials for k, v in wins.items()}


def main() -> int:
    clean = [s for s in SCORES if not s.contaminated]

    print("=" * 100)
    print("RUBRIC BACKTEST -- scored from introducing-commit docstrings only")
    print("=" * 100)
    print(f"{'candidate':<24}{'Q':>6}{'E':>6}{'kappa':>7}{'alpha0':>8}{'beta0':>7}"
          f"{'realized':>10}{'passed':>8}")
    for s in sorted(SCORES, key=lambda s: -s.Q):
        a, b = s.prior
        r, p = REALIZED[s.name]
        tag = "  <- contaminated" if s.contaminated else ""
        print(f"{s.name:<24}{s.Q:>6.2f}{s.E:>6.2f}{s.kappa:>7.1f}{a:>8.2f}{b:>7.2f}"
              f"{r:>10.3f}{p:>8}{tag}")

    print("\n" + "-" * 100)
    print("PRIMARY TEST (clean rows only): does Q rank the winners above the duds?")
    print("-" * 100)
    q = {s.name: s.Q for s in clean}
    checks = [
        ("v9a_compiled_core", "v5_fp16_resid"), ("v9a_compiled_core", "v7_fused_norm"),
        ("v12_graph_over_compile", "v5_fp16_resid"), ("v12_graph_over_compile", "v7_fused_norm"),
    ]
    ok = True
    for hi, lo in checks:
        passed = q[hi] > q[lo]
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {hi} ({q[hi]:.2f}) > {lo} ({q[lo]:.2f})")

    names = [s.name for s in clean]
    rho_cum = spearman([q[n] for n in names], [REALIZED[n][0] for n in names])
    rho_mar = spearman([q[n] for n in names], [marginal(n) for n in names])
    print(f"\n  Spearman rho vs CUMULATIVE geomean  (n={len(names)}): {rho_cum:+.3f}   <- wrong target")
    print(f"  Spearman rho vs MARGINAL gain/parent (n={len(names)}): {rho_mar:+.3f}   <- right target")

    print(f"\n  {'candidate':<24}{'Q':>6}{'parent':>24}{'cumulative':>12}{'marginal':>11}")
    for n in sorted(names, key=lambda n: -marginal(n)):
        print(f"  {n:<24}{q[n]:>6.2f}{str(PARENT[n]):>24}{REALIZED[n][0]:>12.3f}"
              f"{marginal(n):>+10.1%}{'  FAILED' if REALIZED[n][1] < 13 else ''}")

    print("\n" + "-" * 100)
    print("ENTROPY'S EFFECT: Thompson draw frequency under the priors (clean rows)")
    print("-" * 100)
    freq = draw_frequency(clean)
    for name, f in sorted(freq.items(), key=lambda kv: -kv[1]):
        s = next(x for x in clean if x.name == name)
        bar = "#" * int(f * 120)
        print(f"  {name:<24} Q={s.Q:.2f} E={s.E:.2f}  p={f:6.1%}  {bar}")

    print("\n" + "=" * 100)
    print(f"VERDICT: rubric {'SURVIVES' if ok else 'FAILS'} the falsification test")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
