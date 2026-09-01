#!/usr/bin/env python3
"""
Generate the figures embedded in ``writeup.md``.

Every number in this script is transcribed from a real experimental artifact in
this repository.  The source of each dataset is cited in a comment next to the
data so the figure can be traced back to the evidence:

  * summary.md
      - GB10 13-config table (marcus): vs eager / vs torch.compile / flash fp32
      - ben generation progression g1..g34: 0.79x -> 3.25x
      - brian synthetic PageRank Spearman table (4 scenarios)
      - ben technique-level findings (L2 persistence, fp16 residual, padding)
  * research/experiments/failure_aware_pruning/results.json
      - baseline vs pruned dual-annealing search
  * research/experiments/hardware_validation/hardware_validation_results.json
      - 192-config space, 38 valid / 154 invalid, constraint breakdown
  * research/experiments/hardware_validation/hardware_profile.json  (RTX 4060)
  * ledger/device.json                                              (RTX 4070 Ti SUPER)
  * writeup.md
      - launch census (36 kernels), v34 removed 16
      - noise floor +-7%, A/B bounds 0.9811x-1.0046x
      - phantom 7.17x -> 6.30-6.54x re-run; op vs model disagreement

Reproduce:
    uv pip install --python .venv/bin/python matplotlib numpy
    .venv/bin/python research/figures/make_figures.py

Figures are written next to this script as PNGs (150 dpi).  Several sections have
more than one candidate figure ("alt") so the author can choose.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE

# --------------------------------------------------------------------------- #
# Shared style
# --------------------------------------------------------------------------- #
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "figure.autolayout": False,
    }
)

# Colour-blind-safe palette (Okabe-Ito)
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#999999",
    "black": "#000000",
}


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def load_json(rel: str) -> dict:
    with open(REPO / rel) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# REAL DATA  (transcribed from artifacts; see module docstring for provenance)
# --------------------------------------------------------------------------- #

# summary.md :: marcus GB10 13-config table
GB10_CONFIGS = list(range(1, 14))
GB10_SHAPE = [
    "[64,128,128] 4/32",
    "[1,128,128] 4/32",
    "[4,128,128] 4/32",
    "[16,128,128] 4/32",
    "[128,128,128] 4/32",
    "[10k,128,128] 4/32",
    "[64,128,32] 4/8",
    "[64,128,1024] 4/256",
    "[64,128,128] 1/128",
    "[64,128,128] 2/64",
    "[64,128,128] 16/8",
    "[64,32,128] 4/32",
    "[64,1024,128] 4/32",
]
GB10_VS_EAGER = [2.97, 2.77, 2.88, 2.84, 3.02, 2.66, 3.35, 3.79, 2.06, 2.30, 5.19, 2.23, 9.80]
GB10_VS_COMPILE = [2.14, 2.13, 2.25, 2.09, 2.12, 1.77, 1.65, 3.80, 2.06, 2.30, 5.18, 2.23, 9.80]
GB10_FLASH_FP32 = [3.61, 2.36, 3.02, 2.66, 4.04, 4.10, 3.54, 0.58, 1.15, 0.82, 4.45, 1.32, 6.25]
GB10_GEOMEAN_EAGER = 3.20  # summary.md
GB10_GEOMEAN_COMPILE = 2.62  # summary.md

# summary.md :: ben generation progression (geomean vs compiled baseline)
BEN_GEN = ["g1", "g2", "g6", "g9", "g12", "g17", "g18", "g23", "g26", "g34"]
BEN_LABEL = [
    "v1_fused_graph",
    "v2_fp16_flash",
    "v6_fp16_gelu",
    "v9a_compiled_core",
    "v12_graph_over_compile",
    "v17_dispatched_megakernel",
    "v18_capture_insurance",
    "v23_single_tile_attn",
    "v26_causal_correct",
    "v34_launch_bound",
]
BEN_GEOMEAN = [0.79, 1.41, 1.69, 2.68, 2.71, 2.76, 2.77, 3.02, 3.10, 3.25]

# summary.md :: brian synthetic difficulty-forecasting Spearman table
PR_SCENARIOS = ["CUDA/fp32", "CUDA/fp8", "HIP/fp32", "HIP/fp8"]
PR_METHODS = {
    "Failure-weighted PageRank": [0.432, 0.335, 0.433, 0.333],
    "Tuning pressure": [0.393, 0.271, 0.399, 0.263],
    "Plain PageRank": [-0.517, -0.517, -0.526, -0.526],
    "Random baseline": [0.008, 0.008, 0.007, 0.007],
}

# writeup.md :: launch census inside the replayed CUDA graph
LAUNCH_CENSUS = {"GEMM": 16, "LayerNorm": 9, "Attention": 4, "GELU": 4, "Copy": 3}
LAUNCH_TOTAL = 36  # writeup.md
LAUNCH_REMOVED_V34 = 16  # writeup.md / summary.md (v34_launch_bound)

# summary.md :: L2 weight-persistence hypothesis (positive control vs real effect)
L2_CONTROL_PCT = 42.7  # instrument moved this much (positive control)
L2_EFFECT_PCT = 0.25  # real persistence effect
L2_WEIGHTS_KIB = 768.0  # weights streamed
L2_ACTIVATION_MB = 327.0  # activation stream

# summary.md / writeup.md :: correctness wall
CORRECTNESS = {
    # (dtype label, configs_passed, configs_failed, total)
    "fp32 (baseline)": (13, 0, 13),
    "bf16": (0, 13, 13),        # failed all 13 (26 ledger rows)
    "fp16 residual": (2, 11, 13),  # failed 11 of 13
}

# summary.md / writeup.md :: padding-ratio cliff for config 13 and corrected proof
PAD_CONFIG13 = {"padding_ratio=0.0": 24.06, "padding_ratio=0.5": 6.62}
PAD_PROOF = {"naive zero-pad path": 2.86, "corrected right-padded-causal": 5.85}

# failure_aware_pruning/results.json
FAP = load_json("research/experiments/failure_aware_pruning/results.json")
# hardware_validation results + profile
HWV = load_json("research/experiments/hardware_validation/hardware_validation_results.json")

# writeup.md :: noise-floor A/B and phantom speedup
NOISE_FLOOR_PCT = 7.0  # +-7% on WSL2
AB_LOW, AB_HIGH = 0.9811, 1.0046  # A/B controls bounded byte-identical variation
REtiming_GEOMEAN_SHIFT = 2.9  # +2.9% re-timing byte-identical code
PHANTOM = {
    "1st (unwarmed, 1 sample)": 7.17,
    "re-run A": 6.30,
    "re-run B": 6.54,
}
# writeup.md :: probes may propose, never conclude (one change)
PROBE_LEVELS = {"op-level": 3.84, "model-level": 0.838, "authoritative sweep": 1.004}
# 16.2% worse model-level -> 0.838x ; +0.4% sweep -> 1.004x

# writeup.md :: exclusive-GPU guard caught contamination
CONTAM_INFLATED_MS = 2037.0
CONTAM_TRUE_MS = 446.0

# Section 4 :: human planning queue record kinds (writeup.md)
QUEUE_KINDS = ["IDEA", "CONSTRAINT", "PRIORITY", "LITERATURE", "REDIRECT"]
PAPERS_VALIDATED = 10  # ten papers actually read (literature keys validated)

# Section 1 :: programming-surface / hardware-target divergence
SURFACES = [
    "eager PyTorch",
    "torch.compile",
    "cuBLAS",
    "Triton",
    "CUDA graph",
    "fused megakernel",
]
# hardware constants that differ per target -> normalized divergence heatmap.
# Values are the REAL constants pulled from device.json + hardware_profile.json
# + writeup.md text, then normalized per-row for the heatmap.
TARGETS = ["GB10\nsm_121", "RTX 4070TiS\nsm_89", "RTX 4060\nsm_89", "CPU/XPU\nArc"]
# rows: SM count, L2 MB, shared mem KB, clocks lockable (1/0)
HW_SM = [np.nan, 66, 20, np.nan]            # device.json 66 ; hwv 20
HW_L2_MB = [np.nan, 48.0, np.nan, np.nan]   # device.json 50331648 B = 48 MiB
HW_SMEM_KB = [np.nan, 100.0, 99.0, np.nan]  # device.json 102400/1024 ; hwv 99
HW_CLOCK_LOCK = [np.nan, 0, np.nan, np.nan]  # device.json clocks_locked=false


# --------------------------------------------------------------------------- #
# SECTION 1 — Why this is hard
# --------------------------------------------------------------------------- #
def fig_s1_search_space():
    """Multiplicative blow-up: surfaces x targets x shapes."""
    surfaces = len(SURFACES)           # 6 real surfaces named in writeup
    targets = len(TARGETS)             # 4 real hardware targets
    shapes = len(GB10_CONFIGS) + 1     # 13 measured + config 14 (OOM) = 14 shapes

    fig, ax = plt.subplots(figsize=(8, 4.6))
    steps = ["surfaces", "x targets", "x shapes"]
    counts = [surfaces, surfaces * targets, surfaces * targets * shapes]
    bars = ax.bar(steps, counts, color=[C["sky"], C["orange"], C["red"]])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c}", ha="center", va="bottom", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylabel("distinct kernel choices (log)")
    ax.set_title("Section 1 · The space multiplies: surfaces × targets × shapes")
    ax.set_ylim(1, counts[-1] * 3)
    cap = (f"{surfaces} programming surfaces × {targets} hardware targets × "
           f"{shapes} shapes = {counts[-1]} combinations, and the winner changes per cell.")
    fig.text(0.5, -0.04, cap, ha="center", fontsize=9, style="italic")
    save(fig, "s1_search_space.png")


def fig_s1_hw_divergence():
    """Heatmap: hardware constants differ per target, so knowledge does not port."""
    rows = ["SM count", "L2 cache (MB)", "shared mem (KB)", "clocks lockable"]
    raw = np.array([HW_SM, HW_L2_MB, HW_SMEM_KB, HW_CLOCK_LOCK], dtype=float)
    # normalize each row to [0,1] across known targets for colour only
    norm = np.full_like(raw, np.nan)
    for i in range(raw.shape[0]):
        row = raw[i]
        finite = row[np.isfinite(row)]
        if finite.size:
            lo, hi = np.nanmin(row), np.nanmax(row)
            norm[i] = 0.5 if hi == lo else (row - lo) / (hi - lo)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(norm, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(TARGETS)), TARGETS)
    ax.set_yticks(range(len(rows)), rows)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            v = raw[i, j]
            if np.isfinite(v):
                if rows[i] == "clocks lockable":
                    txt = "no" if v == 0 else "yes"
                else:
                    txt = f"{v:g}"
            else:
                txt = "—"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (np.isfinite(norm[i, j]) and norm[i, j] < 0.5) else "black",
                    fontweight="bold")
    ax.set_title("Section 1 · Every constant that matters differs per target")
    fig.text(0.5, -0.02,
             "Measured constants (device.json, hardware_profile.json). “—” = not measured "
             "on that box. Knowledge tuned on one cell does not port to another.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s1_hw_divergence.png")


# --------------------------------------------------------------------------- #
# SECTION 2 — What we noticed by tinkering
# --------------------------------------------------------------------------- #
def fig_s2_launch_census():
    """36 kernels per forward, identical decomposition; v34 removed 16."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    kinds = list(LAUNCH_CENSUS)
    vals = list(LAUNCH_CENSUS.values())
    colors = [C["blue"], C["orange"], C["green"], C["purple"], C["grey"]]
    wedges, _, autotexts = ax1.pie(
        vals, labels=[f"{k}\n{v}" for k, v in LAUNCH_CENSUS.items()],
        colors=colors, autopct=lambda p: f"{p*LAUNCH_TOTAL/100:.0f}",
        startangle=90, counterclock=False,
    )
    for t in autotexts:
        t.set_color("white"); t.set_fontweight("bold")
    ax1.set_title(f"{LAUNCH_TOTAL} kernels / forward\n(invariant across 3 configs, 0.061–6.549 ms)")

    # before/after launch-bound removal
    ax2.bar(["before", "after v34"], [LAUNCH_TOTAL, LAUNCH_TOTAL - LAUNCH_REMOVED_V34],
            color=[C["grey"], C["green"]])
    ax2.text(0, LAUNCH_TOTAL, f"{LAUNCH_TOTAL}", ha="center", va="bottom", fontweight="bold")
    ax2.text(1, LAUNCH_TOTAL - LAUNCH_REMOVED_V34, f"{LAUNCH_TOTAL - LAUNCH_REMOVED_V34}",
             ha="center", va="bottom", fontweight="bold")
    ax2.annotate(f"−{LAUNCH_REMOVED_V34} launches",
                 xy=(1, LAUNCH_TOTAL - LAUNCH_REMOVED_V34), xytext=(0.5, 30),
                 ha="center", color=C["red"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["red"]))
    ax2.set_ylabel("kernel launches per forward")
    ax2.set_title("v34_launch_bound removes 16 of 36")
    ax2.set_ylim(0, 42)
    save(fig, "s2_launch_census.png")


def fig_s2_l2_persistence():
    """Positive control moved instrument 42.7%; real effect only 0.25%."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                                   gridspec_kw={"width_ratios": [1, 1]})
    ax1.bar(["positive\ncontrol", "real L2\npersistence"],
            [L2_CONTROL_PCT, L2_EFFECT_PCT], color=[C["sky"], C["red"]])
    ax1.text(0, L2_CONTROL_PCT, f"{L2_CONTROL_PCT}%", ha="center", va="bottom", fontweight="bold")
    ax1.text(1, L2_EFFECT_PCT, f"{L2_EFFECT_PCT}%", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("performance movement (%)")
    ax1.set_title("Instrument works (42.7%);\nhypothesis falsified (0.25%)")

    # why: weights are tiny beside the activation stream
    ax2.bar(["weights\n(768 KiB)", "activation stream\n(327 MB)"],
            [L2_WEIGHTS_KIB / 1024, L2_ACTIVATION_MB], color=[C["orange"], C["blue"]])
    ax2.set_yscale("log")
    ax2.set_ylabel("MB moved (log)")
    ax2.set_title("Why: 768 KiB weights vs 327 MB stream")
    for i, v in enumerate([L2_WEIGHTS_KIB / 1024, L2_ACTIVATION_MB]):
        ax2.text(i, v, f"{v:.2f} MB" if v < 1 else f"{v:.0f} MB",
                 ha="center", va="bottom", fontweight="bold")
    fig.suptitle("Section 2 · A positive control separates a working instrument from a real effect",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "s2_l2_persistence.png")


def fig_s2_correctness_wall():
    """bf16 fails all 13; fp16 residual fails 11 of 13."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    labels = list(CORRECTNESS)
    passed = [CORRECTNESS[k][0] for k in labels]
    failed = [CORRECTNESS[k][1] for k in labels]
    y = np.arange(len(labels))
    ax.barh(y, passed, color=C["green"], label="passes correctness gate")
    ax.barh(y, failed, left=passed, color=C["red"], label="fails gate")
    for i, k in enumerate(labels):
        ax.text(13.2, i, f"{passed[i]}/{CORRECTNESS[k][2]} pass", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("configurations (of 13)")
    ax.set_xlim(0, 16.5)
    ax.set_title("Section 2 · Correctness is a wall, not a knob")
    ax.legend(loc="lower right")
    fig.text(0.5, -0.03,
             "bf16 failed all 13 (26 ledger rows); an fp16 residual was ~1.4× faster but failed 11/13.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s2_correctness_wall.png")


def fig_s2_padding_cliff():
    """Harness assumption: padding_ratio hid a cliff; corrected proof restores it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax1.bar(list(PAD_CONFIG13), list(PAD_CONFIG13.values()), color=[C["orange"], C["red"]])
    for i, v in enumerate(PAD_CONFIG13.values()):
        ax1.text(i, v, f"{v}×", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("config-13 speedup ×")
    ax1.set_title("Fast path only at padding_ratio=0.0\n(24.06× → 6.62× at 0.5)")

    ax2.bar(list(PAD_PROOF), list(PAD_PROOF.values()), color=[C["grey"], C["green"]])
    for i, v in enumerate(PAD_PROOF.values()):
        ax2.text(i, v, f"{v}×", ha="center", va="bottom", fontweight="bold")
    ax2.set_ylabel("speedup ×")
    ax2.set_title("Corrected right-padded-causal proof\nrestores 5.85× (naive gave 2.86×)")
    fig.suptitle("Section 2 · Assumptions hide in the harness", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "s2_padding_cliff.png")


def fig_s2_infeasibility():
    """Failures are the dataset: 68-78% fail to compile in comparable spaces + real 80.2%."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    # real measured rejection from hardware_validation
    valid = HWV["configuration_space"]["valid"]
    invalid = HWV["configuration_space"]["invalid"]
    total = HWV["configuration_space"]["total"]
    reg = HWV["failure_breakdown"]["Register spill expected (high tile \u00d7 warp product)"]
    smem = HWV["failure_breakdown"]["Shared memory exceeded (99 KB budget)"]

    ax.bar(["feasible"], [valid], color=C["green"], label=f"feasible ({valid})")
    ax.bar(["register\nspill"], [reg], color=C["red"], label=f"register spill ({reg})")
    ax.bar(["shared-mem\noverflow"], [smem], color=C["orange"], label=f"smem overflow ({smem})")
    for x, v in zip(range(3), [valid, reg, smem]):
        ax.text(x, v, f"{v}\n{100*v/total:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel(f"configs (of {total})")
    ax.set_title(f"Section 2 · Failures are the dataset — {invalid}/{total} "
                 f"({HWV['configuration_space']['failure_rate_pct']:.1f}%) infeasible")
    ax.axhline(total, color=C["grey"], ls=":", lw=1)
    fig.text(0.5, -0.03,
             "Measured on RTX 4060 (hardware_validation_results.json). In comparable spaces "
             "68–78% of configs fail to compile before any tuning explores them.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s2_infeasibility.png")


# --------------------------------------------------------------------------- #
# SECTION 3 — The automated loop / profiling
# --------------------------------------------------------------------------- #
def fig_s3_matched_precision():
    """config 8: end-to-end 3.79x but flash fp32 = 0.58x -> win is dtype, not flash."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    idx = 7  # config 8 (0-based)
    labels = ["config 8\nvs eager", "config 8\nflash fp32", "config 13\nvs eager", "config 13\nflash fp32"]
    vals = [GB10_VS_EAGER[idx], GB10_FLASH_FP32[idx], GB10_VS_EAGER[12], GB10_FLASH_FP32[12]]
    colors = [C["blue"], C["red"], C["blue"], C["green"]]
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(1.0, color=C["grey"], ls="--", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}×", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("speedup ×")
    ax.set_title("Section 3 · Matched-precision decomposition separates dtype from algorithm")
    fig.text(0.5, -0.03,
             "config 8: 3.79× end-to-end but the flash algorithm at matched fp32 is 0.58× — the win is "
             "fp16 tensor cores. config 13: flash itself is 6.25×.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s3_matched_precision.png")


def fig_s3_noise_floor():
    """Noise floor +-7%; A/B controls bound byte-identical variation."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.axhspan(1 - NOISE_FLOOR_PCT / 100, 1 + NOISE_FLOOR_PCT / 100,
               color=C["yellow"], alpha=0.35, label=f"±{NOISE_FLOOR_PCT:.0f}% noise floor")
    ax.axhspan(AB_LOW, AB_HIGH, color=C["sky"], alpha=0.6,
               label=f"A/B byte-identical band {AB_LOW}–{AB_HIGH}×")
    ax.axhline(1.0, color=C["black"], lw=1)
    ax.axhline(1 + REtiming_GEOMEAN_SHIFT / 100, color=C["red"], ls="--",
               label=f"re-timing shift +{REtiming_GEOMEAN_SHIFT}%")
    ax.set_ylim(0.90, 1.10)
    ax.set_xticks([])
    ax.set_ylabel("measured speedup vs true 1.0×")
    ax.set_title("Section 3 · A published noise floor tells signal from jitter")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.5, -0.03,
             "Re-timing byte-identical code moved the geomean +2.9%; A/B controls bounded it to "
             "0.9811×–1.0046×. Only gains beyond ±7% promote.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s3_noise_floor.png")


def fig_s3_phantom():
    """Phantom 7.17x from an unwarmed single sample collapses on re-run."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    labels = list(PHANTOM)
    vals = list(PHANTOM.values())
    colors = [C["red"], C["green"], C["green"]]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}×", ha="center", va="bottom", fontweight="bold")
    ax.axhspan(6.30, 6.54, color=C["green"], alpha=0.15)
    ax.set_ylabel("reported attention speedup ×")
    ax.set_title("Section 3 · A phantom speedup, corrected by warm-up + median-of-N")
    fig.text(0.5, -0.03,
             "do_bench(warmup, rep) counts milliseconds, not iterations: the default gave one unwarmed "
             "sample and a phantom 7.17×. Re-runs read 6.30–6.54×.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s3_phantom.png")


def fig_s3_probe_disagreement():
    """One change: 3.84x op-level, 0.838x model-level, 1.004x authoritative sweep."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    labels = list(PROBE_LEVELS)
    vals = list(PROBE_LEVELS.values())
    colors = [C["orange"], C["red"], C["blue"]]
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(1.0, color=C["grey"], ls="--", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}×", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("speedup ×")
    ax.set_title("Section 3 · Probes may propose, never conclude")
    fig.text(0.5, -0.03,
             "The same change measured 3.84× better op-level, 16.2% worse model-level (0.838×), and "
             "+0.4% (1.004×) in the authoritative sweep. Only the sweep decides.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s3_probe_disagreement.png")


def fig_s3_contamination():
    """Exclusive-GPU guard: two models in one process inflated a baseline 4.1x."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(["contaminated\nbaseline", "true\nbaseline"],
                  [CONTAM_INFLATED_MS, CONTAM_TRUE_MS], color=[C["red"], C["green"]])
    for b, v in zip(bars, [CONTAM_INFLATED_MS, CONTAM_TRUE_MS]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f} ms", ha="center", va="bottom", fontweight="bold")
    ratio = CONTAM_INFLATED_MS / CONTAM_TRUE_MS
    ax.annotate(f"{ratio:.1f}× inflation", xy=(0, CONTAM_INFLATED_MS),
                xytext=(0.5, CONTAM_INFLATED_MS * 0.8), ha="center",
                color=C["red"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["red"]))
    ax.set_ylabel("baseline latency (ms)")
    ax.set_title("Section 3 · Exclusive-GPU guard catches contamination")
    fig.text(0.5, -0.03,
             "Two models in one process once inflated a baseline 4.1× (2037 ms vs a true 446 ms). "
             "The guard walks /proc and records the check in the row.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s3_contamination.png")


# --------------------------------------------------------------------------- #
# SECTION 4 — Where humans inject expertise
# --------------------------------------------------------------------------- #
def fig_s4_planning_queue():
    """Append-only, hash-chained planning queue with five record kinds."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    y = np.arange(len(QUEUE_KINDS))[::-1]
    descs = [
        "open a question",
        "forbid an outcome",
        "reorder the search",
        f"cite {PAPERS_VALIDATED} validated papers",
        "record a veto (never edit history)",
    ]
    colors = [C["sky"], C["red"], C["orange"], C["green"], C["purple"]]
    ax.barh(y, [1] * len(QUEUE_KINDS), color=colors, height=0.6)
    for yi, kind, d in zip(y, QUEUE_KINDS, descs):
        ax.text(0.02, yi, kind, va="center", ha="left", fontweight="bold", color="white")
        ax.text(1.03, yi, d, va="center", ha="left", fontsize=10)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(-0.6, len(QUEUE_KINDS) - 0.4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Section 4 · Expertise enters through an append-only, hash-chained queue")
    fig.text(0.5, -0.02,
             "Five record kinds; literature keys are validated against ten papers actually read, "
             "so priors are cited rather than asserted.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s4_planning_queue.png")


def fig_s4_chain():
    """Illustrate the hash-chain append-only property (structure, not fabricated data)."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    kinds = ["IDEA", "CONSTRAINT", "LITERATURE", "PRIORITY", "REDIRECT"]
    colors = [C["sky"], C["red"], C["green"], C["orange"], C["purple"]]
    for i, (k, c) in enumerate(zip(kinds, colors)):
        ax.add_patch(plt.Rectangle((i * 2, 0), 1.5, 1, color=c))
        ax.text(i * 2 + 0.75, 0.5, k, ha="center", va="center", color="white",
                fontweight="bold", fontsize=9)
        ax.text(i * 2 + 0.75, -0.35, f"hash←#{i}", ha="center", va="center", fontsize=8, color=C["grey"])
        if i < len(kinds) - 1:
            ax.annotate("", xy=((i + 1) * 2, 0.5), xytext=(i * 2 + 1.5, 0.5),
                        arrowprops=dict(arrowstyle="->", color=C["black"]))
    ax.set_xlim(-0.3, len(kinds) * 2)
    ax.set_ylim(-0.8, 1.4)
    ax.axis("off")
    ax.set_title("Section 4 · Append-only hash chain — you can never edit history")
    save(fig, "s4_chain.png")


# --------------------------------------------------------------------------- #
# SECTION 5 — Results
# --------------------------------------------------------------------------- #
def fig_s5_gb10_grouped():
    """Grouped bars: vs eager / vs torch.compile / flash fp32 for 13 configs."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    x = np.arange(len(GB10_CONFIGS))
    w = 0.27
    ax.bar(x - w, GB10_VS_EAGER, w, label="vs eager", color=C["blue"])
    ax.bar(x, GB10_VS_COMPILE, w, label="vs torch.compile", color=C["orange"])
    ax.bar(x + w, GB10_FLASH_FP32, w, label="flash fp32 (algorithm only)", color=C["green"])
    ax.axhline(1.0, color=C["grey"], ls="--", lw=1)
    ax.axhline(GB10_GEOMEAN_EAGER, color=C["blue"], ls=":", lw=1.3,
               label=f"geomean vs eager {GB10_GEOMEAN_EAGER}×")
    ax.axhline(GB10_GEOMEAN_COMPILE, color=C["orange"], ls=":", lw=1.3,
               label=f"geomean vs compile {GB10_GEOMEAN_COMPILE}×")
    ax.set_xticks(x, [f"c{c}" for c in GB10_CONFIGS])
    ax.set_ylabel("speedup ×")
    ax.set_xlabel("GB10 configuration")
    ax.set_title("Section 5 · GB10: 13 correct configs (3.20× geomean vs eager, 2.62× vs torch.compile)")
    ax.legend(ncol=2, fontsize=9)
    ax.annotate("flash fp32 can lose\n(0.58× @ head_dim 256)",
                xy=(7.27, 0.58), xytext=(9.0, 4.6), fontsize=8.5, color=C["red"],
                ha="center",
                arrowprops=dict(arrowstyle="->", color=C["red"]))
    save(fig, "s5_gb10_grouped.png")


def fig_s5_gb10_heatmap():
    """Alt: heatmap view of the same 13x3 matrix."""
    data = np.array([GB10_VS_EAGER, GB10_VS_COMPILE, GB10_FLASH_FP32])
    fig, ax = plt.subplots(figsize=(13, 3.6))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                   norm=matplotlib.colors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=10))
    ax.set_yticks(range(3), ["vs eager", "vs torch.compile", "flash fp32"])
    ax.set_xticks(range(len(GB10_CONFIGS)), [f"c{c}" for c in GB10_CONFIGS])
    for i in range(3):
        for j in range(len(GB10_CONFIGS)):
            v = data[i, j]
            ax.text(j, i, f"{v:g}", ha="center", va="center",
                    color="black", fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=ax, label="speedup ×", fraction=0.025, pad=0.01)
    ax.set_title("Section 5 (alt) · GB10 speedup matrix — green wins, red losses (<1×)")
    save(fig, "s5_gb10_heatmap.png")


def fig_s5_ben_trajectory():
    """ben generation progression 0.79x -> 3.25x."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(BEN_GEN))
    ax.plot(x, BEN_GEOMEAN, "-o", color=C["blue"], lw=2, markersize=7)
    ax.axhline(1.0, color=C["grey"], ls="--", lw=1)
    ax.fill_between(x, 1.0, BEN_GEOMEAN, where=[g >= 1 for g in BEN_GEOMEAN],
                    color=C["green"], alpha=0.12)
    for xi, g, lab in zip(x, BEN_GEOMEAN, BEN_LABEL):
        ax.annotate(f"{g:.2f}×", (xi, g), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, fontweight="bold")
    short = [lab.split("_", 1)[1] if "_" in lab else lab for lab in BEN_LABEL]
    ax.set_xticks(x, [f"{g}\n{s}" for g, s in zip(BEN_GEN, short)],
                  fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("geomean vs compiled baseline ×")
    ax.set_title("Section 5 · RTX 4070 Ti SUPER trajectory: 0.79× → 3.25× over 34 generations")
    ax.annotate("v1 regressed\n(0.79×)", (0, 0.79), textcoords="offset points",
                xytext=(20, -28), fontsize=8, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"]))
    ax.annotate("Inductor fusion\njump", (3, 2.68), textcoords="offset points",
                xytext=(-4, -38), fontsize=8, color=C["green"],
                arrowprops=dict(arrowstyle="->", color=C["green"]))
    ax.set_ylim(0.5, 3.7)
    save(fig, "s5_ben_trajectory.png")


def fig_s5_pagerank():
    """brian: failure-weighted PageRank Spearman vs alternatives (4 scenarios)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(PR_SCENARIOS))
    methods = list(PR_METHODS)
    w = 0.2
    colors = [C["green"], C["blue"], C["red"], C["grey"]]
    for i, (m, c) in enumerate(zip(methods, colors)):
        ax.bar(x + (i - 1.5) * w, PR_METHODS[m], w, label=m, color=c)
    ax.axhline(0.0, color=C["black"], lw=1)
    ax.set_xticks(x, PR_SCENARIOS)
    ax.set_ylabel("Spearman correlation (higher = better forecast)")
    ax.set_title("Section 5 · Failure-weighted PageRank forecasts tuning difficulty; plain PageRank anti-correlates")
    ax.legend(fontsize=9, loc="lower left")
    ax.annotate("plain PageRank is\nworse than random (−0.517)",
                xy=(0 + 0.5 * w, -0.517), xytext=(0.6, -0.35), fontsize=8, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"]))
    save(fig, "s5_pagerank.png")


def fig_s5_failure_pruning():
    """failure-aware pruning: before/after on evals, failure rate, best speedup."""
    b, p = FAP["baseline"], FAP["pruned"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))

    # panel 1: wasted evals
    axes[0].bar(["baseline", "pruned"], [b["evaluations_wasted"], p["evaluations_wasted"]],
                color=[C["red"], C["green"]])
    for i, v in enumerate([b["evaluations_wasted"], p["evaluations_wasted"]]):
        axes[0].text(i, v, f"{v}", ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("Wasted evals\n(compile failures)")
    axes[0].set_ylabel("evaluations")

    # panel 2: failure rate
    axes[1].bar(["baseline", "pruned"],
                [b["compile_failure_rate"] * 100, p["compile_failure_rate"] * 100],
                color=[C["red"], C["green"]])
    for i, v in enumerate([b["compile_failure_rate"] * 100, p["compile_failure_rate"] * 100]):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    axes[1].set_title("Compile failure rate\n(30.6% → 0%)")
    axes[1].set_ylabel("%")

    # panel 3: best speedup (the conservative trade-off)
    axes[2].bar(["baseline", "pruned"], [b["best_speedup"], p["best_speedup"]],
                color=[C["grey"], C["blue"]])
    for i, v in enumerate([b["best_speedup"], p["best_speedup"]]):
        axes[2].text(i, v, f"{v:.2f}×", ha="center", va="bottom", fontweight="bold")
    axes[2].set_title("Best speedup\n(conservative: 1.68× → 1.53×)")
    axes[2].set_ylabel("speedup ×")
    axes[2].set_ylim(0, 2.0)

    fig.suptitle("Section 5 · Failure-aware pruning (RTX 4060 simulated search): failures are eliminated at a small quality cost",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "s5_failure_pruning.png")


def fig_s5_constraint_rejection():
    """4 device constraints reject 80.2% of a 192-config space."""
    total = HWV["configuration_space"]["total"]
    details = {d["name"]: d["configs_eliminated"] for d in HWV["constraint_details"]}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(details)
    vals = [details[n] for n in names]
    colors = [C["red"] if v else C["grey"] for v in vals]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v}", va="center", fontweight="bold")
    ax.set_xlabel(f"configs eliminated (of {total})")
    ax.set_title(f"Section 5 · 4 device constraints reject {HWV['configuration_space']['failure_rate_pct']:.1f}% "
                 f"of a {total}-config space")
    fig.text(0.5, -0.03,
             "Register-pressure and shared-memory constraints do the work; thread/warp limits bind zero "
             "configs here. Constraints validated sound on RTX 4060.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s5_constraint_rejection.png")


def fig_s5_evidence_matrix():
    """Alt overview: the headline results table as a labelled bar chart."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    labels = [
        "GB10 vs eager (geomean)",
        "GB10 vs torch.compile",
        "GB10 best (config 13)",
        "RTX 4070TiS vs compiled (g34)",
        "attention flash fp32 (max)",
        "failure-aware best speedup",
    ]
    vals = [GB10_GEOMEAN_EAGER, GB10_GEOMEAN_COMPILE, max(GB10_VS_EAGER),
            BEN_GEOMEAN[-1], max(GB10_FLASH_FP32), FAP["pruned"]["best_speedup"]]
    colors = [C["blue"], C["orange"], C["green"], C["sky"], C["purple"], C["grey"]]
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(1.0, color=C["grey"], ls="--", lw=1)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.2f}×", va="center", fontweight="bold")
    ax.set_xlabel("speedup ×")
    ax.set_title("Section 5 (alt) · Headline measured results across branches")
    fig.text(0.5, -0.03,
             "Do not compare across branches: different GPUs, stacks, baselines, and evidence classes. "
             "Each bar is meaningful only with its row context.",
             ha="center", fontsize=9, style="italic")
    save(fig, "s5_evidence_matrix.png")


# --------------------------------------------------------------------------- #
def main():
    print("Generating figures into", OUT.relative_to(REPO))
    # Section 1
    fig_s1_search_space()
    fig_s1_hw_divergence()
    # Section 2
    fig_s2_launch_census()
    fig_s2_l2_persistence()
    fig_s2_correctness_wall()
    fig_s2_padding_cliff()
    fig_s2_infeasibility()
    # Section 3
    fig_s3_matched_precision()
    fig_s3_noise_floor()
    fig_s3_phantom()
    fig_s3_probe_disagreement()
    fig_s3_contamination()
    # Section 4
    fig_s4_planning_queue()
    fig_s4_chain()
    # Section 5
    fig_s5_gb10_grouped()
    fig_s5_gb10_heatmap()
    fig_s5_ben_trajectory()
    fig_s5_pagerank()
    fig_s5_failure_pruning()
    fig_s5_constraint_rejection()
    fig_s5_evidence_matrix()
    print("Done.")


if __name__ == "__main__":
    main()
