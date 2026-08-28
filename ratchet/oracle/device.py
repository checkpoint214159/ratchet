"""Device introspection and calibration.

ZONE A -- IMMUTABLE. Do not edit as part of an optimization step.

Everything the dispatch table branches on comes from here. The rule is: query what the
driver will tell you, MEASURE what it will not, and never dispatch on a number you got
from a datasheet.

Two numbers in particular must be measured rather than computed:

  * Bandwidth. The textbook formula 2 * mem_clock * bus_width / 8 is unreliable on HBM3
    parts -- some driver versions report MEMORY_CLOCK_RATE as 0 or as a boost value that
    is never sustained. Real achievable bandwidth is 80-90% of theoretical anyway, and
    the ridge point you dispatch on should reflect the machine you have.

  * Launch overhead. It varies by driver, by WSL vs native, and by whether the GPU is
    shared. It is the entire basis of the launch-bound dispatch branch, so guessing 2us
    because a blog said so is not acceptable.

Peak FLOPs is the exception: there is no runtime query, so it comes from a table and is
tagged `peak_source="table"` so nobody mistakes it for a measurement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from functools import cache
from pathlib import Path
from typing import Optional

import torch

try:
    import triton
    import triton.testing as tt
    _HAS_TRITON = True
except Exception:  # pragma: no cover - triton absent is a valid state to report
    _HAS_TRITON = False


# --------------------------------------------------------------------------------------
# Peak dense throughput table.
#
# TFLOP/s, DENSE (not the 2:4-sparsity marketing numbers), BF16 with FP32 accumulate.
# Keyed by (compute_capability, device_name_substring). The name substring is needed
# because sm_89 covers both a 4090 and an L40S, and sm_120 covers several parts.
#
# NOTE ON GEFORCE PARTS: consumer cards run mma with .f32 accumulate at HALF the .f16
# rate. These numbers are the FP32-accumulate figures, which is what you get with BF16
# (BF16 has no FP16-accumulate path). If you switch a kernel to FP16 inputs with FP16
# accumulate you may see up to 2x this -- and you will almost certainly fail the
# abs < 0.002 tolerance. See docs/04-failure-modes.md.
#
# If your device is not here, ADD IT and cite the whitepaper in the comment. Do not
# guess; an inaccurate ridge point silently mis-routes the dispatch.
# --------------------------------------------------------------------------------------
_PEAK_BF16_DENSE_TFLOPS = {
    ("sm_80", "A100"): 312.0,     # Ampere whitepaper
    ("sm_86", "A10"): 125.0,      # Ampere whitepaper
    ("sm_89", "4090"): 165.2,     # Ada whitepaper Table 2, FP32-accumulate row
    ("sm_89", "4070 Ti SUPER"): 88.2,  # AD103, 66 SMs @ 2610 MHz boost x 512 FLOP/SM/clk
                                       # (Ada whitepaper FP32-accumulate dense rate; equals
                                       # 2x the 44.10 TFLOPS FP32 shader figure NVIDIA lists)
    ("sm_89", "L40"): 181.0,      # Ada whitepaper
    ("sm_90", "H100"): 989.5,     # Hopper whitepaper, SXM
    ("sm_90", "H200"): 989.5,     # same compute die as H100
    ("sm_100", "B200"): 2250.0,   # Blackwell datacenter
    ("sm_120", "5090"): 209.5,    # RTX Blackwell whitepaper, FP32-accumulate row
    ("sm_120", "5080"): 000.0,    # TODO: fill from whitepaper before relying on it
}

_L2_FLUSH_FLOOR_BYTES = 256 * 1024 * 1024  # never flush with less than this


@dataclass
class DeviceProfile:
    """Everything the dispatch table is allowed to branch on."""

    # --- queried -----------------------------------------------------------------
    device_name: str
    compute_capability: str          # "sm_120"
    sm_count: int
    warp_size: int
    smem_per_block_optin: int        # bytes -- the REAL budget, not the 48KB default
    smem_per_sm: int
    l2_cache_size: int
    regs_per_sm: int
    total_memory: int
    backend: str                     # "cuda" | "hip"

    # --- measured ----------------------------------------------------------------
    measured_bandwidth_gbs: float
    launch_overhead_us: float

    # --- table -------------------------------------------------------------------
    peak_bf16_tflops: float
    peak_source: str                 # always "table" -- there is no runtime query

    # --- derived -----------------------------------------------------------------
    ridge_point_flop_per_byte: float

    # --- provenance --------------------------------------------------------------
    torch_version: str = ""
    triton_version: str = ""
    driver_version: str = ""
    clocks_locked: bool = False
    locked_sm_clock_mhz: Optional[int] = None
    notes: list[str] = field(default_factory=list)

    @property
    def l2_flush_bytes(self) -> int:
        return max(_L2_FLUSH_FLOOR_BYTES, 4 * self.l2_cache_size)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# --------------------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------------------

@cache
def _torch_props(index: int = 0):
    return torch.cuda.get_device_properties(index)


@cache
def _triton_props(index: int = 0) -> dict:
    """Triton exposes exactly seven keys, and spells multiprocessor_count without the
    underscore that torch uses. Wrapped because it is absent on some backends."""
    if not _HAS_TRITON:
        return {}
    try:
        return dict(triton.runtime.driver.active.utils.get_device_properties(index))
    except Exception:
        return {}


@cache
def compute_capability(index: int = 0) -> str:
    p = _torch_props(index)
    return f"sm_{p.major}{p.minor}"


@cache
def smem_at_least(arch: str = "ampere", index: int = 0) -> bool:
    """Capability test by RESOURCE, not by compute capability.

    Borrowed from fla/utils/_device.py, and the reason is worth stating: asking
    "do I have at least as much shared memory as an A100" degrades correctly onto
    hardware nobody has enumerated yet, whereas `if cc >= 80` silently mis-answers on
    every part released after this file was written.
    """
    thresholds = {"ada": 101376, "ampere": 166912, "hopper": 232448, "default": 102400}
    want = thresholds.get(arch, thresholds["default"])
    try:
        return _torch_props(index).shared_memory_per_block_optin >= want
    except Exception:
        return False


def _peak_tflops(cc: str, name: str) -> tuple[float, list[str]]:
    notes: list[str] = []
    for (k_cc, k_name), v in _PEAK_BF16_DENSE_TFLOPS.items():
        if k_cc == cc and k_name.lower() in name.lower():
            if v <= 0.0:
                notes.append(f"peak table entry for ({cc}, {k_name}) is a placeholder")
            return v, notes
    notes.append(
        f"NO PEAK ENTRY for ({cc}, {name}). Ridge point is meaningless until you add one. "
        f"Find the FP32-accumulate dense BF16 figure in the architecture whitepaper."
    )
    return 0.0, notes


# --------------------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------------------

def measure_bandwidth_gbs(index: int = 0, repeats: int = 7) -> float:
    """Achievable HBM/GDDR bandwidth, measured with a streaming copy.

    Working set is sized to 4x L2 so the copy cannot be served from cache. We take the
    MINIMUM across repeats, not the mean: clock throttling and co-tenancy only ever make
    a sample worse, so the minimum is the closest thing to a clean run.
    """
    props = _torch_props(index)
    dev = torch.device(f"cuda:{index}")
    n_bytes = max(4 * props.L2_cache_size, 512 * 1024 * 1024)
    n_elems = n_bytes // 4

    src = torch.empty(n_elems, dtype=torch.float32, device=dev)
    dst = torch.empty_like(src)
    src.uniform_()

    # warm up: allocator, context, and any lazy kernel load
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize(dev)

    best_s = float("inf")
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src)
        end.record()
        torch.cuda.synchronize(dev)
        best_s = min(best_s, start.elapsed_time(end) / 1e3)

    moved_bytes = 2 * n_elems * 4  # one read + one write
    return moved_bytes / best_s / 1e9


def measure_launch_overhead_us(index: int = 0) -> float:
    """Per-launch cost, measured as the gap between graph and non-graph timing.

    do_bench pays the launch on every iteration; do_bench_cudagraph captures n_repeat
    unrolled calls into one graph so the launch is amortized away. The difference is the
    per-launch overhead. This is the number the launch-bound dispatch branch turns on, so
    it is measured rather than assumed.

    Falls back to a plain CUDA-event loop if Triton is unavailable, which measures
    launch+dispatch rather than launch alone -- close enough to branch on, and the
    fallback is recorded in notes.
    """
    dev = torch.device(f"cuda:{index}")
    x = torch.empty(1, device=dev)

    def trivial():
        x.add_(1.0)

    if _HAS_TRITON:
        try:
            with_launch_ms = tt.do_bench(trivial, warmup=25, rep=100, return_mode="min")
            graphed_ms = tt.do_bench_cudagraph(trivial, rep=20, return_mode="min")
            return max(0.0, (with_launch_ms - graphed_ms) * 1e3)
        except Exception:
            pass

    # Fallback: time an empty-ish launch directly.
    for _ in range(50):
        trivial()
    torch.cuda.synchronize(dev)
    n = 500
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        trivial()
    end.record()
    torch.cuda.synchronize(dev)
    return start.elapsed_time(end) * 1e3 / n


def detect_clock_lock(index: int = 0) -> tuple[bool, Optional[int]]:
    """Best-effort read of whether SM clocks are pinned.

    WSL frequently cannot lock clocks. That is not a blocker, but it MUST be recorded:
    an unlocked GPU boosts on short benchmarks and throttles on long ones, and a speedup
    measured against an unlocked baseline is not a result.
    """
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.applications.graphics,clocks.max.sm",
             "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return False, None
        applied, _maxsm = (s.strip() for s in out.stdout.strip().split(","))
        if applied.lower().startswith("n/a") or applied == "":
            return False, None
        return True, int(float(applied))
    except Exception:
        return False, None


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def calibrate(index: int = 0, cache_path: str | os.PathLike = "ledger/device.json",
              force: bool = False) -> DeviceProfile:
    """Build (or load) the DeviceProfile. Cached per (device, driver, torch, triton)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device. Everything in this package is conditional on real hardware; "
            "there is no CPU fallback and there should not be one."
        )

    props = _torch_props(index)
    cc = compute_capability(index)
    tri = _triton_props(index)
    torch_v = torch.__version__
    triton_v = triton.__version__ if _HAS_TRITON else ""
    driver_v = getattr(torch.version, "cuda", "") or ""

    key = f"{props.name}|{driver_v}|{torch_v}|{triton_v}"
    path = Path(cache_path)
    if path.exists() and not force:
        try:
            blob = json.loads(path.read_text())
            if blob.get("_key") == key:
                blob.pop("_key", None)
                return DeviceProfile(**blob)
        except Exception:
            pass  # a corrupt cache is not worth failing over; recalibrate

    notes: list[str] = []
    peak, peak_notes = _peak_tflops(cc, props.name)
    notes.extend(peak_notes)

    bw = measure_bandwidth_gbs(index)
    launch_us = measure_launch_overhead_us(index)
    if not (1.0 <= launch_us <= 50.0):
        notes.append(
            f"launch overhead {launch_us:.2f}us is outside the plausible 1-50us band; "
            f"the measurement is probably wrong, not the hardware"
        )
    if not _HAS_TRITON:
        notes.append("triton unavailable: launch overhead measured by fallback path")

    locked, locked_mhz = detect_clock_lock(index)
    if not locked:
        notes.append(
            "SM clocks are NOT locked. Use minimum-of-N rather than mean, interleave "
            "candidate and baseline timing in one process, and say so in the report."
        )

    ridge = (peak * 1e12) / (bw * 1e9) if bw > 0 and peak > 0 else 0.0

    profile = DeviceProfile(
        device_name=props.name,
        compute_capability=cc,
        sm_count=props.multi_processor_count,
        warp_size=int(tri.get("warpSize", getattr(props, "warp_size", 32))),
        smem_per_block_optin=int(
            getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)
        ),
        smem_per_sm=int(getattr(props, "shared_memory_per_multiprocessor", 0)),
        l2_cache_size=int(getattr(props, "L2_cache_size", 0)),
        regs_per_sm=int(getattr(props, "regs_per_multiprocessor", 0)),
        total_memory=int(props.total_memory),
        backend="hip" if torch.version.hip else "cuda",
        measured_bandwidth_gbs=bw,
        launch_overhead_us=launch_us,
        peak_bf16_tflops=peak,
        peak_source="table",
        ridge_point_flop_per_byte=ridge,
        torch_version=torch_v,
        triton_version=triton_v,
        driver_version=driver_v,
        clocks_locked=locked,
        locked_sm_clock_mhz=locked_mhz,
        notes=notes,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    blob = asdict(profile)
    blob["_key"] = key
    path.write_text(json.dumps(blob, indent=2, sort_keys=True))
    return profile


if __name__ == "__main__":
    p = calibrate(force=True)
    print(p.to_json())
    print()
    print(f"ridge point: {p.ridge_point_flop_per_byte:.0f} FLOP/B")
    print(f"  a kernel below this is memory bound and no amount of tuning will help;")
    print(f"  raise arithmetic intensity instead. See docs/01-architecture.md.")
    for n in p.notes:
        print(f"  NOTE: {n}")
