"""The FFN megakernel with the MMA accumulator width as a parameter, and the two
predicates that decide whether narrowing it is affordable.

WHY THIS FILE EXISTS
--------------------
Consumer Ada (sm_89) runs tensor-core FP16-with-FP32-accumulate at HALF the rate of
FP16-with-FP16-accumulate. That is a real hardware fact and it is confirmed here rather
than taken from a datasheet: `tl.dot(out_dtype=tl.float16)` really does emit

    mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

against `...f32.f16.f16.f32` for the fp32 accumulator, and in an MMA-saturated loop the
f16 form measures **1.569x** faster (36.6 us -> 23.4 us, 1024 mma instructions each).
So the instruction really is ~1.6x, and the question is only ever whether our kernels are
standing anywhere near that instruction.

`ffn_fused.py` is deliberately NOT modified. It is the frontier's kernel and three
candidates import it; this is a parameterized copy so the frontier's Triton cache key and
numerics are untouched.

THE TWO PREDICATES, AND WHY THEY ARE SCISSORS
---------------------------------------------
Narrowing the accumulator is affordable only where BOTH hold:

  1. `mma_bound(...)`  -- the shape is above the device's MEASURED ridge point, so
     tensor-core throughput is what limits the kernel and making the instruction faster
     makes the kernel faster.
  2. `accumulator_affordable(...)` -- the fp16 accumulator's error over a K-term dot
     product fits inside the locked absolute tolerance.

Both are functions of shapes and measured device properties. Neither can see a config id.

They are monotone in OPPOSITE directions in the contraction depth K, and in this
architecture K == d_model == ffn_dim, so one shape parameter drives both. Intensity rises
linearly in d_model; accumulator error rises as sqrt(K). On this device that opens a gap
of more than an order of magnitude between them, and `no_shape_satisfies_both()` reports
the gap rather than asserting it. See docs/findings/30.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

# fp16 has a 10-bit stored mantissa plus the implicit leading bit.
FP16_MANTISSA_BITS = 11
FP16_EPS = 2.0 ** -FP16_MANTISSA_BITS

# sm_89's tensor-core instruction is m16n8k16, so the SHALLOWEST contraction an MMA can
# perform is 16. Any accumulator-width argument that only works below K=16 has no
# hardware to run on -- which is exactly what happens here.
MMA_MIN_K = 16


@triton.jit
def _ffn_block_accum(
    XN,          # normalized input, fp16   [M, D]
    RES,         # residual stream, fp32    [M, D]
    W1, B1,      # [D, F] , [F]             fp16
    W2, B2,      # [F, D] , [D]             fp16
    Y,           # output, fp32             [M, D]
    M,
    D: tl.constexpr, F: tl.constexpr, BM: tl.constexpr,
    ACC_A: tl.constexpr,   # 0 = fp32 accumulator on the first GEMM, 1 = fp16
    ACC_B: tl.constexpr,   # 0 = fp32 accumulator on the second GEMM, 1 = fp16
):
    """Identical to `ffn_fused._ffn_block` at ACC_A == ACC_B == 0, including the exact
    erf GELU and the fp32 residual add, which finding 08 proved load-bearing and which
    this file never touches. The ONLY variable is the MMA accumulator width."""
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    rf = tl.arange(0, F)
    keep = rm < M

    xn = tl.load(XN + rm[:, None] * D + rd[None, :], mask=keep[:, None], other=0.0)
    w1 = tl.load(W1 + rd[:, None] * F + rf[None, :])
    w2 = tl.load(W2 + rf[:, None] * D + rd[None, :])
    b1 = tl.load(B1 + rf)
    b2 = tl.load(B2 + rd)

    # Site A -- first GEMM, contraction over D.
    if ACC_A == 0:
        h = tl.dot(xn, w1, out_dtype=tl.float32) + b1[None, :].to(tl.float32)
    else:
        h = (tl.dot(xn, w1, out_dtype=tl.float16) + b1[None, :]).to(tl.float32)

    # Exact erf GELU, matching the reference's approximate="none". Computed in fp32 in
    # both arms so the arms differ in the accumulator ONLY, not in the activation.
    h = h * 0.5 * (1.0 + tl.erf(h * 0.70710678118654752440))

    # Site B -- second GEMM, contraction over F.
    if ACC_B == 0:
        y = tl.dot(h.to(w2.dtype), w2, out_dtype=tl.float32) + b2[None, :].to(tl.float32)
    else:
        y = (tl.dot(h.to(w2.dtype), w2, out_dtype=tl.float16) + b2[None, :]).to(tl.float32)

    res = tl.load(RES + rm[:, None] * D + rd[None, :], mask=keep[:, None], other=0.0)
    tl.store(Y + rm[:, None] * D + rd[None, :], res + y, mask=keep[:, None])


def fused_ffn_accum(xn: torch.Tensor, res: torch.Tensor,
                    w1: torch.Tensor, b1: torch.Tensor,
                    w2: torch.Tensor, b2: torch.Tensor,
                    block_m: int = 64, num_warps: int = 4,
                    acc_a: int = 0, acc_b: int = 0) -> torch.Tensor:
    """res + (gelu(xn @ w1 + b1) @ w2 + b2), with per-site accumulator width.

    `w1`/`w2` are already TRANSPOSED relative to nn.Linear's [out, in] layout.
    """
    m, d = xn.shape
    f = w1.shape[1]
    y = torch.empty((m, d), device=xn.device, dtype=torch.float32)
    _ffn_block_accum[(triton.cdiv(m, block_m),)](
        xn, res, w1, b1, w2, b2, y, m,
        D=d, F=f, BM=block_m, ACC_A=acc_a, ACC_B=acc_b, num_warps=num_warps,
    )
    return y


# ---------------------------------------------------------------- predicate 1: is it fast

def activation_bytes_per_token(d_model: int, elem_size: int = 2) -> int:
    """HBM traffic the fused block moves per token: fp16 normalized input read, fp32
    residual read, fp32 output write. Weights are hoisted once and amortized, so they
    are not per-token traffic."""
    return d_model * elem_size + d_model * 4 + d_model * 4


def arithmetic_intensity(d_model: int, ffn_dim: int, elem_size: int = 2) -> float:
    """FLOP per byte of HBM traffic for the fused FFN block. Two GEMMs of 2*D*F FLOP."""
    flops = 4.0 * d_model * ffn_dim
    return flops / activation_bytes_per_token(d_model, elem_size)


def ridge_point(peak_flops: float, bandwidth_bytes_s: float) -> float:
    """The device's balance point in FLOP/B. Above it a kernel is compute-bound and the
    MMA instruction is on the critical path; below it the tensor cores idle waiting on
    HBM and a faster MMA changes nothing. Both inputs are MEASURED (ledger/device.json),
    never assumed."""
    return peak_flops / bandwidth_bytes_s


def mma_bound(d_model: int, ffn_dim: int, peak_flops: float,
              bandwidth_bytes_s: float, elem_size: int = 2) -> bool:
    """Would a faster MMA make this shape faster at all?"""
    return arithmetic_intensity(d_model, ffn_dim, elem_size) >= ridge_point(
        peak_flops, bandwidth_bytes_s)


# ------------------------------------------------------- predicate 2: is it accurate enough

def accumulator_error(k: int, magnitude: float = 1.0,
                      mantissa_bits: int = FP16_MANTISSA_BITS) -> float:
    """Absolute error an fp16 accumulator contributes to a K-term dot product.

    Each of the K sequential adds rounds to the accumulator's precision, and the roundoff
    of independent terms accumulates as a random walk, so the error grows as
    `eps * sqrt(K)` scaled by the magnitude of the running sum. This is the standard
    stochastic bound, not a worst case (the worst case is `eps * K`), and it is the
    OPTIMISTIC direction -- if the optimistic bound already fails, the real one does too.
    """
    return magnitude * (2.0 ** -mantissa_bits) * math.sqrt(max(k, 1))


def accumulator_affordable(k: int, atol: float, magnitude: float = 1.0,
                           mantissa_bits: int = FP16_MANTISSA_BITS) -> bool:
    """Does an fp16 accumulator over K terms fit the LOCKED absolute tolerance?

    `atol` is passed in rather than hardcoded: the tolerance belongs to the oracle and
    this file must not be a second place it is written down.
    """
    return accumulator_error(k, magnitude, mantissa_bits) <= atol


def max_affordable_k(atol: float, magnitude: float = 1.0,
                     mantissa_bits: int = FP16_MANTISSA_BITS) -> float:
    """Largest contraction depth an fp16 accumulator can carry inside `atol`."""
    return (atol / (magnitude * 2.0 ** -mantissa_bits)) ** 2


def min_mma_bound_dim(peak_flops: float, bandwidth_bytes_s: float,
                      elem_size: int = 2) -> float:
    """Smallest square d_model == ffn_dim that is compute-bound on this device.

    Intensity is `4*D*F / (D*(elem_size+8))`, which at F == D is linear in D:
    `4*D / (elem_size + 8)`. Setting that equal to the ridge point gives the crossover.
    """
    return ridge_point(peak_flops, bandwidth_bytes_s) * (elem_size + 8) / 4.0


def no_shape_satisfies_both(peak_flops: float, bandwidth_bytes_s: float, atol: float,
                            magnitude: float = 1.0, elem_size: int = 2) -> tuple[bool, float, float]:
    """The scissors, as a computation rather than an assertion.

    Returns `(disjoint, k_ceiling, d_floor)` where `k_ceiling` is the deepest contraction
    an fp16 accumulator can carry inside `atol` and `d_floor` is the narrowest square
    model that is compute-bound. In this architecture K == d_model == ffn_dim, so the two
    regions overlap only if `d_floor <= k_ceiling`.
    """
    k_ceiling = max_affordable_k(atol, magnitude)
    d_floor = min_mma_bound_dim(peak_flops, bandwidth_bytes_s, elem_size)
    return d_floor > k_ceiling, k_ceiling, d_floor


# max_abs of an fp16 accumulator at K=16 (the shallowest MMA sm_89 can issue), N=128,
# unit-magnitude output, measured 2026-08-30. Recorded as data, not as a threshold.
# See docs/findings/30. `tests/bench/test_v25_fp16_accum.py` re-measures it rather than
# trusting this constant.
MEASURED_K16_MAX_ABS = 2.800e-3


def affordable_region_is_empty(atol: float, magnitude: float = 1.0,
                               measured_k16_max_abs: float = MEASURED_K16_MAX_ABS) -> bool:
    """Stronger than the scissors, and the result that actually closes this direction.

    The stochastic model puts the accuracy ceiling at K <= 16.8 for atol=2e-3, and the
    shallowest contraction the hardware's MMA can perform is K=16 -- so the model leaves
    a window exactly ONE legal value wide. Measured at that value, an fp16 accumulator
    gives max_abs 2.80e-3 against the 2.0e-3 budget: the model is optimistic by ~1.4x,
    exactly as `accumulator_error` warns it is, and the last surviving value fails too.

    So the affordable region is not merely disjoint from the fast region. It is EMPTY:
    there is no contraction depth this instruction can execute at which its accumulator
    fits the locked tolerance.
    """
    ceiling = max_affordable_k(atol, magnitude)
    if ceiling < MMA_MIN_K:
        return True                       # the model alone closes it
    return measured_k16_max_abs > atol    # the measurement closes the one value it left


def device_roofline(device=None) -> tuple[float, float]:
    """(peak_flops, bandwidth_bytes_s) from the calibration cache, falling back to
    torch's reported properties so the predicate still evaluates on an uncalibrated card.
    """
    import json
    from pathlib import Path

    cache = Path(__file__).resolve().parents[2] / "ledger" / "device.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text())
            return float(d["peak_bf16_tflops"]) * 1e12, float(
                d["measured_bandwidth_gbs"]) * 1e9
        except (KeyError, ValueError):
            pass
    props = torch.cuda.get_device_properties(device or "cuda")
    # Conservative fallback: memory clock and bus width are exposed, peak FLOPs is not,
    # so decline (a zero peak makes the ridge zero and mma_bound trivially true) is the
    # WRONG direction -- refuse instead by reporting an unreachable ridge.
    return float("inf"), float(getattr(props, "total_memory", 1))
