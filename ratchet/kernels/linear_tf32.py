"""TF32 fused linear kernel: C = gelu?(A @ W^T + b), W in nn.Linear [out, in] layout.

The eager baseline runs its projections and FFN as true-fp32 cuBLAS GEMMs (torch defaults
`allow_tf32=False`), which do NOT use the tensor cores. On GB10 that leaves most of the
matmul throughput unused. TF32 keeps a 19-bit float with a 10-bit mantissa -- relative
error ~5e-4 per product -- which stays inside the authoritative fp32 gate (abs<=0.002 OR
rel<=0.02) while running on the tensor cores. This is the E3 speed lever; attention stays
in the flash kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for bm in (64, 128):
        for bn in (64, 128):
            for bk in (32, 64):
                for w in (4, 8):
                    for s in (3, 4):
                        cfgs.append(triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _linear_fwd(
    A, W, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr, GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptr = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptr = W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk  # [N, K] layout

    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = (k0 + offs_k) < K
        a = tl.load(a_ptr, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(w_ptr, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0)
        # tf32x3: three TF32 passes recover near-fp32 accuracy (the single-pass TF32 ~5e-4
        # error compounds over 6 layers to ~0.004, just over the 0.002 gate) while still
        # running on the tensor cores -- far faster than the baseline's non-tensor-core fp32.
        acc += tl.dot(a, tl.trans(w), input_precision="tf32x3")
        a_ptr += BLOCK_K * stride_ak
        w_ptr += BLOCK_K * stride_wk

    if HAS_BIAS:
        acc += tl.load(B + offs_n, mask=offs_n < N, other=0.0)[None, :]
    if GELU:
        # exact (erf) GELU, matching F.gelu(approximate="none")
        acc = acc * 0.5 * (1.0 + tl.erf(acc * 0.7071067811865476))

    c_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr, acc.to(C.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def linear_tf32(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                gelu: bool = False) -> torch.Tensor:
    """x:[..., K] @ weight[N,K]^T + bias -> [..., N]. TF32 tensor-core GEMM."""
    *lead, K = x.shape
    M = 1
    for d in lead:
        M *= d
    N = weight.shape[0]
    x2 = x.reshape(M, K).contiguous()
    w = weight.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    _linear_fwd[grid](
        x2, w, bias if bias is not None else x2, out,
        M, N, K,
        x2.stride(0), x2.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None, GELU=gelu,
    )
    return out.reshape(*lead, N)
