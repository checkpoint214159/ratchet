"""Per-config kernel decomposition: prove the win is a kernel improvement, not a dtype trick.

For each announced config, isolate the attention call and measure my flash kernel vs the
baseline's explicit attention at MATCHED precision (fp32) -- so the ratio is purely the
kernel/algorithm (streaming online-softmax + exact causal-skip), with no dtype advantage --
then again at fp16 to show the additional tensor-core factor. Grounds the end-to-end
numbers to the problem statement's "implement a GPU kernel for the layer".
"""
import sys

import torch
import triton

from ratchet.kernels.dispatch import MATRIX
from ratchet.kernels.flash_attention import flash_attention


def baseline_attn(q, k, v, causal, D):
    s = (q @ k.transpose(-2, -1)) * (D ** -0.5)
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, device=q.device, dtype=torch.bool).triu(1),
                          float("-inf"))
    return torch.softmax(s.float(), -1).to(q.dtype) @ v


def bench(fn):
    try:
        return triton.testing.do_bench(fn, warmup=25, rep=100)
    except Exception:
        return None


ONLY = set(int(a) for a in sys.argv[1:] if a.isdigit())
print("cfg | shape[B,H,N,D] | flash-vs-baseline fp32 (pure kernel) | fp16 | flash_fp16-vs-baseline_fp32")
for cfg in MATRIX:
    if cfg.id == 14 or (ONLY and cfg.id not in ONLY):
        continue
    B, H, N, D = cfg.batch_size, cfg.heads, cfg.seq_len, cfg.head_dim
    try:
        q32 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        q16 = q32.to(torch.float16)
        tb32 = bench(lambda: baseline_attn(q32, q32, q32, True, D))
        tf32 = bench(lambda: flash_attention(q32, q32, q32, causal=True))
        tb16 = bench(lambda: baseline_attn(q16, q16, q16, True, D))
        tf16 = bench(lambda: flash_attention(q16, q16, q16, causal=True))
        pk = tb32 / tf32 if tb32 and tf32 else float("nan")
        f16 = tb16 / tf16 if tb16 and tf16 else float("nan")
        cross = tb32 / tf16 if tb32 and tf16 else float("nan")
        print(f"cfg{cfg.id:<2} [{B},{H},{N},{D}] | {pk:.2f}x | {f16:.2f}x | {cross:.2f}x")
    except Exception as e:
        print(f"cfg{cfg.id}: FAILED {type(e).__name__}: {str(e)[:100]}")
