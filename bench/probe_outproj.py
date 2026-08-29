"""The g24 out-projection probe: what layout does SDPA actually return, and does fusing
the projection with the residual add pay?

TWO QUESTIONS, AND THE FIRST ONE KILLED THE PROPOSAL'S HEADLINE
---------------------------------------------------------------
Proposal D-02 asked for a GEMM whose PROLOGUE does a head-major gather, on the premise
that `ctx.transpose(1, 2).reshape(B, S, D)` forces a copy of the attention context. Part
1 of this probe asks that question directly and answers NO: on this card SDPA returns a
`[B, S, H, hd]`-contiguous buffer wearing a `[B, H, S, hd]` view, so the transpose is
already contiguous and the reshape is a free view. There is no gather to absorb.

Part 2 measures what is left -- fusing the fp32 widening and the fp32 residual add into
the GEMM's epilogue, so the fp16 `[M, D]` temporary between the two kernels never exists
-- and picks the tiling.

THIS IS A PROBE, NOT A MEASUREMENT (L41)
----------------------------------------
Every number here is op-level and isolated, which is the exact shape of the three wrong
numbers in L41 and L33. It compares against a `torch.compile`d two-kernel path rather
than an eager one, because eager is not what the candidate replaces -- but that still
does not make it the harness. **A probe may propose; it may never conclude.** The number
that decides anything comes from `bench/run_matrix.py`.

    python3 bench/probe_outproj.py [--layout-only]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bench.gpu_lock import gpu_lock                                      # noqa: E402
from bench.kernels.outproj_resid import fits, outproj_resid, tiling_for  # noqa: E402
from bench.matrix import MATRIX                                          # noqa: E402


def sdpa_ctx(bsz, seq, heads, hd, seed=0, scale=0.2):
    torch.manual_seed(seed)
    d = heads * hd
    qkv = torch.randn(bsz, seq, 3 * d, device="cuda", dtype=torch.float16) * scale
    q, k, v = qkv.split(d, dim=-1)
    q = q.view(bsz, seq, heads, hd).transpose(1, 2)
    k = k.view(bsz, seq, heads, hd).transpose(1, 2)
    v = v.view(bsz, seq, heads, hd).transpose(1, 2)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def two_kernel(ctx, w, b, res, m, d):
    """Exactly what v18's `_core` does at this point in the layer."""
    return res + F.linear(ctx.transpose(1, 2).reshape(m, d), w, b).float()


# Shapes taken from the announced matrix, plus two small ones for the sub-saturation
# branch of the tiling rule. Config 14 is excluded: its inputs cannot be built here.
def probe_shapes():
    seen, out = set(), []
    for c in MATRIX:
        if c.id == 14:
            continue
        bsz = min(c.batch_size, 10000)
        key = (bsz, c.seq_len, c.heads, c.head_dim)
        if key not in seen:
            seen.add(key)
            out.append((f"cfg{c.id}", *key))
    return out


def part1_layout():
    print("PART 1 -- what layout does F.scaled_dot_product_attention return?\n")
    print(f"{'shape':>8} {'D':>5} {'stride':>28} {'head-major':>11} {'transpose':>12}")
    all_token_major = True
    for name, bsz, seq, heads, hd in probe_shapes():
        ctx = sdpa_ctx(min(bsz, 64), seq, heads, hd)
        hm = ctx.stride() == (heads * seq * hd, seq * hd, hd, 1)
        tm = ctx.transpose(1, 2).is_contiguous()
        all_token_major &= tm
        print(f"{name:>8} {heads*hd:>5} {str(ctx.stride()):>28} "
              f"{str(hm):>11} {('contiguous' if tm else 'STRIDED'):>12}")
        del ctx
        torch.cuda.empty_cache()
    print()
    if all_token_major:
        print("VERDICT: token-major everywhere. `.transpose(1, 2).reshape(...)` is a FREE")
        print("VIEW, not a copy. D-02's head-major gather does not exist to be absorbed.")
    else:
        print("VERDICT: at least one shape is genuinely head-major -- the general gather")
        print("path in outproj_resid.py is load-bearing there, not merely defensive.")
    return all_token_major


def part2_fusion():
    from triton.testing import do_bench

    props = torch.cuda.get_device_properties("cuda")
    smem, sms = props.shared_memory_per_block_optin, props.multi_processor_count
    print(f"\nPART 2 -- the epilogue fusion. {sms} SMs, {smem} B opt-in smem.")
    print("Isolated op-level numbers against the torch.compile'd two-kernel path.\n")
    print(f"{'shape':>8} {'D':>5} {'tokens':>9} {'tile':>20} "
          f"{'compiled':>9} {'fused':>9} {'gain':>8} {'agree':>10}")
    for name, bsz, seq, heads, hd in probe_shapes():
        d = heads * hd
        m = bsz * seq
        tile = tiling_for(m, d, sms)
        if not fits(d, heads, 2, tile[0], tile[1], tile[2], smem):
            print(f"{name:>8} {d:>5} {m:>9} {str(tile):>20}  DECLINED by fits()")
            continue
        ctx = sdpa_ctx(bsz, seq, heads, hd)
        torch.manual_seed(1)
        w = torch.randn(d, d, device="cuda", dtype=torch.float16) * 0.05
        b = torch.randn(d, device="cuda", dtype=torch.float16) * 0.05
        res = torch.randn(m, d, device="cuda", dtype=torch.float32)
        wt = w.t().contiguous()

        comp = torch.compile(two_kernel, dynamic=False)
        for _ in range(3):
            comp(ctx, w, b, res, m, d)
        # min-of-N: clocks are not lockable under WSL (CLAUDE.md, hardware truth).
        base = min(do_bench(lambda: comp(ctx, w, b, res, m, d), warmup=25, rep=100)
                   for _ in range(3))
        fused = min(do_bench(lambda: outproj_resid(ctx, res, wt, b, *tile),
                             warmup=25, rep=100) for _ in range(3))
        agree = float((outproj_resid(ctx, res, wt, b, *tile)
                       - comp(ctx, w, b, res, m, d)).abs().max())
        print(f"{name:>8} {d:>5} {m:>9} {str(tile):>20} {base:>9.4f} {fused:>9.4f} "
              f"{base/fused:>7.3f}x {agree:>10.2e}", flush=True)
        del ctx, w, b, res, wt, comp
        torch._dynamo.reset()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-only", action="store_true",
                    help="part 1 only; costs no measurable GPU time and needs no lock")
    args = ap.parse_args()

    if args.layout_only:
        part1_layout()
    else:
        with gpu_lock("g24 out-projection probe", timeout_s=3600):
            part1_layout()
            part2_fusion()
