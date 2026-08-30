"""Falsifiers for v31's out-projection epilogue. INDICATIVE ONLY (L41).

A probe may propose; it may never conclude. Nothing this prints belongs in the ledger, and
where it disagrees with `bench/run_matrix.py` the harness is right until proven otherwise
(L9, L41 -- three recurrences and counting).

    python3 bench/probe_outproj_epilogue.py --layout-only    # part 1, no GPU lock needed
    python3 bench/probe_outproj_epilogue.py --accuracy       # parts 1-3, no timing
    python3 bench/probe_outproj_epilogue.py --time           # adds part 4, takes the lock

Four parts, in decreasing order of how much they can be trusted:

1. LAYOUT. Prints `ctx.stride()` and `ctx.is_contiguous()` for the tensor v23's kernel
   produces. Finding 30 established that SDPA's `ctx` is token-major contiguous despite
   wearing a head-major view, so the gather `g24` was commissioned to absorb does not
   exist. This asks the same question of OUR kernel's output. A stride is observable in
   one line; an argument about a layout is not evidence.
2. TRAFFIC. Pure arithmetic, no GPU. What the fusion removes, per token per layer.
3. ACCURACY. The segment against an fp64 reference, fused against split. This is the one
   claim the sweep cannot see and the reason the candidate is worth keeping even if the
   speed is a wash (L26, L39).
4. TIMING. The segment only, and BIASED unless the comparison is built carefully: the
   split path's `.float()` and residual add are fused by Inductor into a neighbouring
   kernel in the real candidate, so timing them as separate eager ops invents a win that
   was never available (L33/L41 -- exactly how v19's op probe read 3.84x on a candidate
   the harness measured flat). The baseline arm here is therefore `torch.compile`d.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

from bench.matrix import MATRIX                                       # noqa: E402
from bench.kernels.attn_outproj import (applies, attn_outproj,        # noqa: E402
                                        autotune_tile, register_bytes)
from bench.kernels.attn_single_tile import single_tile_attention      # noqa: E402


# ---------------------------------------------------------------- 1. the layout premise

def layout_report() -> None:
    """Does v23's kernel output need a head-major gather? (Finding 30, asked of our code.)"""
    print("\n1. LAYOUT -- what single_tile_attention actually returns\n")
    print(f"{'shape':<26} {'ctx.shape':<20} {'ctx.stride()':<22} contiguous  "
          f"needs a gather?")
    for c in MATRIX:
        props = torch.cuda.get_device_properties("cuda")
        from bench.kernels.attn_single_tile import applies as v23_applies
        ok, _ = v23_applies(c.seq_len, c.head_dim, props)
        if not ok:
            continue
        b = min(c.batch_size, 4)
        qkv = torch.randn(b, c.seq_len, 3 * c.d_model, device="cuda",
                          dtype=torch.float16)
        from bench.kernels.attn_single_tile import choose_tile
        bm, w, st = choose_tile(c.seq_len, c.head_dim,
                                props.regs_per_multiprocessor,
                                props.max_threads_per_multi_processor, props.warp_size)
        ctx = single_tile_attention(qkv, c.heads, c.head_dim, c.head_dim ** -0.5,
                                    bm, w, st)
        contig = ctx.is_contiguous()
        print(f"cfg{c.id:<3} D{c.d_model:<5} H{c.heads:<3} hd{c.head_dim:<4} "
              f"{str(tuple(ctx.shape)):<20} {str(ctx.stride()):<22} {str(contig):<11} "
              f"{'NO' if contig else 'yes'}")
        del qkv, ctx
    print("\n   v23 allocates torch.empty((B, S, d_model)) and writes head h at column")
    print("   offset h*head_dim, so `ctx` is token-major contiguous BY CONSTRUCTION --")
    print("   no view, no transpose, no stride to inspect. There is no gather to absorb")
    print("   and g24's CONTIG constexpr has no analogue here. The win is")
    print("   MATERIALIZATION, not layout -- the same correction g24 had to make.")


# ---------------------------------------------------------------------- 2. the traffic

def traffic_report() -> None:
    print("\n2. TRAFFIC -- bytes per token per layer, attention epilogue only\n")
    print("   v23  write ctx 2D | read ctx 2D + write o 2D | read o 2D + read x 4D"
          " + write y 4D  = 16D, 3 launches")
    print("   v31                                          |            read x 4D"
          " + write y 4D  =  8D, 1 launch")
    print("   g24  (against SDPA, for comparison)                                  "
          "           = 14D -> 10D, 2 launches -> 1\n")
    print(f"{'shape':<30} {'v23 B/token':>12} {'v31 B/token':>12} {'saved':>8} "
          f"{'reg bytes/program':>18}")
    for c in MATRIX:
        props = torch.cuda.get_device_properties("cuda")
        ok, _ = applies(c.seq_len, c.head_dim, c.heads, c.batch_size, props)
        if not ok:
            continue
        from bench.kernels.attn_outproj import choose_tile
        bm, _w, _s = choose_tile(c.seq_len, c.head_dim, c.d_model, c.heads,
                                 c.batch_size, props.regs_per_multiprocessor,
                                 props.max_threads_per_multi_processor,
                                 props.multi_processor_count, props.warp_size)
        d = c.d_model
        print(f"cfg{c.id:<3} B{c.batch_size:<6} D{d:<5} H{c.heads:<3} hd{c.head_dim:<4} "
              f"{16*d:>12,} {8*d:>12,} {'50%':>8} "
              f"{register_bytes(c.seq_len, c.head_dim, d, bm):>18,}")


# --------------------------------------------------------------------- 3. the accuracy

def _fp64_segment(qkv, res, w_t, bias, heads, hd):
    b, s, _ = qkv.shape
    dm = heads * hd
    q, k, v = qkv.double().split(dm, dim=-1)
    q = q.view(b, s, heads, hd).transpose(1, 2)
    k = k.view(b, s, heads, hd).transpose(1, 2)
    v = v.view(b, s, heads, hd).transpose(1, 2)
    sc = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
    causal = torch.ones(s, s, device=qkv.device, dtype=torch.bool).triu(1)
    ctx = (torch.softmax(sc.masked_fill(causal, float("-inf")), -1) @ v)
    ctx = ctx.transpose(1, 2).reshape(b * s, dm)
    return res.double() + (ctx @ w_t.double() + bias.double())


def accuracy_report() -> None:
    """Two fp64 references, and the gap between them is the point.

    WHOLE SEGMENT computes the fp64 answer from `qkv`, so the fp16 rounding of `ctx` --
    common to both arms, and unavoidable, since tensor cores take fp16 operands -- is
    inside the comparison. EPILOGUE ONLY takes the fp16 `ctx` as given and scores only the
    projection, which is what `g24` measured when it reported ~600x tighter.

    Both are honest about different things and only the first answers "is this candidate
    more accurate". Reporting the second alone would credit the fusion with removing an
    error term that dominates and that it does not touch (L33's shape, in accuracy rather
    than in speed: an isolated measurement measures the isolation).
    """
    print("\n3. ACCURACY -- the segment against an fp64 reference\n")
    print("   The fp16 rounding of `ctx` is common to both paths (tensor-core operands).")
    print("   What the fusion DELETES is the fp16 rounding of the projection OUTPUT,")
    print("   which F.linear performs before .float() widens it again.\n")
    print(f"{'shape':<30} {'split max_abs':>14} {'fused max_abs':>14} {'tighter':>9}"
          f" | {'epilogue-only tighter':>22}")
    props = torch.cuda.get_device_properties("cuda")
    torch.manual_seed(0)
    for c in MATRIX:
        ok, _ = applies(c.seq_len, c.head_dim, c.heads, c.batch_size, props)
        if not ok:
            continue
        b = min(c.batch_size, 64)
        dm, hd, h, s = c.d_model, c.head_dim, c.heads, c.seq_len
        qkv = torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16)
        w_t = (torch.randn(dm, dm, device="cuda", dtype=torch.float16) * dm ** -0.5)
        bias = torch.randn(dm, device="cuda", dtype=torch.float16) * 0.1
        res = torch.randn(b * s, dm, device="cuda", dtype=torch.float32)
        tile, _ = autotune_tile(s, hd, h, b)

        fused = attn_outproj(qkv, res, w_t, bias, None, h, hd, hd ** -0.5, *tile)

        from bench.kernels.attn_single_tile import choose_tile as v23_tile
        bm, w, st = v23_tile(s, hd, props.regs_per_multiprocessor,
                             props.max_threads_per_multi_processor, props.warp_size)
        ctx = single_tile_attention(qkv, h, hd, hd ** -0.5, bm, w, st)
        split = res + F.linear(ctx.view(b * s, dm), w_t.t().contiguous(), bias).float()

        want = _fp64_segment(qkv, res, w_t, bias, h, hd)
        a = (split.double() - want).abs().max().item()
        f = (fused.double() - want).abs().max().item()

        # g24's reference: the fp16 context taken as given, so only the projection's own
        # rounding is scored. Same two arms, a reference that excludes the attention.
        epi = (res.double() + (ctx.view(b * s, dm).double() @ w_t.double()
                               + bias.double()))
        ea = (split.double() - epi).abs().max().item()
        ef = (fused.double() - epi).abs().max().item()
        print(f"cfg{c.id:<3} B{b:<6} D{dm:<5} H{h:<3} hd{hd:<4} "
              f"{a:>14.3e} {f:>14.3e} {a/f:>8.1f}x | {ea/max(ef, 1e-30):>21.0f}x")
        del qkv, w_t, bias, res, fused, ctx, split, want, epi


# ----------------------------------------------------------------------- 4. the timing

def timing_report(lock_timeout_s: float = 2700.0) -> None:
    """INDICATIVE ONLY. The baseline arm is compiled so the widen and the add are fused,
    which is what the real candidate's split path gets from Inductor (L33/L41).

    Blocks for the lock rather than failing on it: several agents share one GPU and
    finding 26 / L38 is that a number taken alongside another benchmark is not a number.
    """
    import triton.testing as tt
    from bench.gpu_lock import gpu_lock

    print("\n4. TIMING -- the segment, INDICATIVE ONLY, never a verdict\n")
    props = torch.cuda.get_device_properties("cuda")
    with gpu_lock("probe_outproj_epilogue", timeout_s=lock_timeout_s):
        print(f"{'shape':<30} {'split (compiled)':>18} {'fused':>10} {'gain':>7}")
        for c in MATRIX:
            ok, _ = applies(c.seq_len, c.head_dim, c.heads, c.batch_size, props)
            if not ok:
                continue
            b = min(c.batch_size, 256)
            dm, hd, h, s = c.d_model, c.head_dim, c.heads, c.seq_len
            qkv = torch.randn(b, s, 3 * dm, device="cuda", dtype=torch.float16)
            w_t = (torch.randn(dm, dm, device="cuda", dtype=torch.float16) * dm ** -0.5)
            w = w_t.t().contiguous()
            bias = torch.randn(dm, device="cuda", dtype=torch.float16) * 0.1
            res = torch.randn(b * s, dm, device="cuda", dtype=torch.float32)
            tile, _ = autotune_tile(s, hd, h, b)

            from bench.kernels.attn_single_tile import choose_tile as v23_tile
            bm, nw, st = v23_tile(s, hd, props.regs_per_multiprocessor,
                                  props.max_threads_per_multi_processor, props.warp_size)

            def split():
                ctx = single_tile_attention(qkv, h, hd, hd ** -0.5, bm, nw, st)
                return res + F.linear(ctx.view(b * s, dm), w, bias).float()

            torch._dynamo.reset()
            csplit = torch.compile(split, dynamic=False)
            csplit()

            def fused():
                return attn_outproj(qkv, res, w_t, bias, None, h, hd, hd ** -0.5, *tile)

            fused()
            a = min(tt.do_bench(csplit, warmup=25, rep=50, return_mode="min")
                    for _ in range(3))
            f = min(tt.do_bench(fused, warmup=25, rep=50, return_mode="min")
                    for _ in range(3))
            print(f"cfg{c.id:<3} B{b:<6} D{dm:<5} H{h:<3} hd{hd:<4} "
                  f"{a:>18.4f} {f:>10.4f} {a/f:>6.3f}x")
            del qkv, w_t, w, bias, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-only", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--time", action="store_true")
    args = ap.parse_args()

    layout_report()
    if args.layout_only:
        raise SystemExit(0)
    traffic_report()
    accuracy_report()
    if args.time:
        timing_report()
