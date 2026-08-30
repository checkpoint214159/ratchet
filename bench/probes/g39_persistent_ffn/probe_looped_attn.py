"""F-02's pre-registered falsifier: a LOOPED, smem-staged attention kernel on the shapes
where `attn_single_tile` declines.

F-02 argues that the loop-free single-tile kernel cannot reach head_dim >= 128 because its
register working set is `block_m*BN*4` (scores) + `(block_m + 2*BN)*pad16(head_dim)*2`
(operands), and the operand term does not shrink as `block_m` narrows -- so `choose_tile`
walks 64 -> 32 -> 16 and returns None, and configs 9 (hd 128) and 8 (hd 256) fall back to
the vendor. Meanwhile 99 KB of opt-in shared memory goes untouched. Putting the K/V axis
back into a loop moves the operands into smem (Triton stages `tl.dot` operands there for
free via `num_stages`), leaving registers holding only Q and the running acc/m/l.

F-02's KILL CONDITION, quoted: "if the best looped tile does not reach 1.4x on config 9's
shape, stop."

F-02's PRE-REGISTERED DISCRIMINATOR, quoted: "if the looped form wins on config 9
(grid 64 -> 256) but not on config 8 (grid already 256), the cause is occupancy and the
predicate should be `grid_ctas < multi_processor_count`, not `head_dim`. If it wins on
both, the cause is the register working set. If it wins on neither, hand-written attention
above head_dim 32 is closed on this card."

The baseline is what the frontier ACTUALLY runs at those shapes, not a strawman [L33/f29]:
`F.scaled_dot_product_attention` on the three transposed views PLUS the
`transpose(1,2).reshape` repack, because that repack is a real copy the fused kernel also
deletes -- finding 31 says a material part of v23's win is not about attention at all.
Both arms are timed from the same fused `[B, S, 3*d_model]` buffer.

n_regs / n_spills / smem are reported for every arm that compiles.

INDICATIVE ONLY [L41]: this is an op-level falsifier, not a candidate measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as Fn
import triton
import triton.language as tl

from bench.kernels.attn_single_tile import (applies, next_pow2,
                                            padded_head_dim, single_tile_attention)


@triton.jit
def _attn_looped(
    QKV, OUT,
    stride_qkv_b, stride_qkv_s,
    stride_o_b, stride_o_s,
    scale,
    S: tl.constexpr, DH: tl.constexpr, DP: tl.constexpr, DM: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    """Flash-style causal attention with a K/V loop, reading the fused QKV buffer.

    Registers hold Q [BM, DP], acc [BM, DP] fp32 and the running m/l vectors. K and V
    tiles are [BN, DP] and are staged through shared memory by Triton's pipeliner, which
    is the whole point: the operand term leaves the register file.

    Output layout is head-major `[B, S, heads*DH]`, byte-identical to
    `sdpa(...).transpose(1, 2).reshape(B, S, DM)`, exactly as `_attn_single_tile` is.
    """
    m_block = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)

    rm = m_block * BM + tl.arange(0, BM)
    rd = tl.arange(0, DP)
    keep_m = rm < S
    keep_d = rd < DH

    head = QKV + b * stride_qkv_b + h * DH
    q = tl.load(head + rm[:, None] * stride_qkv_s + rd[None, :],
                mask=keep_m[:, None] & keep_d[None, :], other=0.0)

    m_i = tl.full([BM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, DP], dtype=tl.float32)

    # Causal: this query block never needs keys beyond its own last row. That is the
    # exact same work-skip the single-tile kernel gets from its `tl.where`, but here it
    # removes whole K tiles from the loop instead of masking them, so it is FREE work
    # avoided rather than work done and discarded.
    kv_end = tl.minimum((m_block + 1) * BM, S)

    for start_n in range(0, kv_end, BN):
        rn = start_n + tl.arange(0, BN)
        keep_n = rn < S
        k = tl.load(head + DM + rn[:, None] * stride_qkv_s + rd[None, :],
                    mask=keep_n[:, None] & keep_d[None, :], other=0.0)
        v = tl.load(head + 2 * DM + rn[:, None] * stride_qkv_s + rd[None, :],
                    mask=keep_n[:, None] & keep_d[None, :], other=0.0)

        s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        valid = (rn[None, :] <= rm[:, None]) & keep_n[None, :]
        s = tl.where(valid, s, float("-inf"))

        # Online softmax. With one K tile this reduces to the textbook softmax, which is
        # why the single-tile kernel could delete it; with several it is what makes the
        # multi-tile reduction equal the single-tile one.
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(OUT + b * stride_o_b + rm[:, None] * stride_o_s + h * DH + rd[None, :],
             acc.to(OUT.dtype.element_ty),
             mask=keep_m[:, None] & keep_d[None, :])


def looped_attention(qkv, heads, head_dim, scale, block_m, block_n, num_warps,
                     num_stages=2):
    b, s, three_dm = qkv.shape
    dm = heads * head_dim
    out = torch.empty((b, s, dm), device=qkv.device, dtype=qkv.dtype)
    h = _attn_looped[(triton.cdiv(s, block_m), heads, b)](
        qkv, out, qkv.stride(0), qkv.stride(1), out.stride(0), out.stride(1), scale,
        S=s, DH=head_dim, DP=padded_head_dim(head_dim), DM=dm,
        BM=block_m, BN=block_n, num_warps=num_warps, num_stages=num_stages,
    )
    return out, h


def sdpa_repack(qkv, heads, head_dim):
    """What the frontier runs where `attn_single_tile` declines, repack included."""
    b, s, _ = qkv.shape
    dm = heads * head_dim
    q, k, v = qkv.split(dm, dim=2)
    q = q.view(b, s, heads, head_dim).transpose(1, 2)
    k = k.view(b, s, heads, head_dim).transpose(1, 2)
    v = v.view(b, s, heads, head_dim).transpose(1, 2)
    o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True)
    return o.transpose(1, 2).reshape(b, s, dm)


def _bench(fn, reps=3):
    return min(triton.testing.do_bench(fn, warmup=25, rep=50) for _ in range(reps))


SHAPES = (
    #  label                              B   H   hd    S
    ("cfg 9  (H=1, hd=128)",             64,  1, 128, 128),
    ("cfg 10 (H=2, hd=64)",              64,  2,  64, 128),
    ("cfg 8  (H=4, hd=256)",             64,  4, 256, 128),
    ("cfg 13 (H=4, hd=32, S=1024)",      64,  4,  32, 1024),
)


def main():
    from bench.gpu_lock import gpu_lock

    props = torch.cuda.get_device_properties(0)
    SM = props.multi_processor_count
    dev = torch.device("cuda")

    with gpu_lock("g39 looped-attention falsifier", timeout_s=7200):
        print(f"device: {props.name}  SMs={SM}  "
              f"regs/SM={props.regs_per_multiprocessor}  "
              f"smem/block optin={props.shared_memory_per_block_optin}\n")

        for label, B, H, hd, S in SHAPES:
            dm = H * hd
            g = torch.Generator(device="cuda").manual_seed(11)
            qkv = (torch.randn(B, S, 3 * dm, device=dev, dtype=torch.float16,
                               generator=g) * 0.3)
            scale = hd ** -0.5

            ok_single, why = applies(S, hd, props)
            ref = sdpa_repack(qkv, H, hd)
            torch.cuda.synchronize()
            t_base = _bench(lambda: sdpa_repack(qkv, H, hd), reps=5)
            grid_flash = B * H * triton.cdiv(S, 128)

            print(f"{'='*92}\n{label}   B={B} H={H} hd={hd} S={S}   "
                  f"flash grid ~= {grid_flash} CTAs on {SM} SMs\n{'='*92}")
            print(f"  single_tile applies? {ok_single}  ({why})")
            print(f"  SDPA + repack (the real baseline)      {t_base*1e3:8.3f} us")

            # The single-tile arm is only informative where it APPLIES; where it does
            # not, the frontier runs SDPA and that is the arm to beat.
            if ok_single:
                for bm_try in (16, 32, 64, 128):
                    if bm_try > S:
                        continue
                    try:
                        o = single_tile_attention(qkv, H, hd, scale, bm_try, 4)
                    except Exception:
                        continue
                    if torch.allclose(o.float(), ref.float(), atol=2e-3, rtol=2e-2):
                        t = _bench(lambda bm_try=bm_try: single_tile_attention(
                            qkv, H, hd, scale, bm_try, 4), reps=3)
                        print(f"  single_tile BM={bm_try:<4}                    "
                              f"{t*1e3:8.3f} us   {t_base/t:.3f}x")

            best = None
            rows = []
            for bm in (16, 32, 64, 128):
                if bm > S:
                    continue
                for bn in (16, 32, 64, 128):
                    if bn > S:
                        continue
                    for w in (2, 4, 8):
                        for st in (1, 2, 3, 4):
                            try:
                                o, h = looped_attention(qkv, H, hd, scale, bm, bn, w, st)
                                torch.cuda.synchronize()
                            except Exception:
                                continue
                            good = torch.allclose(o.float(), ref.float(),
                                                  atol=2e-3, rtol=2e-2)
                            if not good:
                                continue
                            t = _bench(lambda bm=bm, bn=bn, w=w, st=st:
                                       looped_attention(qkv, H, hd, scale, bm, bn,
                                                        w, st)[0], reps=3)
                            rows.append((t, bm, bn, w, st, h.n_regs, h.n_spills,
                                         h.metadata.shared))
            rows.sort()
            if not rows:
                print("  looped: NO ARM COMPILED AND MATCHED\n")
                continue
            print(f"  looped, top 5 of {len(rows)} matching arms:")
            for t, bm, bn, w, st, nr, ns, sm in rows[:5]:
                print(f"    BM={bm:<4} BN={bn:<4} warps={w} stages={st}  "
                      f"{t*1e3:8.3f} us  {t_base/t:6.3f}x   "
                      f"n_regs={nr:<4} n_spills={ns:<4} smem={sm}")
            t, bm, bn, w, st, nr, ns, sm = rows[0]
            grid = B * H * triton.cdiv(S, bm)
            verdict = "PASSES 1.4x" if t_base / t >= 1.4 else "FAILS the 1.4x kill bar"
            print(f"\n  BEST {t_base/t:.3f}x  (grid {grid} CTAs)   -> {verdict}\n")


if __name__ == "__main__":
    main()
