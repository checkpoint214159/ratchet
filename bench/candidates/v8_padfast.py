"""Candidate v8 -- take the fast flash path even when the input is padded.

Generation 8. Parent: v6_fp16_gelu. Branch: cand/g8/right-pad-redundant-mask.

THE BLIND SPOT THIS FIXES
-------------------------
Every measurement in the ledger before 2026-08-29 used `padding_ratio=0.0`. That is the
ONLY value at which v2-v7 take their fast path: they elide the all-True mask so
FlashAttention qualifies, and otherwise fall back to fp32 SDPA with an explicit
`attn_mask` -- which is precisely the v1 defect (finding 04) still living in the padded
branch.

Measured cost of that fallback at `padding_ratio=0.5` on v6:

    config 1    3.68x -> 1.88x     (51% of the unpadded speedup)
    config 5    3.69x -> 1.87x     (51%)
    config 13  24.06x -> 6.62x     (28%)

THE ARGUMENT: FOR RIGHT-PADDED CAUSAL ATTENTION THE KEY MASK IS REDUNDANT
------------------------------------------------------------------------
The reference (`BaselineSelfAttention.forward`) applies three things: a causal mask
`triu(diagonal=1)`, a key mask on invalid positions, and a final zeroing of invalid
output rows. The benchmark's `generate_random_case` builds the mask as
`positions < lengths[:, None]` -- a contiguous VALID PREFIX, padding only on the right.

Under those two facts together:

  * For a VALID query at position i (i < length), causal already restricts attention to
    keys j <= i. Since i < length, every such j is also < length, so every key it can see
    is valid. **The key mask removes nothing.**
  * For an INVALID query (i >= length), the output row is zeroed afterward regardless, so
    whatever it attended to is discarded. It also cannot produce NaN: causal admits keys
    j <= i, and keys 0..length-1 are valid with length >= 1, so at least one key survives
    the softmax.

Therefore the key mask changes no surviving output element, and dropping it lets q/k/v
stay fp16 with no `attn_mask` -- the exact conditions FlashAttention requires.

WHY THIS IS GUARDED RATHER THAN ASSUMED
---------------------------------------
The argument holds only for a contiguous right-padded mask. An arbitrary mask with holes,
or left-padding, would break it: a valid query could then look back at an invalid key.
`_prefix_padded()` verifies the mask really is `arange < lengths` at prime time and the
candidate falls back to the correct slow path if it is not. The optimization is claimed
only where its precondition is checked.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .v6_fp16_gelu import build as build_v6


def prefix_padded(mask: torch.Tensor) -> bool:
    """True iff every row of `mask` is a contiguous run of True followed by False.

    Equivalent to `mask == (arange(S) < lengths[:, None])` for some per-row length. A row
    with a hole (True, False, True) or left-padding fails, and must, because the
    redundancy argument depends on the valid set being a prefix.
    """
    if mask is None:
        return True
    lengths = mask.sum(dim=-1, keepdim=True)
    positions = torch.arange(mask.shape[-1], device=mask.device)[None, :]
    return bool(torch.equal(mask, positions < lengths))


def build(baseline_cls):
    v6_cls = build_v6(baseline_cls)

    class CandidateV8(v6_cls):
        def _prime(self, mask):
            super()._prime(mask)
            # Two ways to reach the fast path: no mask at all (v6's condition), or a
            # right-padded mask under causal attention (the new one). Checked once.
            self._fastpath = self._nomask or (self.config.causal and prefix_padded(mask))
            self._needs_zeroing = not self._nomask

        def _core(self, x, mask):
            if not self._fastpath:
                return super()._core(x, mask)        # unproven mask shape: v6's slow path

            zero = self._needs_zeroing
            for layer, cached in zip(self.layers, self._cache):
                a = layer.attention
                qkv_w, qkv_b, out_w, out_b, in_w, in_b, ffn_w, ffn_b = cached
                b, s, _ = x.shape

                qkv = F.linear(layer.norm1(x).to(torch.float16), qkv_w, qkv_b)
                q, k, v = qkv.split(a.d_model, dim=-1)
                q = q.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                k = k.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)
                v = v.view(b, s, a.num_heads, a.head_dim).transpose(1, 2)

                # fp16, no attn_mask: flash qualifies. The key mask is provably redundant
                # here -- see the module docstring.
                ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)

                o = F.linear(ctx.transpose(1, 2).reshape(b, s, a.d_model),
                             out_w, out_b).float()
                if zero:
                    o = o.masked_fill(~mask[..., None], 0)
                x = x + o

                h = F.linear(layer.norm2(x).to(torch.float16), in_w, in_b)
                x = x + F.linear(F.gelu(h, approximate="none"), ffn_w, ffn_b).float()
                if zero:
                    x = x.masked_fill(~mask[..., None], 0)

            x = self.final_norm(x)
            return x.masked_fill(~mask[..., None], 0) if zero else x

    return CandidateV8
