"""What can honestly be measured on a config whose REFERENCE cannot run.

Config 14 (B=32, S=100000, d=1024, H=16, L=2) is the only row of the announced matrix
that has never produced a measured result. Every sweep records `status: "oom"` and a
truncated traceback, which is the least informative thing the matrix can say about its
most interesting row. This module supplies what a bare `oom` throws away.

THREE DIFFERENT IMPOSSIBILITIES, AND THEY MUST NOT BE CONFLATED
---------------------------------------------------------------
They have different scopes, and the report is only defensible if each is stated with the
scope it actually has.

  1. THE REFERENCE'S ALGORITHM IS INFEASIBLE ON ANY HARDWARE THAT EXISTS.
     `BaselineSelfAttention.forward` materialises `scores = q @ k^T`, shape [B, H, S, S].
     At config 14 that single tensor is 18.63 TiB, and it is not a peak estimate or a
     fragmentation story -- one tensor, one allocation, in the reference's own source.
     Measured: even ONE head of ONE sequence at S=100000 is 37.25 GiB and is refused on
     this 15.99 GiB card. The reference needs 512 of those. This bound is a property of
     the algorithm, not of the machine, and no GPU or node available today changes it.
     `reference_peak_bytes()` derives it; `reference_feasible()` is the predicate.

  2. THE FORWARD SIGNATURE'S OWN FLOOR EXCEEDS THIS CARD, FOR EVERY IMPLEMENTATION.
     `forward(x) -> y` with both `[B, S, d_model]` needs 12.21 GiB in and 12.21 GiB out
     = 24.42 GiB of tensors that no optimisation can remove, because returning a view of
     a mutated input is not an option (it would corrupt the caller's tensor and break on
     the second call -- L25). Against 15.99 GiB of VRAM this is 1.53x over.
     `signature_floor_bytes()` states it. **This one is hardware-specific**: an 80 GiB
     card clears it comfortably, and would still hit impossibility (1).

  3. ON THIS PARTICULAR BOX IT IS REACHED, AND THEN LOST TO FRAGMENTATION.
     WSL2's WDDM driver oversubscribes into host memory. Measured ceiling: 30 GiB of
     2 GiB blocks on a 15.99 GiB card, so 24.42 GiB is nominally reachable. The harness's
     own allocation order then defeats it: `generate_random_case` allocates x (12.21 GiB),
     replaces it with `x * input_scale` (a second 12.21 GiB, freeing the first into the
     allocator's cache), and finally allocates the 3.05 MB `valid_token_mask` -- which
     SPLITS the cached 12.21 GiB segment and pins the whole thing. `empty_cache()` cannot
     release a partly-used segment, so the output tensor must come from a third segment,
     and 36.6 GiB is past the ceiling. A 3 MB mask costs 12.21 GiB.
     This is a fact about one machine on one day and is recorded as such.

WHAT CORRECTNESS CAN MEAN WITH NO BASELINE OUTPUT
--------------------------------------------------
Finding 09 recorded `correctness.passed = null` and checked proxy shapes instead --
same width and depth, survivable sequence lengths. That was right at the time and it is
weaker than what is available, because it never tests a query row that attends over
100,000 keys, which is the only thing about this config that is new.

Two constructions here do test that, and neither needs the reference to fit.

  A. THE CAUSAL-PREFIX THEOREM (`causal_prefix_holds`). Under causal masking with an
     all-valid token mask, every operation in the reference is either position-wise
     (LayerNorm, GELU, the linear projections, the residual adds) or attends only
     backwards. So for any P <= S,

         model(x[:, :P])  ==  model(x)[:, :P]      exactly, in exact arithmetic.

     That makes the UNMODIFIED REFERENCE an oracle at the real shape for the first P
     rows: run the candidate on the full 100,000-token input, run the reference on the
     first P tokens of that same input, compare. No proxy model, no proxy input.
     Measured slack on the reference against itself: 3.88e-4 at P=512, S=4096 with TF32
     on -- about 19% of the 2e-3 budget, spent before the candidate is even involved,
     because a TF32 matmul reduces differently over a different K extent. That cost is
     real and is reported alongside the result rather than hidden in it.
     Its limit is equally real: it says nothing about rows >= P.

  B. THE BLOCKED FP64 ORACLE (`blocked_reference_forward`). The reference's OWN
     arithmetic, evaluated in float64, with the QUERY axis blocked. Blocking queries is
     exact -- softmax is taken along the key axis, so a block of query rows is a closed
     computation -- and it is NOT online/streaming softmax: each query block computes its
     scores against every key it may attend to, subtracts the row max, exponentiates and
     normalises, exactly as `BaselineSelfAttention.forward` does. Deliberately a
     different algorithm from the candidate's flash attention, so a rescaling bug in the
     candidate cannot be mirrored by the oracle.
     Peak is O(q_block * S) instead of O(S^2), so it runs at S=100000, and it verifies
     EVERY row, including the last one.

     The oracle is validated against the reference at sequence lengths where the
     reference runs, and has a negative control, because a check nobody has watched fail
     is not a check (L38, L36).

     MEASURED at config 14's real shape, B=1, S=100000, every row:
     max|candidate - oracle| = 8.0913e-04 in 525 s of fp64, peak 6.37 GiB -- inside the
     1.19e-3 threshold of the certificate below, and three digits from the REFERENCE's
     own 8.086e-04 distance from exact. See docs/findings/40.

WHAT STILL CANNOT BE CLAIMED
-----------------------------
`|candidate - reference|` at S=100000 is not measurable and this module does not pretend
otherwise. What B gives is `|candidate - exact|`, and the triangle inequality then says

    |candidate - reference|  <=  |candidate - exact| + |reference - exact|

with the second term measurable only where the reference runs. Measured there:

    matmul precision "highest" (strict fp32)   1.24e-06 at S=1024, 1.92e-06 at S=4096
    matmul precision "high"    (TF32)          8.086e-04, IDENTICAL at S=128/512/1024/4096

The TF32 figure is flat across a 32x change in sequence length, which is [L4]'s signature
for a representation floor rather than an accumulated error -- TF32 keeps 10 mantissa bits
and these outputs have mean magnitude 0.798. So under the project's own mandated TF32
baseline the REFERENCE spends 40% of the 2e-3 absolute budget before any candidate runs,
and a sufficient condition for passing at S=100000 is

    max |candidate - oracle|  <=  2.0e-3 - 8.09e-4  =  1.19e-3

Flat-in-S is good evidence that 8.09e-04 still holds at S=100000. It is not a measurement
there, it is an extrapolation, and the report says so.

And there is NO SPEEDUP. A speedup is a ratio of two measured times; the denominator
does not exist and cannot be manufactured. Timing our own slower reimplementation of the
baseline and dividing by it would be a number about us, not about the reference.

"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ======================================================================================
# Impossibility 1 -- the reference's algorithm, derived from its source
# ======================================================================================

@dataclass(frozen=True)
class ReferenceRequirement:
    """Bytes `BaselineTransformer.forward` must hold, from reading what it allocates."""

    scores_bytes: int          # ONE [B, H, S, S] tensor. Indisputable lower bound.
    causal_mask_bytes: int     # the [S, S] bool `triu` the reference builds per layer
    activation_bytes: int      # one [B, S, d_model] tensor
    lower_bound_bytes: int     # what MUST be live at once, being generous to the baseline
    realistic_bytes: int       # what it actually peaks at, counting the copies below

    def as_dict(self) -> dict:
        return {"scores_bytes": self.scores_bytes,
                "causal_mask_bytes": self.causal_mask_bytes,
                "activation_bytes": self.activation_bytes,
                "lower_bound_bytes": self.lower_bound_bytes,
                "realistic_bytes": self.realistic_bytes,
                "scores_TiB": self.scores_bytes / 2**40,
                "derivation": "BaselineSelfAttention.forward materialises "
                              "scores=[B,H,S,S]; masked_fill, .float() and softmax each "
                              "produce another tensor of that shape"}


def reference_peak_bytes(batch: int, seq: int, d_model: int, heads: int,
                         dtype_bytes: int = 4) -> ReferenceRequirement:
    """Derived from the reference's source, not estimated from a run.

    The LOWER bound counts one score tensor and nothing else, which is the number no
    reading of the code can argue with: `scores = torch.matmul(q, k.transpose(-2, -1))`
    is a single allocation of [B, H, S, S] and the next line indexes it.

    The realistic figure counts what the source then does to it -- `masked_fill` for the
    causal triangle, `masked_fill` again for the key mask, `.float()`, and `softmax` --
    each of which is out-of-place. Two are live simultaneously at minimum.
    """
    scores = batch * heads * seq * seq * dtype_bytes
    mask = seq * seq                                    # bool
    act = batch * seq * d_model * dtype_bytes
    return ReferenceRequirement(
        scores_bytes=scores,
        causal_mask_bytes=mask,
        activation_bytes=act,
        lower_bound_bytes=scores + act,
        realistic_bytes=2 * scores + mask + 4 * act,
    )


def reference_feasible(batch: int, seq: int, d_model: int, heads: int,
                       dtype_bytes: int, device_bytes: int) -> tuple[bool, str]:
    """Can the reference implementation run this shape on a device of this size?

    Shapes and one measured device property. No config id appears here and none may:
    a predicate that names config 14 is shape detection, which this project's contract
    calls fraud (CLAUDE.md rule 2). This one is evaluable on any card.
    """
    req = reference_peak_bytes(batch, seq, d_model, heads, dtype_bytes)
    if req.lower_bound_bytes <= device_bytes:
        return True, "reference fits"
    return False, (
        f"reference materialises a [{batch},{heads},{seq},{seq}] score tensor = "
        f"{req.scores_bytes / 2**40:.2f} TiB against {device_bytes / 2**30:.2f} GiB of "
        f"device memory; infeasible by {req.lower_bound_bytes / device_bytes:.0f}x")


# ======================================================================================
# Impossibility 2 -- the signature's floor, for every implementation
# ======================================================================================

def signature_floor_bytes(batch: int, seq: int, d_model: int,
                          dtype_bytes: int = 4) -> int:
    """`forward(x) -> y`, both [B, S, d_model]. Two tensors no implementation removes.

    Returning a mutated view of the input would remove one of them and is not available:
    it corrupts the caller's tensor and gives a wrong answer on the second call with the
    same buffer, which is the exact defect L25 catalogued.
    """
    return 2 * batch * seq * d_model * dtype_bytes


# ======================================================================================
# Oracle A -- the causal-prefix theorem
# ======================================================================================

def causal_prefix_holds(causal: bool, valid_token_mask: torch.Tensor | None) -> bool:
    """Is `model(x[:, :P]) == model(x)[:, :P]` available for this call?

    Requires causality (so no query attends forwards) and no invalid tokens inside the
    prefix (a right-padded mask would put the padding boundary inside the truncation).
    Everything else in the reference is position-wise.
    """
    if not causal:
        return False
    if valid_token_mask is None:
        return True
    return bool(valid_token_mask.all())


# ======================================================================================
# Oracle B -- the reference's arithmetic in fp64, blocked over the query axis
# ======================================================================================

def _blocked_attention(q, k, v, scale, causal, q_block, key_mask=None):
    """Softmax attention with the QUERY axis blocked. Exact, and not online softmax.

    q, k, v: [H, S, head_dim] in the oracle dtype. Each query block computes its full
    score row, takes an ordinary max-subtract softmax over the key axis, and multiplies
    by V -- the same three steps, in the same order, as the reference. Blocking the query
    axis changes nothing because softmax is reduced along keys.

    EVERY ITERATION ALLOCATES THE SAME SHAPE, AND THAT IS NOT INCIDENTAL. The obvious
    version truncates the key axis at the causal diagonal (`k[:, :stop]`), which halves
    the arithmetic and makes every iteration a DIFFERENT size. At S=100000 that is ~1500
    distinct allocations of 0.1-1 GB that the caching allocator can never reuse; measured,
    it fails with a driver `CUDA error: out of memory` at S=32768 with 14.18 GiB free.

    So the key axis is full width and fixed, the causal triangle is masked rather than
    skipped, and every step after the matmul is IN PLACE -- one tile is allocated per
    iteration and the allocator hands back the same block every time. It costs the 2x the
    causal truncation would have saved. An oracle that runs at 2x is worth more than one
    that does not run.
    """
    H, S, _ = q.shape
    out = torch.empty_like(q)
    ar = torch.arange(S, device=q.device)
    for start in range(0, S, q_block):
        stop = min(start + q_block, S)
        scores = torch.matmul(q[:, start:stop], k.transpose(-2, -1))
        scores.mul_(scale)
        if causal:
            scores.masked_fill_(ar[None, None, :] > ar[start:stop, None], float("-inf"))
        if key_mask is not None:
            scores.masked_fill_(~key_mask[None, None, :], float("-inf"))
        scores.sub_(scores.amax(dim=-1, keepdim=True))
        scores.exp_()
        scores.div_(scores.sum(dim=-1, keepdim=True))
        out[:, start:stop] = torch.matmul(scores, v)
        del scores
    return out


def choose_q_block(seq: int, heads: int, free_bytes: int,
                   dtype_bytes: int = 8, budget: float = 0.25) -> int:
    """The largest query block whose score tile fits the memory the device reports free.

    `_blocked_attention` batches the head axis, so one tile is [heads, q_block, seq], and
    since every step after the matmul is in place there is one of them live plus its row
    reductions. Getting this wrong is not a slowdown, it is an OOM inside the oracle --
    which is how the first full-S run of this protocol failed, at q_block=1024 with 16
    heads asking the driver for 7.75 GiB.
    """
    per_row = max(1, heads * seq * dtype_bytes * 2)
    return int(max(8, min(1024, (free_bytes * budget) // per_row)))


def blocked_reference_forward(model, x: torch.Tensor,
                              valid_token_mask: torch.Tensor | None = None,
                              causal: bool | None = None,
                              dtype: torch.dtype = torch.float64,
                              q_block: int | None = None) -> torch.Tensor:
    """`BaselineTransformer.forward`'s function, in `dtype`, without an [S,S] tensor.

    Takes a `BaselineTransformer` (its weights, on any device/dtype) and one sequence
    `x` of shape [1, S, d_model]. Returns [1, S, d_model] in `dtype`.

    Every step mirrors the reference source line for line -- pre-norm, attention with
    an out-projection, residual, pre-norm, `ffn_in`, exact GELU, `ffn_out`, residual,
    the per-layer zeroing of invalid tokens, the final norm, the final zeroing. The two
    departures are deliberate and are the point: fp64 instead of fp32, and the query
    blocking above.

    ONE SEQUENCE AT A TIME. Batch elements never interact -- attention is within-sequence
    and the mask is per-sequence -- so a batch is 32 independent calls.
    """
    if x.shape[0] != 1:
        raise ValueError(f"one sequence at a time; got batch {x.shape[0]}")
    if causal is None:
        causal = bool(getattr(model.config, "causal", False))
    dev = x.device
    if q_block is None:
        heads = model.layers[0].attention.num_heads
        free = torch.cuda.mem_get_info(dev)[0] if dev.type == "cuda" else 1 << 30
        q_block = choose_q_block(x.shape[1], heads, free, torch.empty((), dtype=dtype).element_size())
    h = x.to(dtype)
    km = None if valid_token_mask is None else valid_token_mask[0].to(dev)
    zero = km is not None and not bool(km.all())

    def p(t):
        return t.detach().to(device=dev, dtype=dtype)

    for layer in model.layers:
        a = layer.attention
        S = h.shape[1]
        n = F_layer_norm(h, p(layer.norm1.weight), p(layer.norm1.bias), layer.norm1.eps)
        q = torch.nn.functional.linear(n, p(a.q_proj.weight), p(a.q_proj.bias))
        k = torch.nn.functional.linear(n, p(a.k_proj.weight), p(a.k_proj.bias))
        v = torch.nn.functional.linear(n, p(a.v_proj.weight), p(a.v_proj.bias))
        del n
        shape = (S, a.num_heads, a.head_dim)
        q = q.view(shape).transpose(0, 1).contiguous()
        k = k.view(shape).transpose(0, 1).contiguous()
        v = v.view(shape).transpose(0, 1).contiguous()
        ctx = _blocked_attention(q, k, v, a.scale, causal, q_block,
                                 key_mask=km if km is not None and zero else None)
        del q, k, v
        ctx = ctx.transpose(0, 1).contiguous().view(1, S, a.d_model)
        o = torch.nn.functional.linear(ctx, p(a.out_proj.weight), p(a.out_proj.bias))
        del ctx
        if zero:
            o = o.masked_fill(~km[None, :, None], 0)
        h = h + o
        del o

        n2 = F_layer_norm(h, p(layer.norm2.weight), p(layer.norm2.bias), layer.norm2.eps)
        f = torch.nn.functional.linear(n2, p(layer.ffn_in.weight), p(layer.ffn_in.bias))
        del n2
        f = torch.nn.functional.gelu(f, approximate="none")
        h = h + torch.nn.functional.linear(f, p(layer.ffn_out.weight),
                                           p(layer.ffn_out.bias))
        del f
        if zero:
            h = h.masked_fill(~km[None, :, None], 0)

    h = F_layer_norm(h, p(model.final_norm.weight), p(model.final_norm.bias),
                     model.final_norm.eps)
    if zero:
        h = h.masked_fill(~km[None, :, None], 0)
    return h


def F_layer_norm(x, w, b, eps):
    """LayerNorm in x's dtype. Spelled out rather than calling F.layer_norm, which has no
    fp64 fused path on every backend and would silently downcast."""
    mu = x.mean(dim=-1, keepdim=True)
    var = (x - mu).pow(2).mean(dim=-1, keepdim=True)
    return (x - mu) * torch.rsqrt(var + eps) * w + b
