> **HISTORICAL — NVIDIA/Triton competition scaffold. Not current, not an instruction.**
> Ratchet is now hardware-agnostic and hardware-gated (see `docs/hardware-support.md`).
> Single-GPU details below (e.g. `sm_120`, `wgmma`, A100/H100 tile budgets) are retained
> as historical design material, never as a general finding or a fixed target. See
> `docs/handoff-readme.md`.

# Spec 04 — Self-calibrating dispatch

**Zone B.** The most important design decision in the project, and the answer to the
objection every judge will raise: *everyone benchmarked on a different GPU, so how do I
compare you?*

## The principle

The naive version of shape dispatch is a chain of `if N < 256` constants. Those constants
are wrong on the next GPU, and demonstrably so: an attention shape with arithmetic
intensity 256 sits **above** an A100's ridge point of 153 (compute bound) and **below** an
H100's 295 (memory bound). Same shape, same code, opposite regime, opposite correct
response.

So: **every predicate is a function of the M1 calibration**, and no branch boundary is a
literal.

```python
def choose(shape: Shape, dev: DeviceProfile) -> Impl:
    ...
```

`dev` comes from `oracle/device.py` and carries measured bandwidth, measured launch
overhead, SM count, shared-memory budget and the derived ridge point.

## The four branches

Evaluate in order; first match wins.

### 1. Launch-bound

```python
est_work_ns = estimate_work_ns(shape, dev)          # from FLOPs and bytes vs ceilings
if dev.launch_overhead_ns * KERNELS_PER_LAYER > est_work_ns:
```

Kernel quality is invisible here; kernel *count* is everything. At B=1, N=128, H=8, d=64
the whole attention computation is ~33.6 MFLOP moving 512 KiB — about 0.3 µs of work
against 1–5 µs per launch and roughly 7–10 launches in a naive layer.

**Response:** fuse as far as the shared-memory budget allows, capture in a CUDA graph, and
materialize S in shared memory with a plain two-pass softmax. At N=128, d=64, S is 32 KB
against a 99–227 KB budget, so the online rescaling machinery is pure overhead and extra
numerical error — it exists to avoid an HBM round trip that is not happening.

### 2. Occupancy-bound

```python
ctas = shape.B * shape.H * ceil(shape.N / block_m)
if ctas < OCCUPANCY_FACTOR * dev.sm_count:          # start OCCUPANCY_FACTOR = 2
```

The grid is too small to fill the GPU. Note this gets *worse* on bigger GPUs: 8 CTAs is
7.4% of an A100's 108 SMs and 6.1% of an H100's 132.

**Response:** split-K over the KV axis (Flash-Decoding). Each CTA emits a partial
`(O, m, ℓ)`; a cheap second kernel merges them with the same rescaling identity.

Port the heuristic rather than inventing one. Two references, both worth reading:
- MSLK `FwOp.get_split_k(B, G, H, Mk, Mq, page_size, is_paged)` — *choose splits so total
  CTAs ≈ a parallelism target, then halve until each split's chunk is large enough to
  amortize*, with `if Mq > 1 and B*G*H > 64: return 1` as the early exit.
- vLLM's `seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv` with
  `MIN_LAUNCH_GRID_SIZE_2D = 128` — the same predicate, derived from the launch grid.

The reduction math is in `flashinfer/triton/kernels/cascade.py`; do not rewrite it.

### 3. Bandwidth-bound

```python
if arithmetic_intensity(shape) < dev.ridge_point:
```

At peak for your intensity, so tuning buys nothing. **Response:** raise intensity. Bigger
tiles, threadblock swizzling for L2 reuse, and fusing the neighbouring norms and epilogues
so the bytes move once. Note a 128×128 CTA tile has AI ≈ 64, below every modern ridge
point — which is why L2 rasterization is not optional on datacenter parts.

### 4. Compute-bound

Otherwise. **Response:** deepest pipeline that fits `dev.smem_per_block_optin`, widest MMA
the architecture exposes. And check honestly whether you are beating cuBLAS or cuDNN
before claiming the branch is worth having — of 24 operators in one study, only 1 of 9
vendor-backed ones was beaten.

## Tile-size selection is also calibrated

Shared memory is the binding constraint and it differs by a factor of 2.3 across the
plausible hardware:

```
SMEM = BLOCK_M · d · 2  +  stages · (2 · BLOCK_N · d · 2)      bytes, bf16
```

At d=128 with `BLOCK_M=BLOCK_N=128` and 3 stages that is 224 KB: fits an H100's 227 KB,
does **not** fit an A100's 163 KB, and is nowhere near a consumer part's 99 KB. Solve for
the largest feasible `(BLOCK_M, BLOCK_N, stages)` given `dev.smem_per_block_optin` rather
than hardcoding a table.

Record the cost: halving `BLOCK_N` doubles the online-softmax rescale chain
(`rescales = N / BLOCK_N`), which adds non-matmul work *and* accumulated numerical error.

## Returning the choice

Follow ROCm/aiter's signature, which is better than it looks:

```python
def choose(shape, dev) -> tuple[Impl, bool]:   # (implementation, is_tuned)
```

`is_tuned=False` means this shape fell through to a default and nobody has measured it.
Callers can log it, the report can list it, and you never silently present an untuned
path as a tuned one.

## Vendor axis

The same mechanism, one more dimension. Branch on `dev.backend` (`cuda` / `hip`) and
capability family, exactly as vLLM does via `current_platform`. The IBM result is the
justification: the same Triton source went from 19.7% of FA-3 naive to 98.6–105.9% tuned,
and the AMD-specific work was **branches in the config-selection tree, not a second
codebase**. Even if we never run on AMD, structuring for it costs almost nothing and is a
strong Feasibility claim.

## Acceptance

1. Artificially halve `sm_count` and `measured_bandwidth` in `ledger/device.json`; the
   dispatch decisions change **in the direction the roofline predicts**. A dispatch that
   does not respond to device properties is a hardcoded table wearing a costume.
2. Every branch boundary traces to a calibration field. Grep for numeric literals in
   `dispatch/` and justify each survivor in a comment.
3. `is_tuned=False` is returned for at least one shape and appears in the report.
4. For every regime, the report states which implementation won and by how much, including
   the regimes where a vendor path won.
