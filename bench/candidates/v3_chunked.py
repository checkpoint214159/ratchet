"""Candidate v3 — v2 plus L2-sized batch chunking.

Generation 3. Parent: v2_fp16_flash.

THE PROBLEM IT SOLVES
---------------------
Two configs hold 93.4% of all baseline time in the matrix: #6 (B=10000, 459 ms) and
#13 (S=1024, 112 ms). Everything else is a rounding error by comparison, so this is
where the remaining wins are.

#6 processes 1.28 M tokens in one shot. The working set is far larger than L2 (48 MB on
this card), so every layer streams the whole activation tensor from HBM and back, and
nothing is ever resident when the next layer wants it. Splitting the batch into chunks
that FIT in L2 turns those HBM round trips into cache hits. Measured: 139.5 -> 79.8 ms,
with peak memory falling from 4.60 GB to 1.90 GB as a side effect.

THE CHUNK SIZE IS CALIBRATED, NOT CHOSEN
----------------------------------------
This is the part that matters beyond this one card. A hardcoded `chunk = 128` is right
here and wrong on the next GPU -- L2 ranges from 40 MB on an A100 to ~192 MB on
Blackwell, a factor of five. So the chunk is solved for from the measured cache size:

    bytes_per_sample ~= seq_len * d_model * 4 * LIVE_TENSORS
    chunk            =  (L2_bytes * TARGET_OCCUPANCY) / bytes_per_sample

On this device that yields 128 for config 6, and the measured optimum was 125 -- the
prediction lands inside the noise. The sensitivity is real and asymmetric: the same
measurement found chunk 32 to be **3.3x worse** than the optimum, so this is not a
parameter to guess at.

`TARGET_OCCUPANCY = 0.5` because the chunk shares L2 with weights, the fp16 cache, and
whatever the driver is holding; aiming at the whole cache evicts the thing you are trying
to keep resident.
"""

from __future__ import annotations

import torch

from .v2_fp16_flash import build as build_v2

# Roughly how many chunk-sized tensors are live at once inside one layer: the input, the
# fused QKV output, and the attention context. Measured attribution says 3 is close; it
# is a divisor for a cache-residency target, not a memory bound, so being off by one is
# a mis-tuned chunk rather than an OOM.
LIVE_TENSORS = 3
TARGET_OCCUPANCY = 0.5


def solve_chunk(batch_size: int, seq_len: int, d_model: int,
                l2_bytes: int, dtype_bytes: int = 4) -> int:
    """Largest batch chunk whose working set still fits the L2 residency target."""
    per_sample = max(1, seq_len * d_model * dtype_bytes * LIVE_TENSORS)
    chunk = int((l2_bytes * TARGET_OCCUPANCY) // per_sample)
    return max(1, min(batch_size, chunk))


def build(baseline_cls):
    v2_cls = build_v2(baseline_cls)

    class CandidateV3(v2_cls):
        """Chunks the batch when doing so buys L2 residency; otherwise it IS v2.

        The fallthrough matters: on 12 of the 14 configs the whole batch already fits the
        residency target, chunking would add loop overhead for nothing, and this class
        must not be slower than its own parent anywhere.
        """

        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)

            b, s, d = x.shape
            if not hasattr(self, "_chunk"):
                props = torch.cuda.get_device_properties(x.device)
                self._chunk = solve_chunk(b, s, d, props.L2_cache_size,
                                          x.element_size())

            if self._chunk >= b:
                return super().forward(x, valid_token_mask)   # unchunked: plain v2

            out = torch.empty_like(x)
            for start in range(0, b, self._chunk):
                stop = min(start + self._chunk, b)
                xs = x[start:stop]
                ms = None if valid_token_mask is None else valid_token_mask[start:stop]
                # Only the full-size chunks reuse the captured graph; a short tail runs
                # eagerly rather than forcing a second capture for one iteration.
                if stop - start == self._chunk:
                    out[start:stop] = super().forward(xs, ms)
                else:
                    out[start:stop] = self._core(xs, ms)
            return out

    return CandidateV3
