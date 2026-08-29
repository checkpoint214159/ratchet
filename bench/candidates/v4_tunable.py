"""Candidate v4 — v3 with its knobs exposed, so the search loop can turn them.

Generation 4. Parent: v3_chunked.

Behaviourally identical to v3 at the default settings. The only difference is that the
three constants v3 hardcodes are read from the environment, which is what lets
`bench/loop.py` evaluate a point without rewriting source between evaluations.

Reading them from the environment rather than a constructor argument is deliberate: the
custody benchmark constructs the model itself, through a fixed signature we may not
change (`UserOptimizedTransformer(config)`), so there is nowhere to thread a parameter
through. The env var is the seam that exists.

    RATCHET_USE_GRAPH          1 | 0
    RATCHET_TARGET_OCCUPANCY   fraction of L2 the chunk should aim to occupy
    RATCHET_LIVE_TENSORS       chunk-sized tensors assumed live inside one layer

Every one of these is a genuine device-dependent choice, not a magic number: the right
L2 occupancy differs between a 48 MB consumer cache and a 192 MB datacenter one, and the
live-tensor count differs with how much of a layer a compiler has fused.
"""

from __future__ import annotations

import os

import torch

from .v2_fp16_flash import build as build_v2


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def solve_chunk(batch_size: int, seq_len: int, d_model: int, l2_bytes: int,
                dtype_bytes: int, chunk_ratio: float) -> int:
    """Chunk sized so its working set is `chunk_ratio` of L2.

    ONE parameter, not two. The earlier (target_occupancy, live_tensors) pair only ever
    appeared as a quotient, so the search space was degenerate and the optimizer could
    "improve" by re-measuring its own starting point in new coordinates.
    """
    per_sample = max(1, seq_len * d_model * dtype_bytes)
    chunk = int((l2_bytes * chunk_ratio) // per_sample)
    return max(1, min(batch_size, chunk))


def build(baseline_cls):
    v2_cls = build_v2(baseline_cls)

    class CandidateV4(v2_cls):
        def forward(self, x, valid_token_mask=None):
            if not hasattr(self, "_cache"):
                self._prime(valid_token_mask)

            if not hasattr(self, "_chunk"):
                self.use_graph = _env_int("RATCHET_USE_GRAPH", 1) == 1
                b, s, d = x.shape
                props = torch.cuda.get_device_properties(x.device)
                self._chunk = solve_chunk(
                    b, s, d, props.L2_cache_size, x.element_size(),
                    _env_float("RATCHET_CHUNK_RATIO", 0.1667),
                )

            b = x.shape[0]
            if self._chunk >= b:
                return super().forward(x, valid_token_mask)

            out = torch.empty_like(x)
            for start in range(0, b, self._chunk):
                stop = min(start + self._chunk, b)
                xs = x[start:stop]
                ms = None if valid_token_mask is None else valid_token_mask[start:stop]
                if stop - start == self._chunk:
                    out[start:stop] = super().forward(xs, ms)
                else:
                    out[start:stop] = self._core(xs, ms)
            return out

    return CandidateV4
