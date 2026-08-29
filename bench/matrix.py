"""The official TikTok TechJam 2026 shape matrix.

SINGLE SOURCE OF TRUTH. Everything that measures, dispatches, or reports reads the
configs from here so that one edit propagates everywhere. Transcribed from the problem
statement on 2026-08-29.

Read the matrix as an ABLATION GRID rather than a random sample -- it is built to test
whether you dispatch per regime, and one kernel cannot win all of it:

    rows 1-6    sweep batch size 1 -> 10000 with everything else fixed  (occupancy axis)
    rows 7-8    sweep model dim 32 / 1024                               (arithmetic-intensity axis)
    rows 9-11   sweep head count 1 / 2 / 16 at fixed dim                (head_dim axis)
    rows 12-13  sweep sequence length 32 / 1024                         (attention-cost axis)
    row 14      the extreme: seq 100000                                 (feasibility)

Two facts that differ from the reference benchmark's own defaults and change the
optimization calculus completely:

  * EVERY config is causal. The reference defaults to causal=False. Causal means roughly
    half the attention score matrix is structurally zero, and skipping it in a fused
    kernel is EXACT (masked entries carry exactly zero softmax weight), not an
    approximation. Free work avoided, on all 14.

  * ffn_dim == d_model on every row. The reference defaults to a 4x expansion
    (512 -> 2048). So the feed-forward stage is ~4x less dominant here than profiling
    the reference's defaults would suggest, which pushes attention UP in relative cost.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


# --------------------------------------------------------------------------------------
# OPEN QUESTION -- confirm with the organizers before trusting any per-config plan.
#
# The problem statement's column is headed "QKV Dim". We read it as d_model (the model
# width), so head_dim = d_model / heads. The alternative reading is that it names
# head_dim directly, which would make d_model = heads * QKV_Dim and grow rows 9-11 by up
# to 16x.
#
# We take the d_model reading because the FFN Dim column carries the same values, and an
# FFN hidden size is conventionally expressed in model dims, not per-head dims. Under our
# reading head_dim spans 8..256.
#
# CORRECTED 2026-08-30 (finding 23). This comment previously claimed that "cuDNN and
# FlashAttention typically support {32, 64, 128, 256} and may silently fall back to a slow
# path" at head_dim=8, and that claim steered the project's sense of where the prize was
# for a week. It is FALSE. Measured directly, every backend accepts head_dim=8:
#
#     head_dim     flash   mem_eff     cudnn      math
#            8        ok        ok        ok        ok
#           16        ok        ok        ok        ok
#           32        ok        ok        ok        ok
#           64        ok        ok        ok        ok
#          128        ok        ok        ok        ok
#          256        ok        ok   REFUSES        ok
#
# The refusal is at head_dim=256, on the OPPOSITE end of the range -- which is config 8
# (d_model 1024, 4 heads), not configs 7 and 11.
#
# head_dim=8 is still the most interesting region, for a better-founded reason: sm_89's
# tensor-core instruction is m16n8k16, so `tl.dot` requires K>=16 and a 128x128 score
# matrix at head_dim=8 leaves the vendor kernel's tiling mismatched to the hardware.
# Padding D to 16 INSIDE a kernel is free; padding it in HBM is exact but measured
# 1.2-2.7x SLOWER, so that variant is closed without spending a generation on it.
# --------------------------------------------------------------------------------------
DIM_COLUMN_READING = "d_model"


@dataclass(frozen=True)
class Config:
    """One row of the announced matrix."""

    id: int
    batch_size: int
    d_model: int          # the "QKV Dim" column; see DIM_COLUMN_READING
    heads: int
    seq_len: int
    layers: int
    ffn_dim: int
    causal: bool = True   # every announced row is causal

    @property
    def head_dim(self) -> int:
        return self.d_model // self.heads

    @property
    def tokens(self) -> int:
        return self.batch_size * self.seq_len

    def activation_bytes(self, dtype_bytes: int = 4) -> int:
        """One [B, S, D] activation tensor. The stack holds several of these at once."""
        return self.tokens * self.d_model * dtype_bytes

    def dense_scores_bytes(self, dtype_bytes: int = 4) -> int:
        """The B*H*S*S score matrix, IF materialized.

        This is the number that decides whether a shape is flash-attention-mandatory
        rather than flash-attention-preferred: when it exceeds device memory, the
        reference implementation cannot run at all and a fused kernel is the only path.
        """
        return self.batch_size * self.heads * self.seq_len * self.seq_len * dtype_bytes

    def attention_flops(self) -> int:
        """Score + context matmuls, halved for causal. 2 FLOP per MAC."""
        dense = 2 * 2 * self.batch_size * self.heads * self.seq_len * self.seq_len * self.head_dim
        return dense // 2 if self.causal else dense

    def projection_flops(self) -> int:
        """Q, K, V, output projections plus the two FFN matmuls, all layers."""
        per_layer = 2 * self.tokens * self.d_model * (4 * self.d_model + 2 * self.ffn_dim)
        return per_layer * self.layers

    def total_flops(self) -> int:
        return self.projection_flops() + self.attention_flops() * self.layers

    def key(self) -> str:
        return (f"cfg{self.id:02d}_B{self.batch_size}_D{self.d_model}_H{self.heads}"
                f"_S{self.seq_len}_L{self.layers}_F{self.ffn_dim}"
                f"_{'causal' if self.causal else 'full'}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["head_dim"] = self.head_dim
        return d

    def cli_args(self) -> list[str]:
        """Arguments for the authoritative benchmark's CLI, so a config never has to be
        re-typed by hand into a command line."""
        return [
            "--batch-size", str(self.batch_size),
            "--d-model", str(self.d_model),
            "--heads", str(self.heads),
            "--seq-len", str(self.seq_len),
            "--layers", str(self.layers),
            "--ffn-dim", str(self.ffn_dim),
        ] + (["--causal"] if self.causal else [])


MATRIX: tuple[Config, ...] = (
    Config(id=1,  batch_size=64,    d_model=128,  heads=4,  seq_len=128,    layers=4, ffn_dim=128),
    Config(id=2,  batch_size=1,     d_model=128,  heads=4,  seq_len=128,    layers=4, ffn_dim=128),
    Config(id=3,  batch_size=4,     d_model=128,  heads=4,  seq_len=128,    layers=4, ffn_dim=128),
    Config(id=4,  batch_size=16,    d_model=128,  heads=4,  seq_len=128,    layers=4, ffn_dim=128),
    Config(id=5,  batch_size=128,   d_model=128,  heads=4,  seq_len=128,    layers=4, ffn_dim=128),
    Config(id=6,  batch_size=10000, d_model=128,  heads=4,  seq_len=128,    layers=4, ffn_dim=128),
    Config(id=7,  batch_size=64,    d_model=32,   heads=4,  seq_len=128,    layers=4, ffn_dim=32),
    Config(id=8,  batch_size=64,    d_model=1024, heads=4,  seq_len=128,    layers=4, ffn_dim=1024),
    Config(id=9,  batch_size=64,    d_model=128,  heads=1,  seq_len=128,    layers=4, ffn_dim=128),
    Config(id=10, batch_size=64,    d_model=128,  heads=2,  seq_len=128,    layers=4, ffn_dim=128),
    Config(id=11, batch_size=64,    d_model=128,  heads=16, seq_len=128,    layers=4, ffn_dim=128),
    Config(id=12, batch_size=64,    d_model=128,  heads=4,  seq_len=32,     layers=4, ffn_dim=128),
    Config(id=13, batch_size=64,    d_model=128,  heads=4,  seq_len=1024,   layers=4, ffn_dim=128),
    Config(id=14, batch_size=32,    d_model=1024, heads=16, seq_len=100000, layers=2, ffn_dim=1024),
)

BY_ID: dict[int, Config] = {c.id: c for c in MATRIX}


# --------------------------------------------------------------------------------------
# Regimes. These are LABELS for reporting, deliberately not dispatch predicates -- a
# dispatch predicate must be a function of measured device properties (see the historical
# specs/04-dispatch.md), never a hardcoded config id. Grouping here is for humans reading
# the report and for weighting the objective.
# --------------------------------------------------------------------------------------
REGIMES: dict[str, tuple[int, ...]] = {
    "launch_bound":   (2, 3, 4, 12),    # tiny total work; kernel launches dominate
    "mainstream":     (1, 5, 9, 10),    # the middle of the grid
    "awkward_headdim": (7, 11),         # head_dim == 8; vendor fast paths may refuse
    "wide_model":     (8,),             # d_model 1024, head_dim 256
    "throughput":     (6,),             # 1.28M tokens
    "long_context":   (13,),            # seq 1024, attention starts to dominate
    "extreme":        (14,),            # seq 100000; feasibility, not speed
}


def regime_of(config_id: int) -> str:
    for name, ids in REGIMES.items():
        if config_id in ids:
            return name
    return "unclassified"


def weighted_score(speedups: dict[int, float], cap: float = 3.0) -> float:
    """Aggregate a per-config speedup map into one number.

    Equal weight per config, because the problem statement gives no weighting and every
    row is a test case. Speedups are CLIPPED so one spectacular regime cannot carry a
    submission that is mediocre everywhere else -- the failure mode where a system posts
    a huge win on an operator worth 0.12% of real runtime.

    A config with no entry scores 1.0 (no better than baseline), NOT skipped: skipping
    would reward not measuring.

    WHICH SPEEDUPS TO PASS IN. The cap only discriminates if the inputs straddle it.
    Against the EAGER baseline they do not: by generation 12, 17 of 18 config speedups
    exceeded 3.0, every candidate from v3 onward scored 2.79-2.82, and the ranking
    inverted -- the best candidate placed fourth. Pass speedups measured against the
    COMPILED baseline (see docs/findings/12), where the frontier spans 1.4x-10.9x and the
    cap bites on 4 of 13 configs instead of 17 of 18.
    """
    if not speedups:
        return 0.0
    total = sum(min(speedups.get(c.id, 1.0), cap) for c in MATRIX)
    return total / len(MATRIX)


if __name__ == "__main__":
    print(f"{'#':>3} {'regime':<16} {'B':>6} {'D':>5} {'H':>3} {'hd':>4} {'S':>7} "
          f"{'L':>2} {'tokens':>10} {'GFLOP':>9} {'scores@fp32':>12}")
    for c in MATRIX:
        print(f"{c.id:>3} {regime_of(c.id):<16} {c.batch_size:>6} {c.d_model:>5} "
              f"{c.heads:>3} {c.head_dim:>4} {c.seq_len:>7} {c.layers:>2} "
              f"{c.tokens:>10,} {c.total_flops()/1e9:>9.1f} "
              f"{c.dense_scores_bytes()/1e9:>10.2f}GB")
