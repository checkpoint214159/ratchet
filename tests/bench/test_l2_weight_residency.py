"""Finding 33: L2 persistence for weights is inert, and here is what would change that.

Finding 33 measured that the frontier's weight reads already hit L2 by natural reuse --
94.8% of the theoretical full-miss cost is already saved at config 6 -- so
`cudaAccessPolicyWindow` has nothing to pin. That verdict rests on an arithmetic condition
about weight-set size against L2, not on a lucky measurement, and per [L40] a structural
claim needs an executable check rather than prose. If the announced matrix or the card
changes so that the condition stops holding, these tests fail and finding 33 must be
re-opened by re-running `bench/probes/l2_persistence/`.

WHAT THE PROBE ESTABLISHED, AND WHERE THE BOUNDARY IS
-----------------------------------------------------
The probe varied one thing -- how many distinct weight copies a grid touches -- and located
the eviction boundary directly on this card's 48 MiB L2:

    64 KiB arena   resident   (the real kernel: 2.702 ms, 2.1% over a pure-streaming floor)
     1 MiB arena   resident   (2.706 ms)
    32 MiB arena   EVICTED    (4.728 ms, +75%)

So "small against L2" is the condition, and L2/8 = 6 MiB sits inside the measured gap
between the resident and evicted arms. Two announced configs put their all-layer weight set
above it -- 8 (48.00 MiB, equal to L2 to the byte) and 14 (24.00 MiB) -- and finding 33
probed config 8 directly: a persisting window over the whole arena measured -0.62%, because
config 8's GEMMs run at 93-98.5% of measured tensor-core peak and are compute-bound, not
memory-bound. Any NEW config landing in that bucket has not been probed and must be.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from bench.matrix import MATRIX

REPO = pathlib.Path(__file__).resolve().parents[2]

# Measured on this card by `python3 -m ratchet.oracle.device`, read rather than hardcoded
# so the check follows the hardware (CLAUDE.md rule 2: shapes plus MEASURED properties).
_DEVICE_JSON = REPO / "ledger" / "device.json"

# Configs whose all-layer weight set reaches L2/8 and therefore cannot be dismissed by the
# size argument alone. Finding 33 probed config 8 (-0.62%, compute-bound); config 14 shares
# its width and layer structure and the harness cannot even build its input (finding 09).
PROBED_OR_PRICED = {8, 14}


def _l2_bytes() -> int:
    if not _DEVICE_JSON.exists():
        pytest.skip("no measured device properties available")
    d = json.loads(_DEVICE_JSON.read_text())
    l2 = d.get("l2_cache_size")
    if not l2:
        pytest.skip("device.json carries no l2_cache_size")
    return int(l2)


def weight_bytes(cfg, elem_size: int = 2) -> int:
    """The fp16 weight cache for one config: Q, K, V, O plus the two FFN matrices, every
    layer. Biases and norm parameters are O(d_model) and ignored."""
    per_layer = 4 * cfg.d_model ** 2 + 2 * cfg.d_model * cfg.ffn_dim
    return per_layer * cfg.layers * elem_size


def test_config_6_weight_set_is_far_too_small_to_evict():
    """The premise of finding 33 on the config that is 84% of matrix wall time."""
    l2 = _l2_bytes()
    cfg = next(c for c in MATRIX if c.id == 6)
    w = weight_bytes(cfg)
    assert w / l2 < 0.02, (
        f"config 6's weight set is now {w / 2**20:.2f} MiB, {w / l2 * 100:.1f}% of a "
        f"{l2 / 2**20:.0f} MiB L2. Finding 33 closed L2 persistence on it being ~1.6%; "
        "re-run bench/probes/l2_persistence/probe_weight_traffic.py."
    )


def test_every_config_that_could_evict_its_weights_has_been_probed():
    """The size argument dismisses a config only while its weight set is small against L2.
    Anything that is not must be measured, not argued about."""
    l2 = _l2_bytes()
    big = {c.id for c in MATRIX if weight_bytes(c) >= l2 // 8}
    unprobed = big - PROBED_OR_PRICED
    assert not unprobed, (
        f"configs {sorted(unprobed)} have an all-layer weight set at or above "
        f"{l2 // 8 / 2**20:.1f} MiB against a {l2 / 2**20:.0f} MiB L2, which is inside the "
        "band where finding 33's probe measured eviction. They are outside the arithmetic "
        "that closed A-06. Run bench/probes/l2_persistence/probe_config8.py on them before "
        "citing finding 33 for these shapes."
    )


def test_the_probed_set_is_still_justified():
    """The converse: PROBED_OR_PRICED should not silently accumulate rows that no longer
    need it, which would hide a shrinking margin behind an exemption list."""
    l2 = _l2_bytes()
    big = {c.id for c in MATRIX if weight_bytes(c) >= l2 // 8}
    assert PROBED_OR_PRICED <= big, (
        f"{sorted(PROBED_OR_PRICED - big)} no longer reach L2/8; drop them from the list.")


def test_persistence_shim_reports_a_usable_window():
    """[L38]: finding 33's null is only worth something because the API demonstrably works
    -- its positive control recovered +42.7% on a deliberately thrashing working set. Pin
    that the shim still binds and that this card still advertises a set-aside."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import sys

    sys.path.insert(0, str(REPO / "bench" / "probes" / "l2_persistence"))
    import l2_persist as L

    torch.zeros(1, device="cuda")
    assert L.max_persisting_l2_bytes() > 0
    assert L.max_window_bytes() > 0

    t = torch.zeros(1 << 16, device="cuda", dtype=torch.float16)
    stream = torch.cuda.current_stream().cuda_stream
    try:
        L.set_persisting_set_aside(1 << 20)
        L.set_window(stream, t.data_ptr(), t.numel() * 2, 1.0)
        w = L.get_window(stream)
        assert w["num_bytes"] == t.numel() * 2
        assert w["hitProp"] == L.cudaAccessPropertyPersisting
        assert w["hitRatio"] == pytest.approx(1.0)
    finally:
        L.clear_window(stream)
        L.set_persisting_set_aside(0)
