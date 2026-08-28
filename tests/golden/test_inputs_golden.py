"""Golden pins on input generation.

The correctness gate is only as strong as the inputs it runs on. Every test here exists
because a specific weakening is profitable for a search loop: dropping the negated
distribution lets a ReLU-style identity kernel pass, dropping the NaN/Inf cases hides
the single most common defect in machine-generated kernels, and perturbing the seed
derivation silently changes what "passing" means while every diff still looks green.
So this file pins the distribution roster, the placement of the adversarial values,
and -- for one canonical case -- the exact bits.
"""

import hashlib
from dataclasses import replace

import torch

from ratchet.oracle import CORRECTNESS_SHAPES, DISTRIBUTIONS, Shape, correctness_suite, generate

# The full roster, in gate order. Hardcoded rather than imported: if someone removes a
# distribution from inputs.py, importing the (shrunk) list would hide the removal.
ALL_NINE = ("standard", "scaled_up", "scaled_down", "negated",
            "denormal", "near_overflow", "noncontiguous", "with_nan", "with_inf")

_MHA = CORRECTNESS_SHAPES[0]                      # B2 N127 H4 D64, bfloat16
_GQA = Shape(B=1, N=513, H=8, D=128, H_kv=2)      # kv heads != q heads


def _bits(t: torch.Tensor) -> bytes:
    return t.detach().flatten().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def test_distribution_roster_is_complete():
    """correctness_suite must produce all nine distributions, in this order.

    Order matters because check_all returns on the first failing distribution; a
    reordering changes which diagnostic the ledger records.
    """
    assert tuple(DISTRIBUTIONS) == ALL_NINE[:4]
    suite = correctness_suite(_MHA, seed=4242, include_adversarial=True)
    assert tuple(suite.keys()) == ALL_NINE
    lite = correctness_suite(_MHA, seed=4242, include_adversarial=False)
    assert tuple(lite.keys()) == ALL_NINE[:4]


def test_generate_is_bitwise_deterministic():
    """Two calls with the same (shape, seed, distribution) must agree bit for bit.

    Reproducibility is the whole point of an explicit per-case Generator: a rerun of a
    failing case must be the same case, or diagnostics are noise.
    """
    for shape in (_MHA, _GQA):
        for dist in ALL_NINE:
            first = generate(shape, seed=4242, distribution=dist)
            second = generate(shape, seed=4242, distribution=dist)
            for a, b in zip(first, second):
                # torch.equal is False when NaNs are present, so compare raw bits.
                assert _bits(a) == _bits(b), (
                    f"generate() is not deterministic for {shape.key()} / {dist}"
                )


def test_generate_bits_are_pinned():
    """One canonical case is pinned to its exact SHA-256.

    The in-process double-call test above cannot catch a changed seed derivation or a
    swapped RNG -- those are deterministic too, just deterministic about different
    bits. This digest can. It is expected to change only if the toolchain changes
    (torch's Philox implementation); if it fires on an unchanged toolchain, someone
    edited inputs.py.
    """
    h = hashlib.sha256()
    for t in generate(_MHA, seed=4242, distribution="standard"):
        h.update(_bits(t))
    assert h.hexdigest() == (
        "23a743f0b5645e23cc10d1dbbd143147559d0d20798356d6cdde477ec6366582"
    ), "the golden input bits changed: inputs.py was edited or the toolchain moved"


def test_every_distribution_has_correct_shape_and_dtype():
    """All nine distributions, on MHA/GQA/float32 cases: layout (B, H, N, D), k and v
    carry H_kv heads, dtype follows the Shape."""
    for shape in (_MHA, _GQA, replace(_MHA, dtype="float32")):
        want = {"bfloat16": torch.bfloat16, "float32": torch.float32}[shape.dtype]
        for dist in ALL_NINE:
            q, k, v = generate(shape, seed=4242, distribution=dist)
            assert q.shape == (shape.B, shape.H, shape.N, shape.D), (shape.key(), dist)
            assert k.shape == (shape.B, shape.H_kv, shape.N, shape.D), (shape.key(), dist)
            assert v.shape == (shape.B, shape.H_kv, shape.N, shape.D), (shape.key(), dist)
            for t in (q, k, v):
                assert t.dtype == want, (shape.key(), dist)
                assert t.device.type == "cuda", (shape.key(), dist)


def test_negated_actually_contains_negatives():
    """The distribution that catches "return the input unchanged" must really negate.

    Pinned as exactly -standard at the same seed, not merely "has some negatives" --
    N(0,1) has negatives anyway, so anything weaker would not detect a no-op edit.
    """
    q_std, k_std, v_std = generate(_MHA, seed=4242, distribution="standard")
    q_neg, k_neg, v_neg = generate(_MHA, seed=4242, distribution="negated")
    assert bool((q_neg < 0).any()) and bool((k_neg < 0).any()) and bool((v_neg < 0).any())
    for neg, std in ((q_neg, q_std), (k_neg, k_std), (v_neg, v_std)):
        assert torch.equal(neg, -std), "negated is not the exact negation of standard"


def test_with_nan_places_exactly_one_nan_in_q():
    q, k, v = generate(_MHA, seed=4242, distribution="with_nan")
    assert bool(torch.isnan(q[0, 0, 0, 0])), "the sentinel NaN is not at q[0,0,0,0]"
    assert int(torch.isnan(q).sum()) == 1, "there must be exactly one NaN, in q"
    assert bool(torch.isfinite(k).all()) and bool(torch.isfinite(v).all())


def test_with_inf_places_exactly_one_inf_in_k():
    q, k, v = generate(_MHA, seed=4242, distribution="with_inf")
    assert bool(torch.isinf(k[0, 0, 0, 0])), "the sentinel Inf is not at k[0,0,0,0]"
    assert int(torch.isinf(k).sum()) == 1, "there must be exactly one Inf, in k"
    assert bool(torch.isfinite(q).all()) and bool(torch.isfinite(v).all())


def test_noncontiguous_q_and_k_are_noncontiguous():
    """Kernels that assume contiguity must be made to fail: real callers pass views."""
    q, k, v = generate(_MHA, seed=4242, distribution="noncontiguous")
    assert not q.is_contiguous(), "noncontiguous q is contiguous -- the case is defanged"
    assert not k.is_contiguous(), "noncontiguous k is contiguous -- the case is defanged"
    # Same values as standard: only the memory layout is adversarial.
    q_std, k_std, _ = generate(_MHA, seed=4242, distribution="standard")
    assert torch.equal(q, q_std) and torch.equal(k, k_std)


def test_denormal_magnitudes_are_tiny():
    """Values scaled by finfo.tiny probe gradual-underflow handling; anything that
    flushes to zero or normalizes silently shows up against the reference."""
    q, k, v = generate(_MHA, seed=4242, distribution="denormal")
    for t in (q, k, v):
        f32 = t.to(torch.float32)  # bf16 -> f32 widening is exact
        assert bool(torch.isfinite(f32).all())
        assert float(f32.abs().max()) < 1e-36, "denormal inputs are not tiny"
        assert bool((f32 != 0).any()), "denormal inputs all flushed to zero"
