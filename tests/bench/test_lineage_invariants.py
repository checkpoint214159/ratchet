"""Invariants every candidate must satisfy, applied across the WHOLE lineage.

Written after finding 17, where v12 could replay an empty CUDA graph and return a stale
buffer. That bug was invisible to the entire accuracy suite, because every accuracy check
ran one input per trial against a reference computed for that same input -- and a stale
buffer holding a correct PREVIOUS answer matches the reference for the previous input.

These tests are deliberately crude. They assert properties that need no tolerance
reasoning and no reference implementation:

  * a function of its input must not return the same thing for different inputs;
  * it must return the same thing for the same input;
  * the tensor it returns must not change underneath the caller on the next call.

Every candidate that caches, captures, or otherwise holds mutable state across calls is a
candidate for exactly this class of bug, so the whole registry is swept rather than the
one that happened to fail.
"""
import pytest
import torch


def _bench():
    import importlib.util, sys
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "benchmarks/reference/torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("ref_inv", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ref_inv"] = m
    spec.loader.exec_module(m)
    return m


def _candidate_names():
    from bench.candidates import REGISTRY
    # Two documented exclusions, both kept in the registry as lineage rather than as
    # shippable candidates:
    #   v5  -- known-incorrect stepping stone (finding 08), fails the tolerance itself
    #   v12 -- CAN return a stale buffer when CUDA-graph capture silently yields an empty
    #          graph (finding 17). It is superseded by v13, which verifies the capture
    #          and falls back. v12 stays measurable for the lineage; it is not a
    #          candidate for submission, and this test documents why.
    #   v9a/v9b/v11/v15 -- all four let Inductor own CUDA-graph capture (use_graph=False,
    #          mode= reduce-overhead or max-autotune) and return the compiled callable's
    #          result DIRECTLY. Under graph replay that is a static buffer, so the caller's
    #          tensor is rewritten by the next forward. Measured and recorded before the
    #          defect was known; their ledger rows were taken WITHOUT the clone that fixes
    #          it, so they are left as measured rather than silently re-defined. v13
    #          (clones) and v16 (clones) are the safe members of this lineage.
    #          See docs/findings/24.
    known_unsafe = {"v5_fp16_resid", "v12_graph_over_compile",
                    "v9a_compiled_core", "v9b_reduce_overhead",
                    "v11_lean", "v15_lifted_veto"}
    return sorted(k for k in REGISTRY if k not in known_unsafe)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name", _candidate_names())
class TestLineageInvariants:
    @staticmethod
    def _make(name):
        # Dynamo's cache_size_limit (8) is shared per PROCESS. Without this reset, every
        # candidate after the eighth silently falls back to EAGER -- which returns fresh
        # tensors and therefore PASSES the static-buffer test vacuously. That is exactly
        # how this suite reported 113 green while four candidates carried a live
        # silent-wrong-answer bug (finding 24). A test that passes because the thing it
        # tests was never compiled is worse than no test.
        torch._dynamo.reset()

        ref = _bench()
        from bench.candidates import REGISTRY
        cfg = ref.TransformerConfig(batch_size=4, seq_len=64, d_model=64, num_heads=4,
                                    ffn_dim=64, num_layers=2, causal=True)
        torch.manual_seed(0)
        base = ref.BaselineTransformer(cfg)
        cand = REGISTRY[name].build(ref.BaselineTransformer)(cfg)
        ref.copy_model_weights(base, cand)
        return ref, cfg, cand.to("cuda", torch.float32).eval()

    def test_different_inputs_give_different_outputs(self, name):
        """The staleness check. A cached or captured buffer that is never rewritten
        returns the same values regardless of what it was asked."""
        ref, cfg, cand = self._make(name)
        x1, m = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 11, 0.0, 1.0)
        x2, _ = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 22, 0.0, 1.0)
        with torch.inference_mode():
            y1 = cand(x1, m).clone()
            y2 = cand(x2, m).clone()
        assert not torch.allclose(y1, y2), (
            f"{name}: identical output for two different inputs -- stale state")

    def test_same_input_is_reproducible(self, name):
        ref, cfg, cand = self._make(name)
        x, m = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 33, 0.0, 1.0)
        with torch.inference_mode():
            a = cand(x, m).clone()
            cand(torch.randn_like(x), m)          # perturb any internal state
            b = cand(x, m).clone()
        assert torch.allclose(a, b, rtol=1e-3, atol=1e-4), (
            f"{name}: same input gave different answers across calls")

    def test_returned_tensor_survives_the_next_call(self, name):
        """A candidate that returns its own static buffer instead of a clone will have
        the caller's tensor rewritten underneath it on the next forward."""
        ref, cfg, cand = self._make(name)
        x1, m = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 44, 0.0, 1.0)
        x2, _ = ref.generate_random_case(cfg, torch.device("cuda"), torch.float32, 55, 0.0, 1.0)
        with torch.inference_mode():
            held = cand(x1, m)
            snapshot = held.clone()
            cand(x2, m)                            # must not disturb `held`
        assert torch.equal(held, snapshot), (
            f"{name}: the tensor handed to the caller was mutated by a later call")
