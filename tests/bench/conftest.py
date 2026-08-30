"""Per-test Dynamo reset.

WHY THIS EXISTS
---------------
Dynamo's `cache_size_limit` is 8 and is shared per PROCESS. Once exhausted,
`torch.compile` SILENTLY falls back to eager -- no error, no warning in the test output,
just a candidate whose kernels were never generated.

That is not hypothetical: finding 24 records four candidates that carried a live
silent-wrong-answer bug while the suite reported 113 green, because by the time they ran
the budget was gone and eager's fresh output tensors satisfied the static-buffer
assertion vacuously. The suite was green BECAUSE of a second defect.

With 34 registered candidates the budget is now exhausted within the first few tests, and
every mechanism assertion after that -- "a triton_tem kernel appears", "the kernel count
fell" -- becomes a test of eager mode. v34's two mechanism tests passed alone and failed
in the full suite for exactly this reason.

Resetting per test costs a recompile and buys the guarantee that each test measures the
thing it names. That is the trade this project makes everywhere else.
"""
import pytest


@pytest.fixture(autouse=True)
def _reset_dynamo():
    """Give every test its own compile budget."""
    try:
        import torch._dynamo
    except Exception:
        yield
        return
    torch._dynamo.reset()
    yield
