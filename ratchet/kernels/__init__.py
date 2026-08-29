"""Candidate kernel implementations.

This package is an intentional placeholder. It is the reserved home for concrete,
per-backend candidate kernels (Triton/CUDA/HIP/XPU) produced by the autoresearch loop.
It is empty on purpose on this checked-in build: the environment is hardware-gated
(``ENV-0001`` records no qualified runtime), so no candidate has been generated, compiled,
or measured. A candidate is created only inside an isolated experiment worktree on
qualified hardware, after the authoritative correctness matrix passes and before any
timing, and is then recorded as immutable evidence.

Keeping the package (rather than deleting it) documents the architecture's intended
extension point without implying a candidate exists. Do not add a kernel here directly on
``master``; add it through the experiment-workspace lifecycle so its provenance is bound to
a base commit and protocol digest.
"""
