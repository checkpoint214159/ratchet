# Reference benchmark custody

`torch_transformer_benchmark.py` is the supplied **Authoritative Evaluator**. It is
stored here byte-for-byte from the handoff artifact; its SHA-256 is
`5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`.

The source is a compatibility contract, not an optimization surface. In particular,
`UserOptimizedTransformer.forward` is its designated candidate seam, and the
executable correctness condition is absolute-error **OR** relative-error tolerance.
`tests/evaluation/` verifies these facts by parsing the source without importing
PyTorch.

`benchmarks/reference/` is custody-only and exposes no importable application API.
Application code must not import it. A future `ratchet.evaluation` adapter or runner will
integrate candidates against this preserved contract without changing or importing the
reference source.

The supplied timing implementation synchronizes and uses device events only when the
device is CUDA. Its host-timer path for non-CUDA devices is compatibility output, not
Intel XPU (or other accelerator) performance evidence. Backend-aware timing belongs in
the sidecar measurement harness.
