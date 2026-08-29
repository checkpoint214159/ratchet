# NVIDIA CUDA support path

CUDA is a defined adapter contract, not validated performance support in this
workspace. No NVIDIA device was used to qualify it, and its event timing must not be
reported as project evidence until a separate hardware gate records qualification.

When an NVIDIA-capable environment is deliberately being qualified, run exactly:

```bash
.venv/bin/python -m ratchet.backends --backend cuda
```

`available` means only that PyTorch exposed a non-HIP CUDA device. The adapter remains
`unvalidated`; the current dispatch policy therefore selects
`ratchet.nvidia.cuda.eager.v1` as an untuned fallback. A future qualified run must use
CUDA events with explicit synchronization, separate compilation and first-run work,
and reset/read peak-memory statistics. Missing CUDA events, synchronization,
compilation, or peak-memory support is an explicit unsupported condition; host timing
is never substituted.
