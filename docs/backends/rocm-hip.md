# AMD ROCm/HIP support path

ROCm/HIP is a defined adapter contract, not validated performance support in this
workspace. No AMD device was used to qualify it, and its event timing must not be
reported as project evidence until a separate hardware gate records qualification.

When an AMD ROCm environment is deliberately being qualified, run exactly:

```bash
.venv/bin/python -m ratchet.backends --backend hip
```

`available` means only that PyTorch reports a visible HIP runtime and device. The adapter
remains `unvalidated`; the current dispatch policy therefore selects
`ratchet.amd.hip.eager.v1` as an untuned fallback. A future qualified run must use HIP
events with explicit synchronization, separate compilation and first-run work, and
reset/read peak-memory statistics. Missing HIP events, synchronization, compilation, or
required allocated peak-memory support is an explicit unsupported condition; reserved
peak memory is optional and host timing is never substituted.
