# Hardware support

Ratchet isolates vendor APIs behind `ratchet.backends`. CPU is a correctness-only path;
Intel XPU, NVIDIA CUDA, and AMD ROCm/HIP adapters expose availability, capabilities,
synchronization, event timing, memory observation, and compilation contracts without
leaking SDK objects across the public boundary.

| Backend | Current state | Probe | Dispatch state |
| --- | --- | --- | --- |
| Intel XPU | unavailable in `ENV-0001` | `.venv/bin/python -m ratchet.backends --backend xpu` | untuned, no-run |
| NVIDIA CUDA | adapter contract only | `.venv/bin/python -m ratchet.backends --backend cuda` | untuned fallback |
| AMD ROCm/HIP | adapter contract only | `.venv/bin/python -m ratchet.backends --backend hip` | untuned fallback |

A positive probe means only that a runtime/device appears available. It does not qualify
correctness, timing, memory, compilation, or a tuned dispatch. Each vendor needs a
separately ratified hardware hierarchy. Intel FG-01 additionally requires allocation,
SDPA, compiler behavior, synchronization, device events, memory APIs, supported dtypes,
and the full evaluator matrix before any empirical event is allowed.
