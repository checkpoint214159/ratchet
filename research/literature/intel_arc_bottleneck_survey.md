# Intel Arc Transformer Bottleneck Survey

This is a cited literature survey for the future Intel Arc/XPU target. It is **not a
project profile**: Ratchet has no qualified XPU runtime and has produced no project
profiling trace, timing, or kernel result.

## Literature observations

| ID | Literature observation | Reviewed source |
| --- | --- | --- |
| `OBS-001` | Exact attention can reduce IO through tiled algorithms. | `dao2022flashattention` |
| `OBS-002` | Attention implementations can change work partitioning and non-matmul overhead. | `dao2024flashattention2` |
| `OBS-003` | PyTorch 2 introduces optional `torch.compile` graph capture/compilation alongside eager execution. | `ansel2024pytorch` |
| `OBS-004` | Kernel autotuning depends on target, shape, and code configuration. | `schoonhoven2022autotuning` |
| `OBS-005` | Intel XMX joint-matrix operations are a hardware-specific lower-level path. | `intel_joint_matrix_2024` |
| `OBS-006` | Intel Triton-XPU exposes descriptor, block, warp, stage, GRF-mode, and grid-ordering controls. | `intel_triton_xpu_2026` |

## Unmeasured project hypotheses

Every item below is a **Hypothesis**, not an observed Ratchet bottleneck.

- `HYP-LIT-001` — **Hypothesis:** attention data movement may merit isolated sidecar
  measurement after the qualified XPU gate. Source: `dao2022flashattention`.
- `HYP-LIT-002` — **Hypothesis:** attention work partitioning and non-matmul work may be
  worth comparing under the authoritative full workload after the qualified XPU gate.
  Source: `dao2024flashattention2`.
- `HYP-LIT-003` — **Hypothesis:** eager and `torch.compile` should be measured as separate
  baseline conditions after the qualified XPU gate. Source: `ansel2024pytorch`.
- `HYP-LIT-004` — **Hypothesis:** any future tuning search should be bounded and recorded
  per workload configuration. Source: `schoonhoven2022autotuning`.
- `HYP-LIT-005` — **Hypothesis:** after the qualified XPU gate, XMX joint-matrix work is a
  future later branch only if a measured profile establishes headroom and hardware support.
  Source: `intel_joint_matrix_2024`.
- `HYP-LIT-006` — **Hypothesis:** after the qualified XPU gate, Triton-XPU schedule controls
  are a future later versioned experiment branch, not a present implementation path. Source:
  `intel_triton_xpu_2026`.

The machine-readable record is
[`intel_arc_bottleneck_survey.json`](intel_arc_bottleneck_survey.json). Every source key is
reviewed and resolves in `research/paper/bibliography.bib`.
