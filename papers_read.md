# Papers and primary documentation read

This tracker contains only sources read for IB-08. None of the entries is a Ratchet
measurement or a performance claim about this workspace.

| Key | Primary source read | What it contributes to a future protocol |
| --- | --- | --- |
| `dao2022flashattention` | [FlashAttention, NeurIPS 2022](https://arxiv.org/abs/2205.14135), DOI [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135) | Exact attention can be organized around IO-aware tiling; it motivates measuring attention separately only after the qualified-hardware gate. |
| `dao2024flashattention2` | [FlashAttention-2, ICLR 2024](https://arxiv.org/abs/2307.08691), DOI [10.48550/arXiv.2307.08691](https://doi.org/10.48550/arXiv.2307.08691) | Work partitioning and reduced non-matmul work are future candidate dimensions, not an assumed Arc result. |
| `ansel2024pytorch` | [PyTorch 2, ASPLOS 2024](https://pytorch.org/assets/pytorch2-2.pdf), DOI [10.1145/3620665.3640366](https://doi.org/10.1145/3620665.3640366) | `torch.compile`/Inductor is an explicit eager-versus-compiled baseline condition. |
| `schoonhoven2022autotuning` | [Benchmarking optimization algorithms for auto-tuning GPU kernels](https://arxiv.org/abs/2210.01465), DOI [10.1109/TEVC.2022.3210654](https://doi.org/10.1109/TEVC.2022.3210654) | Tuning is target-, shape-, and code-dependent, so every future candidate needs bounded search and recorded budget. |
| `pytorch_xpu_2026` | [PyTorch XPU API](https://docs.pytorch.org/docs/stable/xpu.html) and [Intel-GPU install guide](https://docs.pytorch.org/docs/main/notes/get_start_xpu.html) | Qualification must first check XPU availability, synchronization, events, and memory APIs against the installed version. |
| `intel_ipex_retirement_2026` | [Archived Intel Extension for PyTorch repository](https://github.com/intel/intel-extension-for-pytorch) | The repository was archived on 2026-03-30; its retirement notice says active development and quarterly releases had ceased and recommends direct PyTorch. |
| `intel_joint_matrix_2024` | [Intel oneAPI joint-matrix/XMX guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/joint-matrix.html) | XMX/DPAS is a later lower-level branch: it requires XMX hardware and has no emulation fallback. |
| `intel_triton_xpu_2026` | [Intel XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton) and its [architecture note](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/docs/ARCHITECTURE.md) | Future Triton trials may tune descriptors, blocks, warps, stages, GRF mode, and grid ordering only on a qualified Intel GPU. |
| `spoczynski2026xeforge` | [Xe-Forge paper](https://arxiv.org/abs/2605.26118), DOI [10.48550/arXiv.2605.26118](https://doi.org/10.48550/arXiv.2605.26118), and [official repository](https://github.com/IntelLabs/Xe-Forge) | Its propose-validate-benchmark loop reinforces the required candidate/correctness/evidence ordering; its reported results are external and are not transferred here. |

## IPEX retirement for this project

The future protocol retires Intel Extension for PyTorch (IPEX) as an integration path:
it will not be installed, imported, or used as evidence. Intel's archived IPEX repository
was archived on 2026-03-30; its retirement notice says active development and quarterly
releases had ceased and recommends direct PyTorch. The current PyTorch XPU guide uses
PyTorch XPU wheels, while Intel's
Triton-XPU repository explicitly says its backend is not compatible with IPEX. The planned
stack is therefore qualified PyTorch XPU first, then `torch.compile`, with Triton-XPU
considered only as a separately versioned path.

Every read/to-read state change is retained in the chained immutable records under
[`research/literature/history/`](research/literature/history/).
