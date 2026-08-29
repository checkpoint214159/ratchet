# Papers and primary documentation to read

These sources are known backlog only: they have not been read for this project, are not
in `bibliography.bib`, and must not support a project claim or hypothesis until moved to
`papers_read.md` through a chained transition record.

| Backlog item | Primary source | Why it is queued |
| --- | --- | --- |
| *A methodology for comparing optimization algorithms for auto-tuning* | [Future Generation Computer Systems 159 (2024), 489–504](https://doi.org/10.1016/j.future.2024.05.021) | Read before choosing a future search-budget methodology; no claim is drawn from it yet. |
| PyTorch custom SYCL operators | [Official PyTorch tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops_sycl.html) | Read before considering a custom-SYCL escape hatch after the PyTorch XPU gate. |
| Intel oneAPI GPU Optimization Guide | [Official Intel guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/overview.html) | Read before using low-level occupancy, register, or memory recommendations in a future Intel protocol. |
| FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision | [NeurIPS 2024 / arXiv 2407.08608](https://arxiv.org/abs/2407.08608) | Asynchronous warp-specialization and ping-pong GEMM-softmax interleaving for next-generation attention kernels. |
| FlashDecoding++: Faster Large Language Model Inference on GPUs | [arXiv 2311.01282](https://arxiv.org/abs/2311.01282) | Asynchronized softmax with unified max value, flat GEMM optimization, and heuristic dataflow adaptation across dynamic shapes. |
| KernelPro: Multi-Agent Automated GPU Kernel Generation via Profiler Feedback | [arXiv 2606.26453](https://arxiv.org/abs/2606.26453) | Micro-profiling hardware counter translation into natural-language LLM diagnostic guidance for iterative kernel refinement. |
| Harness Engineering for LLM-Driven GPU Kernel Generation | [arXiv 2607.17979](https://arxiv.org/abs/2607.17979) | Evidence that an isolated, rigorous testing and benchmarking harness outperforms autonomous, unconstrained LLM code generation. |
| KernelBench-Verified: Evaluating LLMs on GPU Kernel Generation | [arXiv 2607.16241](https://arxiv.org/abs/2607.16241) | Rigorous verification discipline to avoid benchmark leakage, timing artifacts (L2 cache flushes), and tolerance violations. |
| The Anatomy of a Triton Attention Kernel | [arXiv 2511.11581](https://arxiv.org/abs/2511.11581) | Replacing static autotuning with offline-calibrated shape-aware dispatch trees (2D vs Split-K 3D grid thresholds). |
| FlashInfer: Fast and Customizable LLM Serving Operators | [arXiv 2410.02536 / GitHub](https://github.com/flashinfer-ai/flashinfer) | Cascade attention and log-sum-exp state merging math for split-KV attention across dynamic batch and sequence lengths. |
| Liger Kernel: Efficient Triton Kernels for LLM Training and Inference | [GitHub](https://github.com/linkedin/Liger-Kernel) | Reference implementations for fused LayerNorm/RMSNorm + residual and fused SwiGLU / GeLU MLP layers. |
| Dr. Kernel: Reinforcement Learning for GPU Kernel Optimization | [arXiv 2602.05885](https://arxiv.org/abs/2602.05885) | Operator-weighted runtime rewards preventing search from spending budget on non-bottleneck transformer operations. |
| CUDA Agent: Autonomous Kernel Engineering with Domain Knowledge | [arXiv 2602.24286](https://arxiv.org/abs/2602.24286) | Iterative LLM search and anti-reward hacking guards for GPU kernel generation. |
| KernelBrain: Multi-Fidelity Iterative GPU Kernel Synthesis | [arXiv 2608.02611](https://arxiv.org/abs/2608.02611) | Successive halving and multi-fidelity screening for efficient kernel parameter exploration. |

Future additions enter this tracker first and must preserve a chained transition record
under `research/literature/history/` before they may be cited.
