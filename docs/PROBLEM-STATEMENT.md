# Problem Statement — TikTok TechJam 2026 #3: Implement a GPU Kernel for a Transformer Layer

**Read this before touching kernel code.** This file is the canonical, repo-owned copy of the
competition problem statement plus the engineering contract derived from the supplied
evaluator. It exists so that any agent harness working in this repository has the rules
without needing chat history or access to the (auth-gated) organizer document.

| | |
| --- | --- |
| Source | `https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf` (auth-gated) |
| Statement last updated | 27 August 2026, 6:25 PM |
| Changes in that update | Added *Appendix: Test Shapes*; updated `torch_transformer_benchmark.py` |
| Technical workshop | 28 Aug 2026, 3:00–3:45 PM — recording: *#3 Implement a GPU Kernel for a Transformer Layer.mp4* |
| Transcribed to repo | 2026-08-31 |

Sections 1–5 below are the organizer's text. Section 6 onward is **derived** from the
supplied evaluator source and is this repo's interpretation — clearly separated so the two
are never confused.

---

# PART A — The organizer's statement (authoritative text)

## 1. Background

Transformer is a widely used neural network architecture in modern AI. It is the core
structure behind many natural language processing, computer vision, speech, recommendation,
and large language model systems.

The main idea of Transformer is self-attention. Self-attention allows each token in a
sequence to interact with other tokens directly. Compared with recurrent models, Transformer
can process tokens in parallel, which makes it suitable for GPU acceleration.

Given an input sequence represented as a matrix $X \in \mathbb{R}^{N \times d}$, where $N$ is
the sequence length and $d$ is the hidden dimension, the Transformer first projects the input
into Query, Key, and Value matrices:

$$Q = XW_Q \qquad K = XW_K \qquad V = XW_V$$

The scaled dot-product attention is computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $d_k$ is the dimension of each attention head. The scaling factor $\sqrt{d_k}$ prevents
the dot-product values from becoming too large, which could make the softmax distribution
unstable.

However, the computation of Transformer is expensive. Important operations include matrix
multiplication, attention score calculation, softmax, normalization, and feed-forward layers.
These operations may be limited by GPU compute throughput, memory bandwidth, cache
efficiency, kernel launch overhead, and tensor core utilization.

In this competition, participants are asked to use AI-assisted methods to optimize the
runtime efficiency of a Transformer structure on a given GPU model. The optimized
implementation should improve performance while keeping the output numerically correct
compared with the reference implementation.

Participants may consider optimization methods such as operator fusion, memory layout
optimization, reduced-precision computation, tensor core usage, softmax optimization, and
custom CUDA, Triton, TensorFlow, or PyTorch implementations.

The goal of this task is to explore how AI can help developers analyze Transformer workloads,
identify bottlenecks, and generate more efficient implementations for specific GPU hardware.

## 2. Problem Statement

- Given a fixed formula of transformer layer, participants need to submit **one or several GPU
  kernels** that implement the layers that can pass the given test cases.
- The test cases would be written in PyTorch or TensorFlow and the participants **can modify
  the layer implementation** if they need, which means they can decide **which parts of the
  layers should be fused into 1 kernel**.
- The test case would compare the differences between the implementation of participants and
  the original PyTorch/TensorFlow implementation; the diff should be small enough
  (**relative error < 0.02, abs error < 0.002**).
- The test cases would contain **different shapes of input**, including large/small batch
  size, large/small sequence length, large/small dimensions, etc. The participants **can
  choose different implementations for different shapes by adding shape checks** in the
  implementation of layers. **All the combinations of input shapes will be told to the
  participants.**
- The use of AI tools is encouraged so that the participants can implement different kernels
  for different input shapes in limited time.
- **Optimize & test your codes on your own machine.** Different methods may be used to
  optimize the codes depending on the machine (GPU cards) you use.
- Provide a clear tech report including details on the **AI skills/tools used** to get bonus
  points.

What participants need to do:

1. Download the benchmark scripts (choose either torch or tensorflow, one of them is enough).
2. Implement the customized-implementation part and optimize it as fast as you can, by AI or
   by hand.
3. Run the script on your own machine.
4. Provide a clear tech report illustrating what the environment is (CPU, GPU, DISK, etc),
   what kind of optimizations you have done, and the final test results.

## 3. Constraints & Scope

| Category | Details |
| --- | --- |
| In scope | AI-based code generation, GPU kernel fusion, profile tools usage, etc. |
| Out of scope | Production-ready deployment. |

## 4. Available Resources

Download one and run it on your own machine:

- Torch benchmark script — `torch_transformer_benchmark.py` ← **this repo uses the torch track**
- TensorFlow benchmark script — `tensorflow_transformer_benchmark.py`

## 5. Deliverables

1. **Written Project Description (via Devpost)** — how the solution addresses the problem
   statement; development tools used (e.g. VSCode, Colab, Jupyter); APIs used; libraries and
   frameworks used; datasets and assets used.
2. **Public Code/GitHub Repository** — well-structured, commented code covering all
   components, plus a README containing: project overview, setup and installation
   instructions, steps to reproduce results, a brief reflection on limitations and what would
   be improved given more time, and team member contributions.
3. **Demo Video** — demonstrates the solution working end-to-end, uploaded to YouTube, public
   visibility, linked in the Devpost description, no third-party trademarks or copyrighted
   content without permission. For backend/NLP tracks where a front-end is not applicable, a
   walkthrough video showing API usage, inference examples, or result analysis is accepted.

## 6. Judging Criteria

| Criterion | Definition | Weight |
| --- | --- | --- |
| Technical Execution | Strong engineering fundamentals: well-structured code, thoughtful architecture, effective use of APIs or models. The demo runs reliably; technical complexity reflects deliberate, capable decision-making. | **35%** |
| Innovation & Problem Insight | Originality in idea and approach. Stands out for sharpness of problem understanding — how clearly the challenge is framed, why it matters, how directly the solution addresses it. | **20%** |
| Impact & Relevance | Clear potential to deliver value to real users or stakeholders — meaningful reach, tangible benefit, relevance beyond the hackathon prompt. | **20%** |
| Feasibility & Practicality | Realistic and buildable beyond a prototype. Technically and operationally sustainable; resource usage proportionate; architecture holds under real-world conditions; grounded rather than speculative. | **15%** |
| Presentation & Communication | *(Final event only)* Clear communication; coherent problem→solution→potential story; depth in Q&A. | **10%** |

Non-Technical-Execution criteria total **55%** — the methodology, the framing, and the
report carry more weight than raw kernel speed.

## 7. Appendix: Test Shapes

The appendix is published as an **unlabelled** 8-column table. Decoded column order:

```
id | batch_size | d_model | num_heads | seq_len | num_layers | causal | ffn_dim
```

Raw rows, verbatim:

```
1     64      128    4     128       4    TRUE    128
2     1       128    4     128       4    TRUE    128
3     4       128    4     128       4    TRUE    128
4     16      128    4     128       4    TRUE    128
5     128     128    4     128       4    TRUE    128
6     10000   128    4     128       4    TRUE    128
7     64      32     4     128       4    TRUE    32
8     64      1024   4     128       4    TRUE    1024
9     64      128    1     128       4    TRUE    128
10    64      128    2     128       4    TRUE    128
11    64      128    16    128       4    TRUE    128
12    64      128    4     32        4    TRUE    128
13    64      128    4     1024      4    TRUE    128
14    32      1024   16    100000    2    TRUE    1024
```

Same data, labelled, with `head_dim = d_model / num_heads` and the swept axis called out:

| id | batch | d_model | heads | seq | layers | causal | ffn_dim | head_dim | axis under test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 64 | 128 | 4 | 128 | 4 | ✓ | 128 | 32 | baseline |
| 2 | 1 | 128 | 4 | 128 | 4 | ✓ | 128 | 32 | batch ↓ (launch-bound) |
| 3 | 4 | 128 | 4 | 128 | 4 | ✓ | 128 | 32 | batch |
| 4 | 16 | 128 | 4 | 128 | 4 | ✓ | 128 | 32 | batch |
| 5 | 128 | 128 | 4 | 128 | 4 | ✓ | 128 | 32 | batch |
| 6 | 10000 | 128 | 4 | 128 | 4 | ✓ | 128 | 32 | batch ↑ (occupancy) |
| 7 | 64 | 32 | 4 | 128 | 4 | ✓ | 32 | 8 | d_model ↓ |
| 8 | 64 | 1024 | 4 | 128 | 4 | ✓ | 1024 | 256 | d_model ↑ (smem pressure) |
| 9 | 64 | 128 | 1 | 128 | 4 | ✓ | 128 | 128 | heads ↓ |
| 10 | 64 | 128 | 2 | 128 | 4 | ✓ | 128 | 64 | heads |
| 11 | 64 | 128 | 16 | 128 | 4 | ✓ | 128 | 8 | heads ↑ (many small heads) |
| 12 | 64 | 128 | 4 | 32 | 4 | ✓ | 128 | 32 | seq ↓ |
| 13 | 64 | 128 | 4 | 1024 | 4 | ✓ | 128 | 32 | seq ↑ (attention-heavy) |
| 14 | 32 | 1024 | 16 | 100000 | 2 | ✓ | 1024 | 64 | extreme seq |

Three invariants hold across **every** row:

1. **`causal == True` everywhere.** No non-causal config is tested. Optimizing the
   non-causal path is off-target.
2. **`ffn_dim == d_model` everywhere.** There is no 4× FFN expansion. Work tuned for a 4×
   FFN is tuned for a shape that is never tested.
3. **`num_layers` is 4** on rows 1–13 and **2** on row 14.

Config 14 (`seq_len = 100000`) makes the fp32 reference materialize an
`N×N` score matrix of roughly 1.3 TB per layer and **OOMs the baseline**, so it cannot be
scored against a same-machine fp32 reference. Report the other 13 and say so explicitly.

---

# PART B — Engineering contract derived from the evaluator

Everything below is read out of `benchmarks/reference/torch_transformer_benchmark.py`, held
in this repo byte-for-byte at SHA-256
`5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`.

**Custody rule:** that file is a compatibility contract, not an optimization surface. Do not
edit it, do not import it from application code. See `benchmarks/reference/README.md`.

## 8. The seam — the only thing you replace

```python
class UserOptimizedTransformer(BaselineTransformer):
    def forward(self, x, valid_token_mask=None) -> torch.Tensor:
```

Stated requirements: keep the forward signature unchanged; return a tensor of shape
`[batch_size, seq_len, d_model]`; keep compatible parameter names or customize
`copy_model_weights()`.

It **subclasses** the baseline, and `copy_model_weights(..., strict=True)` is the default, so
any parameter you add must either match a baseline key or be stored outside the state dict.

## 9. The layer you must reproduce

`BaselineTransformerBlock.forward` — **pre-norm**, two residuals:

```python
x = x + self.attention(self.norm1(x), valid_token_mask, causal)
x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))
if valid_token_mask is not None:
    x = x.masked_fill(~valid_token_mask[..., None], 0)
```

`BaselineTransformer.forward` runs the blocks, then `final_norm`, then zeroes invalid tokens
again.

`BaselineSelfAttention.forward` is deliberately unfused — this is the thing to beat:

```python
q, k, v = split_heads(q_proj(x)), split_heads(k_proj(x)), split_heads(v_proj(x))
scores = matmul(q, k.transpose(-2, -1)) * scale        # scale = head_dim ** -0.5
scores = scores.masked_fill(triu(1) causal_mask, -inf)  # when causal
scores = scores.masked_fill(~valid_token_mask, -inf)    # invalid KEY positions
probs = softmax(scores.float(), -1).to(x.dtype)         # softmax in fp32, cast back
context = matmul(probs, v)
output = out_proj(context.transpose(1,2).contiguous().view(B, S, d_model))
output = output.masked_fill(~valid_token_mask[..., None], 0)
```

Numerics that must be matched exactly:

- **GELU is exact erf** (`approximate="none"`), not the tanh approximation.
- **Softmax is computed in fp32** and cast back — so a streaming/online softmax must carry an
  fp32 accumulator.
- `scale = head_dim ** -0.5`, using the **true** head_dim (not a padded one).
- All four projections (`q/k/v/out`) and both FFN linears **have bias**.

## 10. Correctness gate

```python
abs_ok = abs_error <= atol                # default 0.002
rel_ok = abs_error <= rtol * ref.abs()    # default 0.02
passed = finite_mask & (abs_ok | rel_ok)
```

- The condition is **OR**, not AND, and it is applied **elementwise**. Every element must
  pass; `passed = (failed_elements == 0)`.
- Deliberately **not** `torch.isclose`, which uses `atol + rtol*|ref|` and is more permissive.
- `--accuracy-trials` defaults to **5**, each with a different seed.
- If accuracy fails, the benchmark is **skipped entirely** and the script returns exit code 2
  (unless `--benchmark-on-failure`). A fast, wrong kernel scores nothing.

⚠️ **Known stale docstring.** Line 11 of the evaluator says *"The default thresholds are
atol=0.001 and rtol=0.01 (1%)"*. That is **wrong** — it contradicts both `parse_args`
(`--atol 0.002`, `--rtol 0.02`) and §2 of the statement. Trust argparse and the statement.
Do not tune to the 2×-stricter docstring bound.

## 11. How speed is measured

- `torch.cuda.Event` on the current stream, per iteration.
- `--warmup 20`, `--repeats 100`, `--benchmark-rounds 3`.
- Rounds **alternate order** (baseline-first / optimized-first) to cancel thermal and clock
  drift.
- Reported `speedup = baseline.median_ms / optimized.median_ms` — **median**, not mean.
- Also reports `tokens/s`, where `tokens_per_call = batch_size * seq_len`.
- Timing uses a single fixed input generated with `seed + 100000`; data generation is excluded.

## 12. CLI defaults that change what "winning" means

| Flag | Default | Consequence |
| --- | --- | --- |
| `--causal` | **False** (`store_true`) | Every appendix row is causal — you **must** pass `--causal`. |
| `--dtype` | `float32` | Model and input are cast to this. The reference is fp32 unless overridden. |
| `--padding-ratio` | `0.0` | Mask is all-ones — but a mask tensor is **always** passed, never `None`. See §13. |
| `--compile-baseline` | **off** | The official baseline is **eager**, not `torch.compile`. |
| `--compile-user` | off | Your code is not compiled for you. |
| `--allow-tf32` | **True** | The baseline already gets TF32 tensor cores. |
| `--matmul-precision` | `high` | Same — `torch.set_float32_matmul_precision("high")`. |
| `--input-scale` | `1.0` | Scales the input; larger values stress the relative-error bound. |
| batch/seq/d_model/heads/ffn/layers | `8 / 128 / 512 / 8 / 2048 / 6` | **Not an appendix row.** Script defaults only; always pass the matrix shape explicitly. |

## 13. Traps

**T1 — `valid_token_mask` is always a tensor.** `run_accuracy_tests` and `benchmark_models`
both call `model(x, valid_mask)`. At `--padding-ratio 0.0` the mask is all-ones, so a
fast path guarded on "mask is all true" survives. At **any `--padding-ratio > 0`**, a kernel
without a padding path must fall back, and the measured speedup collapses toward 1.0x. Decide
deliberately whether to support padding, and state the decision in the report.

**T2 — masking is on invalid *key* positions**, shaped `[B,1,1,S]`, and is applied *after*
the causal mask. Both the block output and the final output are additionally zeroed at
invalid *query* positions.

**T3 — a fully-masked row is possible in principle.** The baseline would produce `NaN` from
`softmax(-inf ...)`; `finite_mask` then excludes those elements from the gate. A streaming
softmax that guards `l_i == 0` produces `0` instead. This only diverges if such a row occurs;
`generate_random_case` guarantees `min_valid >= 1`, so under the supplied generator it does
not.

**T4 — dtype mismatch is a warning, not an error.** `compare_outputs` prints
`[warning] dtype mismatch` and proceeds by casting both to fp32. Returning a different dtype
than the reference will not fail the gate, but it is a smell and may cost Technical Execution.

**T5 — reduced precision must survive `num_layers` of accumulation.** The gate is applied to
the *final* output after 4 layers plus `final_norm`, not per-op. bf16 (8 mantissa bits) has
been measured to fail this gate on every appendix config; fp16 (10 mantissa bits) with fp32
accumulation passes. See `ledger/bench_results.jsonl`.

## 14. Verification status of this document

Checked on 2026-08-31 against the repo:

- ✅ `ratchet/kernels/dispatch.py::MATRIX` is an **exact 14/14 match** to the appendix above
  (verified field-by-field, including `causal` and `ffn_dim`).
- ✅ `benchmarks/reference/torch_transformer_benchmark.py` **is the 27 Aug 2026 updated
  version** — line count and every distinguishing fingerprint (`--benchmark-on-failure`,
  `--allow-tf32`, `--matmul-precision`, `--input-scale`, `--non-strict-weight-copy`,
  `clamp_min(1e-12)`, the alternating-round comment, and the stale atol/rtol docstring) match
  the organizer's published source.

## 15. How this repo answers the statement

| Statement clause | Where it is answered |
| --- | --- |
| "one or several GPU kernels that implement the layers" | `ratchet/kernels/flash_attention.py`, `ratchet/kernels/linear_tf32.py` |
| "decide which parts should be fused into 1 kernel" | fused QKV projection; bias+GELU folded into the GEMM epilogue; whole attention in one kernel |
| "different implementations for different shapes by adding shape checks" | `ratchet/kernels/dispatch.py::select(cfg, prof) -> Recipe` |
| "rel error < 0.02, abs error < 0.002" | fp32 accumulation in every kernel; gate enforced in `tests/manual/search_loop.py` and `matrix_bench.py` |
| "all the combinations of input shapes will be told to the participants" | `dispatch.py::MATRIX`, verified against §7 |
| "optimize & test on your own machine" | GB10 / `sm_121`; device facts in `ledger/device.gb10.json` |
| "AI skills/tools used to get bonus points" | `tests/manual/search_loop.py` — LLM-in-the-loop propose→gate→measure→record→select, with an append-only git-provenanced ledger |

**Runtime note:** the Triton kernels require the system CUDA-13 ptxas —
`export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.

See `SUBMISSION.md` for the measured results and the submittable file manifest.
