# Hardware Testing Guide: Failure-Aware Pruning on RTX 4060

## Setup Checklist

- [ ] NVIDIA drivers installed (`nvidia-smi` works)
- [ ] CUDA 12.1+ available
- [ ] PyTorch 2.5+ with GPU support
- [ ] Triton 2.1+ (must support NVIDIA)
- [ ] Python 3.10+

### Verify RTX 4060 Setup

```bash
# In your Ratchet environment:
.venv\Scripts\python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

Expected output:
```
GPU: NVIDIA GeForce RTX 4060
Memory: 8.0 GB
```

---

## Step 1: Profile Your RTX 4060

Create a calibration record to feed into the pruning logic.

```bash
.venv\Scripts\python -c """
import torch

dev = torch.cuda.get_device_properties(0)
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Compute Capability: {dev.major}.{dev.minor}')
print(f'SM Count: {dev.multi_processor_count}')
print(f'Max Threads Per Block: {dev.max_threads_per_block}')
print(f'Shared Memory Per Block: {dev.shared_memory_per_block // 1024} KB')
print(f'L2 Cache: {dev.l2_cache_size // (1024*1024) if dev.l2_cache_size else 0} MB')

# Test bandwidth (rough estimate)
import time
size_mb = 256
x = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device=0)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    y = x.clone()
    torch.cuda.synchronize()
t1 = time.time()
bw_gb_s = (size_mb * 100 * 2) / (t1 - t0) / 1024
print(f'Measured Bandwidth: ~{bw_gb_s:.1f} GB/s (theoretical max: 288 GB/s)')
"""
```

Record the output in `ledger/device.json` (or create a new entry).

---

## Step 2: Create a Simple Test Kernel

Start with a minimal Triton attention kernel to test against.

```python
# test_kernel_simple.py
import torch
import triton
import triton.language as tl

@triton.jit
def attention_fwd_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    M, N, D,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Minimal attention kernel for testing."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # Load query block
    q_offset = pid_b * stride_qb + pid_h * stride_qh + (pid_m * BLOCK_M) * stride_qm
    q_ptrs = Q + q_offset + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, BLOCK_D)[None, :] * 1
    q = tl.load(q_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < M) & (tl.arange(0, BLOCK_D)[None, :] < D))
    
    # Output
    o_offset = pid_b * stride_ob + pid_h * stride_oh + (pid_m * BLOCK_M) * stride_om
    o_ptrs = Out + o_offset + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, BLOCK_D)[None, :] * 1
    tl.store(o_ptrs, q, mask=(tl.arange(0, BLOCK_M)[:, None] < M) & (tl.arange(0, BLOCK_D)[None, :] < D))


def attention_fwd(q, k, v, block_m=128, block_n=64, block_d=64):
    """Forward pass."""
    b, h, m, d = q.shape
    _, _, n, _ = k.shape
    
    out = torch.empty_like(q)
    
    grid = (b, h, (m + block_m - 1) // block_m)
    
    attention_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        m, n, d,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
    )
    
    return out


if __name__ == "__main__":
    # Test shapes
    b, h, m, n, d = 2, 8, 512, 512, 64
    q = torch.randn(b, h, m, d, device=0, dtype=torch.bfloat16)
    k = torch.randn(b, h, n, d, device=0, dtype=torch.bfloat16)
    v = torch.randn(b, h, n, d, device=0, dtype=torch.bfloat16)
    
    out = attention_fwd(q, k, v)
    print(f"Output shape: {out.shape}")
```

---

## Step 3: Test Pruning on Real Failure Patterns

Modify `failure_aware_pruning.py` to capture real Triton compile errors:

```python
# Add to failure_aware_pruning.py

def capture_real_failures(test_kernel_file: str, num_evals: int = 50):
    """
    Test a real Triton kernel against the config space.
    Record actual compile failures.
    """
    import subprocess
    import json
    
    failures = []
    
    for i, config in enumerate(generate_config_space()[:num_evals]):
        # Generate a temporary kernel with this config
        temp_code = generate_triton_kernel_with_config(test_kernel_file, config)
        
        # Try to compile
        try:
            compile_result = subprocess.run(
                [".venv/bin/python", "-c", f"import triton; {temp_code}"],
                capture_output=True,
                timeout=5
            )
            if compile_result.returncode != 0:
                error_msg = compile_result.stderr.decode()
                failures.append({
                    "config": config.to_tuple(),
                    "error": error_msg,
                    "category": categorize_error(error_msg)
                })
        except subprocess.TimeoutExpired:
            failures.append({
                "config": config.to_tuple(),
                "error": "Timeout",
                "category": "timeout"
            })
    
    return failures

def categorize_error(error_msg: str) -> str:
    """Classify compile error."""
    if "shared" in error_msg.lower() or "memory" in error_msg.lower():
        return "shared_memory"
    if "register" in error_msg.lower():
        return "register_spill"
    if "out of memory" in error_msg.lower():
        return "oom"
    if "timeout" in error_msg.lower():
        return "timeout"
    return "other"
```

---

## Step 4: Compare Baseline vs Pruned on RTX 4060

Run the experiment directly on your kernel:

```bash
# Generate calibration
.venv\Scripts\python -c "
from ratchet.experiments.failure_aware_pruning import *
import json
from pathlib import Path

# Profile RTX 4060
device_info = {
    'name': 'RTX 4060',
    'shared_memory_kb': 99,
    'l2_cache_mb': 0,  # RTX 4060 has minimal L2
    'sm_count': 20,
    'bandwidth_gb_s': 288,  # Theoretical max
}

Path('ledger/device_rtx4060.json').write_text(json.dumps(device_info, indent=2))
print('✓ Device profile saved')

# Run experiment
results = run_experiment()
"
```

---

## Step 5: Measure Time Savings

The real metric: **how much compile time is saved?**

```python
import time
from pathlib import Path

def measure_compile_cost(kernel_file: str, config_space: list[KernelConfig]):
    """Time how long compile attempts take."""
    total_time = 0
    failed_count = 0
    
    for cfg in config_space:
        t0 = time.time()
        try:
            # Attempt compilation
            result = compile_kernel(kernel_file, cfg)
            if not result:
                failed_count += 1
        except Exception:
            failed_count += 1
        t1 = time.time()
        
        total_time += (t1 - t0)
        if failed_count > 0:
            print(f"  Config {cfg}: FAIL ({t1-t0:.2f}s wasted)")
    
    print(f"Total wasted on failures: {total_time:.2f}s")
    return total_time, failed_count
```

---

## Troubleshooting

### "Triton not found"
```bash
.venv\Scripts\python -m pip install triton
# Or: .venv\Scripts\python -m pip install torch
```

### "CUDA not available"
```bash
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
# Should print True
```

### "Shared memory exceeded"
Check your RTX 4060's actual shared memory:
```bash
nvidia-smi --query-gpu=memory.shared --format=csv,noheader
```

Update `simulate_kernel_performance()` with the correct value.

---

## Expected Findings

On RTX 4060, you should see:

| Scenario | Failure Rate | Speedup | Notes |
|---|---|---|---|
| Large tiles (BLOCK_M=256) | ~80% | N/A | Shared memory exhausted |
| Register-heavy (many warps) | ~30% | N/A | Register spill |
| Balanced config | ~5% | 1.2–1.5x | Sweet spot |
| Pruned search | ~0% | 1.1–1.4x | Conservative but reliable |

---

## Next: Other Unexplored Methods

After validating failure-aware pruning on hardware, consider:

1. **Cross-hardware transfer** — Use RTX 4060 failures to predict A100/H100 failures
2. **PageRank difficulty forecasting** — Predict which kernels need more tuning budget
3. **FP8 quantization variants** — Test lower-precision attention
4. **Pipelining strategies** — Asynchronous producer/consumer patterns

See `FAILURE_AWARE_PRUNING_SUMMARY.md` for details.
