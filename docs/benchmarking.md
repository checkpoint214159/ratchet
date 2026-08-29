# Benchmarking and measurement

The byte-preserved reference evaluator at
`benchmarks/reference/torch_transformer_benchmark.py` controls workload and correctness
semantics. Its SHA-256 is recorded in `.beryl/agent/project-brief.md`. Do not change its
baseline, input generation, CLI semantics, or absolute-OR-relative tolerance to improve a
reported result.

## Qualification measurement contract

A qualified accelerator run must use the identical evaluator models, copied weights,
inputs, configuration, dtype, compiler policy, and process for baseline and candidate.
Run five correctness trials first. Only a correct candidate can be timed.

Separate compilation time and first-run latency from steady state. Warm each arm for 20
completed calls, then perform ten alternating ABBA/BAAB blocks with 30 completed calls
per arm. Device events and explicit synchronization are required; a bounded synchronized
host timer is a cross-check. Record raw samples, median, mean, p90, minimum, standard
error, paired-bootstrap 95% speedup interval, peak memory, clocks/power, toolchain, and
method.

Promotion requires complete correctness, non-overlapping latency intervals, paired lower
speedup above 1.02, and no unexplained peak-memory increase above 5 percent. CPU timing
and the evaluator's unsynchronized non-CUDA host timer are diagnostic only.

## Current gate

The XPU doctor recorded unavailable hardware in `ENV-0001`; no candidate may be generated,
compiled, tested, profiled, or timed in this build. Cached or historical timings cannot
substitute for a new result.
