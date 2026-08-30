"""Measure a candidate against the announced matrix and record it to the ledger.

Measurement lives IN-TREE rather than in a scratch script so that the commit sha the
ledger records actually describes the code that ran. A measurement whose provenance is a
throwaway file is not reproducible, and an irreproducible number is not evidence.

Correctness runs BEFORE timing, in the same process, and a candidate that fails is never
timed. The tolerances come from the custody benchmark's own CLI defaults; they are not
ours to choose.

    python3 bench/run_matrix.py --candidate v2_fp16_flash            # all configs
    python3 bench/run_matrix.py --candidate v2_fp16_flash --ids 6 13 # a subset
    python3 bench/run_matrix.py --candidate v2_fp16_flash --dry-run  # no ledger write

One config per subprocess: a candidate can OOM, hang, or take the CUDA context down with
it, and none of those may cost us the rest of the run. An OOM is a RESULT -- it is
recorded with `status="oom"`, because "the reference cannot run this shape" is one of the
more valuable things the matrix can tell us.

WHEN THE REFERENCE CANNOT RUN THE SHAPE
---------------------------------------
That last sentence was true and the code did not honour it. A shape the reference cannot
execute produced `status="oom"` and 600 characters of somebody's traceback -- the least
informative thing the matrix can say about its most interesting row, and it said it 27
times before anyone read it.

So there is a second measurement path now, taken when `bench/feasibility.py`'s predicate
says the REFERENCE's own algorithm does not fit the device. It is not a fallback for a
candidate that failed: it is the honest protocol for a config where the usual method is
unavailable because there is no baseline to compare against.

  * The baseline's requirement is DERIVED from the reference's source and then CONFIRMED
    empirically -- the score tensor is actually asked for, and the driver's refusal, with
    its byte count, goes in the row.
  * The candidate is run at the real shape, one slice at a time, and what completed is
    recorded as a capability with its peak memory.
  * Correctness uses the two oracles in `bench/feasibility.py`: the causal-prefix theorem
    against the UNMODIFIED reference at the real sequence length, and a blocked fp64
    evaluation of the reference's own arithmetic covering every row.
  * `status="reference_infeasible"`, and `timing.speedup` is None. There is no speedup.
    A ratio needs two measured times and the denominator does not exist; manufacturing
    one would be a number about us, not about the reference. The config therefore scores
    1.0 in `weighted_score` -- the same as not measuring it -- and that understatement is
    deliberate and is stated in the report rather than quietly corrected here.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "benchmarks" / "reference" / "torch_transformer_benchmark.py"

# The custody benchmark's own defaults. Never widen these to make something pass.
RTOL, ATOL = 0.02, 0.002
ACCURACY_TRIALS = 5
SEED = 1234


def load_reference():
    """Import the pinned benchmark without importing it as a package or touching it."""
    spec = importlib.util.spec_from_file_location("ref_bench", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ref_bench"] = module
    spec.loader.exec_module(module)
    return module


# ======================================================================================
# The measured work -- runs inside the child process
# ======================================================================================

def measure_one(config_id: int, candidate_name: str, samples: int = 300,
                padding: float = 0.0, dtype_name: str = "float32",
                input_scale: float = 1.0, oracle_sequences: int = 1) -> dict:
    import torch

    sys.path.insert(0, str(REPO))
    from bench.matrix import BY_ID
    from bench.candidates import REGISTRY

    ref = load_reference()
    cfg = BY_ID[config_id]
    device = torch.device("cuda")
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[dtype_name]

    torch.set_float32_matmul_precision("high")     # TF32 on, for BOTH arms
    torch.backends.cuda.matmul.allow_tf32 = True

    tcfg = ref.TransformerConfig(
        batch_size=cfg.batch_size, seq_len=cfg.seq_len, d_model=cfg.d_model,
        num_heads=cfg.heads, ffn_dim=cfg.ffn_dim, num_layers=cfg.layers,
        causal=cfg.causal,
    )
    tcfg.validate()

    # Does the REFERENCE's own algorithm fit this device? Shapes and one measured device
    # property; no config id (CLAUDE.md rule 2), and it answers differently on a
    # different card. When it does not fit there is no baseline to time against, and
    # `measure_one` below would spend its first act building one.
    from bench.feasibility import reference_feasible
    _total = torch.cuda.get_device_properties(device).total_memory
    _feasible, _why = reference_feasible(cfg.batch_size, cfg.seq_len, cfg.d_model,
                                         cfg.heads, dtype.itemsize, _total)
    if not _feasible:
        r = capability_one(config_id, candidate_name, ref, tcfg, cfg, device, dtype,
                           oracle_sequences=oracle_sequences)
        r["padding_ratio"] = padding
        return r

    out: dict = {"config_id": config_id, "candidate": candidate_name,
                 "padding_ratio": padding,
                 "dtype": dtype_name}

    def make_input(seed):
        return ref.generate_random_case(tcfg, device, dtype, seed=seed,
                                        padding_ratio=padding, input_scale=input_scale)

    def median_ms(model, x, mask, n):
        with torch.inference_mode():
            for _ in range(20):
                model(x, mask)
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
            for i in range(n):
                starts[i].record()
                model(x, mask)
                ends[i].record()
            torch.cuda.synchronize()
        vals = sorted(a.elapsed_time(b) for a, b in zip(starts, ends))
        return vals[len(vals) // 2]

    # ORDER MATTERS, and it is the fix for a measurement bug we made and caught:
    # holding both models resident inflated config 6's baseline 4.1x (446 -> 1851 ms)
    # because 18.4 GB of reserved memory on a 16 GB card spills to host over PCIe.
    # So each arm is timed while it is the ONLY model on the device.
    #
    # The cost is losing cross-arm interleaving, which is our defence against thermal
    # drift on a card whose clocks cannot be locked. That trade is deliberate: the
    # memory-pressure distortion is 410%, drift between two adjacent measurements is a
    # few percent, and `interleaved: False` is recorded so nobody has to guess.
    torch.manual_seed(SEED)
    baseline = ref.BaselineTransformer(tcfg).to(device=device, dtype=dtype).eval()

    xt, mt = make_input(SEED + 100000)
    probe = median_ms(baseline, xt, mt, 3)
    n = max(11, min(samples, int(2000.0 / max(probe, 0.05))))

    torch.cuda.reset_peak_memory_stats()
    base_ms = min(median_ms(baseline, xt, mt, n), median_ms(baseline, xt, mt, n))
    base_peak = torch.cuda.max_memory_allocated() / 1e6

    # Now build the candidate and check correctness -- both models resident, which is
    # unavoidable here, but correctness is not timed so pressure cannot distort it.
    torch.manual_seed(SEED)
    fresh_baseline = ref.BaselineTransformer(tcfg)
    cand_cls = REGISTRY[candidate_name].build(ref.BaselineTransformer)
    candidate = cand_cls(tcfg)
    ref.copy_model_weights(fresh_baseline, candidate)   # CPU, fp32, before .to()
    fresh_baseline = fresh_baseline.to(device=device, dtype=dtype).eval()
    candidate = candidate.to(device=device, dtype=dtype).eval()

    trials, worst_abs, worst_rel, failed = [], 0.0, 0.0, 0
    with torch.inference_mode():
        for t in range(ACCURACY_TRIALS):
            x, mask = make_input(SEED + t)
            expected = fresh_baseline(x, mask)
            got = candidate(x, mask)
            res = ref.compare_outputs(expected, got, rtol=RTOL, atol=ATOL)
            trials.append({"passed": bool(res.passed),
                           "max_abs": float(res.max_abs_error),
                           "max_rel": float(res.max_relative_error),
                           "failed": int(res.failed_elements)})
            worst_abs = max(worst_abs, float(res.max_abs_error))
            worst_rel = max(worst_rel, float(res.max_relative_error))
            failed += int(res.failed_elements)
            del x, mask, expected, got
    passed = all(t["passed"] for t in trials)
    out["correctness"] = {"passed": passed, "max_abs": worst_abs, "max_rel": worst_rel,
                          "failed_elements": failed, "trials": len(trials)}
    if not passed:
        out["status"] = "incorrect"
        return out

    # Free every baseline before timing the candidate, for the same reason as above.
    del baseline, fresh_baseline
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    cand_ms = min(median_ms(candidate, xt, mt, n), median_ms(candidate, xt, mt, n))
    cand_peak = torch.cuda.max_memory_allocated() / 1e6

    out["status"] = "ok"
    out["timing"] = {
        "baseline_ms": base_ms, "candidate_ms": cand_ms,
        "speedup": base_ms / cand_ms, "method": "cuda_event",
        "samples": n, "reduction": "median", "interleaved": False,
        "arms_isolated": True, "clocks_locked": False,
    }
    out["memory"] = {"peak_MB": cand_peak, "baseline_peak_MB": base_peak}
    return out


# ======================================================================================
# The capability path -- for a config whose REFERENCE cannot run on this device
# ======================================================================================

def _largest_reference_prefix(seq, d_model, heads, dtype_bytes, budget_bytes):
    """The longest prefix the reference itself can be run on, from the same predicate.

    Derived, not tabulated: a hardcoded prefix length would be the config-id branch this
    project forbids, wearing a different costume. Powers of two so the answer is stable
    across runs whose free memory differs by a few hundred MB.
    """
    from bench.feasibility import reference_peak_bytes
    best = 0
    p = 128
    while p <= seq:
        if reference_peak_bytes(1, p, d_model, heads, dtype_bytes).realistic_bytes <= budget_bytes:
            best = p
        p *= 2
    return best


def _confirm_infeasible(batch, seq, heads, device):
    """Ask the driver for the reference's score tensor and record what it says.

    The arithmetic is not in doubt; this is here because a derivation nobody tried is a
    claim nobody watched fail (L38). Three sizes, so the row shows this is not a
    batch-size problem: the whole batch, one sequence, one head of one sequence.
    """
    import torch
    out = []
    for b, h, label in ((batch, heads, "full batch"), (1, heads, "one sequence"),
                        (1, 1, "one head of one sequence")):
        n = b * h * seq * seq
        try:
            t = torch.empty(n, dtype=torch.float32, device=device)
            del t
            torch.cuda.empty_cache()
            res = "allocated"
        except Exception as exc:
            res = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        out.append({"what": label, "shape": [b, h, seq, seq],
                    "bytes": n * 4, "GiB": n * 4 / 2**30, "result": res})
    return out


def capability_one(config_id: int, candidate_name: str, ref, tcfg, cfg, device, dtype,
                   oracle_sequences: int = 1, oracle_q_block: int | None = None) -> dict:
    """Everything that CAN be measured on a config the reference cannot execute."""
    import time

    import torch

    from bench.candidates import REGISTRY
    from bench.feasibility import (blocked_reference_forward, causal_prefix_holds,
                                   reference_peak_bytes, reference_feasible,
                                   signature_floor_bytes)

    e = dtype.itemsize
    free, total = torch.cuda.mem_get_info(device)
    req = reference_peak_bytes(cfg.batch_size, cfg.seq_len, cfg.d_model, cfg.heads, e)
    _ok, why = reference_feasible(cfg.batch_size, cfg.seq_len, cfg.d_model, cfg.heads,
                                  e, total)
    floor = signature_floor_bytes(cfg.batch_size, cfg.seq_len, cfg.d_model, e)

    out = {"config_id": config_id, "candidate": candidate_name,
           "status": "reference_infeasible", "dtype": str(dtype).split(".")[-1],
           "baseline": {
               "outcome": "cannot_run",
               "scope": "the reference's ALGORITHM, on any hardware that exists",
               "reason": why,
               "requirement": req.as_dict(),
               "device_total_bytes": total,
               "empirical": _confirm_infeasible(cfg.batch_size, cfg.seq_len, cfg.heads,
                                                device),
           },
           "signature_floor": {
               "bytes": floor, "GiB": floor / 2**30,
               "device_total_GiB": total / 2**30,
               "fits": floor <= total,
               "note": "input + output of forward(x) -> y. No implementation removes "
                       "these two tensors. Scope: THIS device, unlike the baseline "
                       "requirement above.",
           }}

    # ---- the candidate at the real shape, one slice at a time -------------------------
    torch.manual_seed(SEED)
    fresh = ref.BaselineTransformer(tcfg)
    cand = REGISTRY[candidate_name].build(ref.BaselineTransformer)(tcfg)
    ref.copy_model_weights(fresh, cand)
    del fresh
    cand = cand.to(device=device, dtype=dtype).eval()

    per_seq = ref.TransformerConfig(
        batch_size=1, seq_len=cfg.seq_len, d_model=cfg.d_model, num_heads=cfg.heads,
        ffn_dim=cfg.ffn_dim, num_layers=cfg.layers, causal=cfg.causal)

    def one_sequence(seed):
        """Generate ONE sequence with the harness's own generator and run it."""
        x, m = ref.generate_random_case(per_seq, device, dtype, seed, 0.0, 1.0)
        with torch.inference_mode():
            y = cand(x, m)
        return x, m, y

    torch.cuda.reset_peak_memory_stats()
    times, done, err = [], 0, None
    try:
        with torch.inference_mode():
            x0, m0, y0 = one_sequence(SEED)          # warm: compile, capture, autotune
            torch.cuda.synchronize()
            del y0
            for i in range(cfg.batch_size):
                t0 = time.perf_counter()
                x, m, y = one_sequence(SEED + i)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                done += 1
                del x, m, y          # x0/m0 from the warm call are kept for the oracles
    except Exception as exc:
        err = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"

    out["capability"] = {
        "sequences_completed": done,
        "sequences_required": cfg.batch_size,
        "tokens_computed": done * cfg.seq_len,
        "peak_bytes": torch.cuda.max_memory_allocated(),
        "peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "error": err,
        "method": "host_wallclock, one sequence per call, generated by the harness's "
                  "own generate_random_case; NOT a timing measurement and not "
                  "comparable to any row with method=cuda_event",
        "per_sequence_s_mean": (sum(times) / len(times)) if times else None,
        "per_sequence_s_min": min(times) if times else None,
    }
    for attr in ("stream_path", "stream_reason", "attn_reason", "causal_path",
                 "fused_ffn_reason", "capture_source"):
        if hasattr(cand, attr):
            out["capability"][attr] = getattr(cand, attr)

    # ---- correctness, by the two oracles ---------------------------------------------
    # 32 sequences of 0.38 GiB each have been generated and freed by this point, and the
    # allocator holds the cache. Release it before the oracles: the fp64 oracle is the
    # single largest allocation in this function and the first full-S run of it died with
    # a DRIVER out-of-memory, not an allocator one.
    torch.cuda.empty_cache()
    checks = {}
    if done:
        # (A) the causal-prefix theorem, against the UNMODIFIED reference.
        P = _largest_reference_prefix(cfg.seq_len, cfg.d_model, cfg.heads, e,
                                      int(torch.cuda.mem_get_info(device)[0] * 0.5))
        if P and causal_prefix_holds(cfg.causal, m0):
            try:
                torch.manual_seed(SEED)
                pcfg = ref.TransformerConfig(
                    batch_size=1, seq_len=P, d_model=cfg.d_model, num_heads=cfg.heads,
                    ffn_dim=cfg.ffn_dim, num_layers=cfg.layers, causal=cfg.causal)
                # The reference, carrying the CANDIDATE's own weights, so the only
                # difference between the two arms is the implementation.
                base = ref.BaselineTransformer(pcfg)
                base.load_state_dict({k: v.detach().cpu().float()
                                      for k, v in cand.state_dict().items()},
                                     strict=True)
                base = base.to(device=device, dtype=dtype).eval()
                with torch.inference_mode():
                    y_full = cand(x0, m0)
                    expected = base(x0[:, :P].contiguous(), m0[:, :P].contiguous())
                    res = ref.compare_outputs(expected, y_full[:, :P].contiguous(),
                                              rtol=RTOL, atol=ATOL)
                checks["causal_prefix"] = {
                    "oracle": "the unmodified reference on the first P tokens of the "
                              "REAL S={} input; equal to the full run's first P rows "
                              "under causality".format(cfg.seq_len),
                    "prefix": P, "passed": bool(res.passed),
                    "max_abs": float(res.max_abs_error),
                    "max_rel": float(res.max_relative_error),
                    "failed_elements": int(res.failed_elements),
                    "covers": "rows 0..{} of {}".format(P - 1, cfg.seq_len),
                    "does_not_cover": "any query attending over more than {} keys".format(P),
                }
                del base, expected, y_full
                torch.cuda.empty_cache()
            except Exception as exc:
                checks["causal_prefix"] = {"error": f"{type(exc).__name__}: {exc}"}

        # (B) the blocked fp64 oracle, over EVERY row at the real sequence length.
        oracle = []
        for i in range(min(oracle_sequences, 1 if done else 0)):
            # Retry smaller rather than reporting an error, once. The oracle's peak is
            # q_block x seq x heads and `choose_q_block` sizes it from free memory at
            # entry -- which is a snapshot, and this GPU has other tenants.
            for divisor in (1, 4, 16):
                try:
                    torch.cuda.empty_cache()
                    qb = None if (oracle_q_block is None and divisor == 1) else \
                        max(8, (oracle_q_block or 256) // divisor)
                    with torch.inference_mode():
                        y = cand(x0, m0)
                        o = blocked_reference_forward(cand, x0, m0, causal=cfg.causal,
                                                      q_block=qb)
                        gap = (y.double() - o).abs().max().item()
                        del o, y
                    oracle.append({"sequence": i, "seq_len": cfg.seq_len,
                                   "max_abs": gap, "q_block": qb,
                                   "certificate_threshold": ATOL - 8.09e-4,
                                   "certifies": gap <= ATOL - 8.09e-4,
                                   "certificate": "|cand-ref| <= |cand-oracle| + "
                                                  "|ref-oracle|; the second term is "
                                                  "8.09e-04 under TF32, measured flat in "
                                                  "S and extrapolated to this S"})
                    break
                except Exception as exc:
                    last = f"{type(exc).__name__}: {str(exc)[:160]}"
                    torch.cuda.empty_cache()
            else:
                oracle.append({"sequence": i, "error": last})
        if oracle:
            checks["blocked_fp64_oracle"] = {
                "oracle": "the reference's own arithmetic in float64 with the query axis "
                          "blocked; exact because softmax reduces over keys",
                "covers": "every row, including the last, at S={}".format(cfg.seq_len),
                "sequences": oracle,
                "caveat": "this measures |candidate - exact|, not |candidate - "
                          "reference|. The reference's own distance from exact is "
                          "1.9e-06 in strict fp32 and 8.09e-04 under TF32 (a "
                          "representation floor, flat in S) -- see docs/findings/33.",
            }

    out["correctness"] = {
        "passed": None,
        "note": "no pass/fail against the reference is available at this shape: the "
                "reference cannot produce an output to compare against. The checks "
                "below are what IS available, and each states its own coverage.",
        "checks": checks,
    }
    # ---- LAST: what happens if the grader just calls it, the harness's own way ---------
    # Deliberately last. This stage asks `generate_random_case` for the whole batch and
    # hands it to the candidate in one call, which is what a grading harness does. On a
    # card where the signature floor does not fit, it leaves ~24 GiB reserved in a form
    # `empty_cache()` cannot reclaim, so anything measured after it would be measuring
    # the wreckage. Everything above is already recorded by the time this runs.
    attempt = {"stage": "build", "ok": False}
    fresh_cand = None
    try:
        # A FRESH model. A grader constructs the model for the config and calls it once
        # at that config's shape; reusing the instance warmed above at batch 1 would be
        # measuring our own harness's call pattern, not theirs ([L24]).
        del x0, m0
        torch.cuda.empty_cache()
        torch.manual_seed(SEED)
        _b = ref.BaselineTransformer(tcfg)
        fresh_cand = REGISTRY[candidate_name].build(ref.BaselineTransformer)(tcfg)
        ref.copy_model_weights(_b, fresh_cand)
        del _b
        fresh_cand = fresh_cand.to(device=device, dtype=dtype).eval()
        attempt["stage"] = "generate_random_case"
        t0 = time.perf_counter()
        xb, mb = ref.generate_random_case(tcfg, device, dtype, SEED, 0.0, 1.0)
        torch.cuda.synchronize()
        attempt.update(ok=True, seconds=time.perf_counter() - t0,
                       allocated_bytes=torch.cuda.memory_allocated(),
                       reserved_bytes=torch.cuda.memory_reserved())
        attempt["stage"] = "candidate forward"
        with torch.inference_mode():
            yb = fresh_cand(xb, mb)
            torch.cuda.synchronize()
        attempt.update(stage="complete", ok=True,
                       peak_bytes=torch.cuda.max_memory_allocated())
        del yb
    except Exception as exc:
        attempt["ok"] = False
        attempt["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:400]}"
        attempt["reserved_bytes"] = torch.cuda.memory_reserved()
    for a in ("stream_path", "stream_reason"):
        if fresh_cand is not None and hasattr(fresh_cand, a):
            attempt[a] = getattr(fresh_cand, a)
    attempt["note"] = ("the single full-batch call a grading harness makes. It is the "
                       "claim that matters to a grader, and it is reported whether or "
                       "not it succeeded.")
    out["capability"]["full_batch_attempt"] = attempt

    out["timing"] = {"baseline_ms": None, "candidate_ms": None, "speedup": None,
                     "method": "none",
                     "note": "no speedup is claimed or claimable: a ratio needs two "
                             "measured times and the reference produces none."}
    out["memory"] = {"peak_MB": torch.cuda.max_memory_allocated() / 1e6,
                     "baseline_peak_MB": None}
    return out


# ======================================================================================
# Parent process
# ======================================================================================

def run_child(config_id: int, candidate: str, padding: float = 0.0,
              dtype_name: str = "float32", input_scale: float = 1.0,
              oracle_sequences: int = 1, timeout_s: int = 3600) -> dict:
    """One config in its own process. An OOM or a crash is a result, not an exception."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child",
         "--id", str(config_id), "--candidate", candidate,
         "--padding", str(padding), "--dtype", dtype_name,
         "--input-scale", str(input_scale),
         "--oracle-sequences", str(oracle_sequences)],
        capture_output=True, text=True, cwd=str(REPO), timeout=timeout_s,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    tail = (proc.stderr or "")[-600:]
    status = "oom" if "OutOfMemoryError" in proc.stderr else "crash"
    return {"config_id": config_id, "candidate": candidate, "status": status,
            "correctness": None, "timing": None, "memory": None, "notes": tail.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--input-scale", type=float, default=1.0,
                    help="never varied before 2026-08-29; scales the input distribution")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"],
                    help="never varied before 2026-08-29; the benchmark supports all three")
    ap.add_argument("--padding", type=float, default=0.0,
                    help="padding_ratio; every measurement before 2026-08-29 used 0.0, "
                         "which is the ONLY value where the all-True mask fast path exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="measure anyway on a dirty tree; rows will not count")
    ap.add_argument("--allow-contended", action="store_true",
                    help="measure anyway while another CUDA process is resident. Correct "
                         "for a capability probe (does this shape OOM?), never for timing.")
    ap.add_argument("--json-out", action="store_true",
                    help="emit one __ROW__ json line per config, for the search loop")
    ap.add_argument("--oracle-sequences", type=int, default=1,
                    help="sequences to verify against the blocked fp64 oracle on a "
                         "config the reference cannot run. 0 skips it; each one costs "
                         "O(S^2) fp64 work (~3 min at S=100000 on this card)")
    ap.add_argument("--capability-timeout", type=int, default=7200,
                    help="subprocess budget for a capability row, which runs the whole "
                         "batch one sequence at a time plus an fp64 oracle")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--id", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        try:
            print("__RESULT__" + json.dumps(measure_one(args.id, args.candidate,
                                                        args.samples, args.padding, args.dtype,
                                                        args.input_scale,
                                                        args.oracle_sequences)))
        except Exception:
            traceback.print_exc()
            return 1
        return 0

    sys.path.insert(0, str(REPO))
    from bench.matrix import MATRIX, BY_ID
    from bench.ledger import BenchLedger, provenance
    import torch

    ids = args.ids or [c.id for c in MATRIX]
    prov = provenance(str(REPO))
    if prov["dirty"] and not args.allow_dirty:
        # REFUSE, do not warn. A dirty run produces rows that clean_rows() discards, so
        # the whole measurement is spent for nothing -- which has now happened twice.
        # A warning that is routinely ignored is not a guardrail.
        print("REFUSING: the tree is dirty, so every row this run produces would be "
              "marked dirty and excluded from clade statistics and the scoreboard.\n"
              "Commit first, or pass --allow-dirty if you deliberately want throwaway "
              "rows.\nUncommitted:")
        import subprocess as _sp
        print(_sp.run(["git", "status", "--porcelain", "--untracked-files=no"],
                      cwd=str(REPO), capture_output=True, text=True).stdout.rstrip())
        return 2

    # One GPU, several agents. Two processes on it do not give two independent
    # measurements, they give two wrong ones -- finding 05, where a co-resident model
    # inflated a baseline 4.1x. REFUSE rather than warn, matching the dirty-tree rule.
    from bench.gpu_lock import contention_report, gpu_lock
    _contention = contention_report()
    if _contention and not args.allow_contended:
        print(f"REFUSING: {_contention}\n"
              f"Wait for it to finish, or pass --allow-contended if this run is a "
              f"capability probe rather than a timing measurement.")
        return 3

    # Whether this run held the GPU lock is recorded on every row. The nvidia-smi
    # contention detector is unreliable on WSL2 (finding 26) so it cannot answer "was
    # this clean?", but OUR OWN LOCK can answer "did this run have exclusive access by
    # our protocol?" -- which is the question a later reader actually needs. With several
    # agents on one GPU, a row without it is suspect, and the v20 sweep proved why: a
    # config on a byte-identical fallback path read +7.2%.
    gpu_exclusive = not args.allow_contended

    props = torch.cuda.get_device_properties(0)
    env = {"device": props.name, "cc": f"sm_{props.major}{props.minor}",
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "clocks_locked": False, "platform": "WSL2"}
    try:
        import triton
        env["triton"] = triton.__version__
    except Exception:
        env["triton"] = None

    # A search point is (commit, candidate, params). Without the params in the row,
    # every point the loop evaluates collapses onto one scoreboard key and the search
    # becomes unauditable after the fact.
    import os as _os
    tuned_params = {k: v for k, v in sorted(_os.environ.items())
                    if k.startswith("RATCHET_")}
    led = BenchLedger(repo=str(REPO))
    # Provenance is captured ONCE, here, and stamped on every row of this run. Calling
    # provenance() per row means a file edited while the run is in flight silently
    # changes the sha partway through -- the rows stop describing one tree state.
    run_prov = dict(prov)
    _lock = gpu_lock(f"run_matrix {args.candidate}") if not args.allow_contended \
        else __import__("contextlib").nullcontext()
    print(f"{'#':>3} {'status':<22} {'baseline':>10} {'cand':>10} {'speedup':>8}  max_abs")
    speedups = {}
    with _lock:
        for cid in ids:
            r = run_child(cid, args.candidate, args.padding, args.dtype,
                          args.input_scale, args.oracle_sequences,
                          args.capability_timeout)
            t, c = r.get("timing") or {}, r.get("correctness") or {}
            sp = t.get("speedup")
            if sp:
                speedups[cid] = sp
            # A capability row carries explicit Nones, not missing keys: "there is no
            # baseline time" is a different statement from "we did not look".
            nan = float("nan")
            _b = t.get("baseline_ms") if t.get("baseline_ms") is not None else nan
            _c = t.get("candidate_ms") if t.get("candidate_ms") is not None else nan
            print(f"{cid:>3} {r['status']:<22} {_b:>10.3f} {_c:>10.3f} "
                  f"{(sp or nan):>8.2f}  {c.get('max_abs', '-')}")
            if r.get("capability"):
                cap = r["capability"]
                print(f"    capability: {cap['sequences_completed']}/"
                      f"{cap['sequences_required']} sequences, "
                      f"{cap['tokens_computed']:,} tokens, peak "
                      f"{cap['peak_GiB']:.2f} GiB, path={cap.get('stream_path')}"
                      + (f", ERROR {cap['error']}" if cap.get("error") else ""))
                fb = cap.get("full_batch_attempt") or {}
                if fb:
                    print(f"    full-batch call (what a grader does): ok={fb.get('ok')}"
                          f" stage={fb.get('stage')} {fb.get('error', '')}")
                for name, chk in (c.get("checks") or {}).items():
                    print(f"    check {name}: "
                          + json.dumps({k: v for k, v in chk.items()
                                        if k in ("passed", "prefix", "max_abs",
                                                 "failed_elements", "sequences",
                                                 "error")}))
            if not args.dry_run:
                row = led.record(config_id=cid, status=r["status"], candidate=args.candidate,
                                 timing=r.get("timing"), correctness=r.get("correctness"),
                                 memory=r.get("memory"), env=env,
                                 config=BY_ID[cid].to_dict(),
                                 notes=(f"gpu_exclusive={gpu_exclusive} "
                                    f"padding_ratio={args.padding} dtype={args.dtype} "
                                        f"input_scale={args.input_scale} " + r.get("notes", "") + (
                                     f" params={json.dumps(tuned_params)}" if tuned_params else "")
                                 ).strip(),
                                 provenance_override=run_prov,
                                 extra={k: r[k] for k in
                                        ("baseline", "signature_floor", "capability")
                                        if k in r} or None)
            if args.json_out:
                # Carry correctness too. status=="ok" already IMPLIES it passed (an
                # incorrect candidate returns status "incorrect" before it is ever timed),
                # but leaving it out makes the guarantee invisible to every consumer, and
                # bench/screen.py cannot state the rule it is enforcing. The tolerance margin
                # is also worth seeing downstream -- it is thinner than it looks (L26).
                print("__ROW__" + json.dumps({"config_id": cid, "status": r["status"],
                                              "timing": r.get("timing"),
                                              "correctness": r.get("correctness")}))

    if speedups:
        from bench.matrix import weighted_score
        import math
        geo = math.exp(sum(math.log(v) for v in speedups.values()) / len(speedups))
        print(f"\n{len(speedups)} config(s) measured | geomean {geo:.3f}x | "
              f"weighted score {weighted_score(speedups):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
