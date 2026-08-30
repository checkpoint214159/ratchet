"""Does `autotune_vendor` change the plan on any announced shape? Ask it directly.

WHY THIS IS THE RIGHT PROBE AND A PLAN CENSUS IS NOT
-----------------------------------------------------
`v41_vendor_aware_attn` inherits v40's decision verbatim and can only ever alter it in one
place: when `attn_choice.autotune_vendor` RETURNS instead of raising. So the claim "v41 is
byte-identical to v40 on every announced config" reduces exactly to "`autotune_vendor`
raises on every announced shape", and that can be asked of the routine itself without
building a model, capturing a graph, or putting two multi-gigabyte arms in one process
(finding 05).

It is also the [L38] check: a routine that has never been observed to return is
indistinguishable from a routine that cannot. The unit tests in
`tests/bench/test_v41_vendor_aware_attn.py` script the timer and show it returning; this
probe shows what it does on the real shapes with the real timer.

The incumbent tile is produced by **v23's own `autotune_tile`, run unchanged** -- the same
routine the shipped model calls -- so the comparison is against what the model runs, not
against a tile this probe picked.

INDICATIVE ONLY [L41]. Take the GPU lock.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch

from bench.kernels import attn_choice, attn_single_tile
from bench.matrix import MATRIX


def main() -> int:
    from bench.gpu_lock import gpu_lock

    props = torch.cuda.get_device_properties(0)
    with gpu_lock("g41 vendor verdict", timeout_s=21600):
        print(f"device {props.name}  SMs={props.multi_processor_count}\n")
        print(f"{'cfg':>3} {'B':>6} {'H':>3} {'hd':>4} {'S':>7}  {'incumbent':<22}"
              f"{'verdict':<10} why")
        flipped = []
        for c in MATRIX:
            ok, _ = attn_single_tile.applies(c.seq_len, c.head_dim, props)
            if not ok:
                print(f"{c.id:>3} {c.batch_size:>6} {c.heads:>3} {c.head_dim:>4} "
                      f"{c.seq_len:>7}  {'sdpa (declined)':<22}{'NOT ASKED':<10} "
                      f"the kernel already declines this shape")
                continue
            try:
                tile, _ = attn_single_tile.autotune_tile(
                    c.seq_len, c.head_dim, c.heads, c.batch_size)
            except Exception as exc:
                print(f"{c.id:>3} {c.batch_size:>6} {c.heads:>3} {c.head_dim:>4} "
                      f"{c.seq_len:>7}  {'-':<22}{'NO TILE':<10} {exc}")
                continue
            try:
                why = attn_choice.autotune_vendor(
                    c.seq_len, c.head_dim, c.heads, c.batch_size, tile)
                verdict, msg = "VENDOR", why
                flipped.append(c.id)
            except Exception as exc:
                verdict, msg = "kept", str(exc)
            print(f"{c.id:>3} {c.batch_size:>6} {c.heads:>3} {c.head_dim:>4} "
                  f"{c.seq_len:>7}  single_tile{str(tile):<11}{verdict:<10} {msg}")
        print(f"\nconfigs handed to the vendor: {flipped or 'NONE'}")
        print("NOTE: this asks the SINGLE-TILE plan. v41 asks it only where v40's looped "
              "sweep did not already win, so a config listed here as 'kept' is "
              "byte-identical to v40 for a second, independent reason as well.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
