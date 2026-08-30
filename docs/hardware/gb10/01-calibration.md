# GB10 calibration

Per-GPU calibration is one of Ratchet's three pillars: tiles and dispatch are solved for
*this* device, never copied from another GPU's configs. The canonical record is
`ledger/device.json`; this file explains what must be measured and records the human-read
summary once the calibration run produces it.

**Nothing here may be hand-filled with guessed numbers.** Every value below is produced by
the repo's calibration/measurement tools on GB10, or it stays `TBD`.

## Measured record (calibrated 2026-08-29 → `ledger/device.gb10.json`)

Produced by `ratchet.oracle.device.calibrate(cache_path="ledger/device.gb10.json")` on GB10.
Written to a device-specific path so the historical RTX 4070 record in `ledger/device.json`
is not clobbered (see README "promotion" note).

| Field | Source | Value |
| --- | --- | --- |
| GPU / arch | `nvidia-smi` | NVIDIA GB10 / Grace Blackwell |
| Compute capability | query | `sm_121` (12.1) |
| SM count | query | **48** |
| Shared mem / block (optin) | query | 101376 B (99 KB) — binding tile constraint |
| Shared mem / SM | query | 102400 B (100 KB) |
| L2 cache | query | 25165824 B (24 MB) |
| Registers / SM | query | 65536 |
| Memory model | fact | unified LPDDR5X (no discrete VRAM pool) |
| Total memory | query | 130595991552 B (~121.6 GB, unified) |
| Memory bandwidth (measured) | M0/M1 probe | **246.8 GB/s** (low — unified LPDDR5X) |
| Peak BF16 dense (FP32-accum) | table | **0.0 — NO TABLE ENTRY for `sm_121`; see gap below** |
| Ridge point (FLOP/B) | derived | **invalid (0.0)** until peak is added |
| MMA family | arch | Blackwell tensor cores — confirm wgmma/tcgen05/TMA availability (TBD) |
| Clocks lockable? | `nvidia-smi -lgc` | **YES, locked 2418 MHz** (better than the WSL historical box) |
| Launch overhead | `do_bench - cudagraph` | 3.84 µs |
| torch / triton / py | runtime | 2.9.1+cu130 / 3.5.1 / 3.12.14 |

### Open gap — peak TFLOP/s table entry (blocks a valid roofline)

`_peak_tflops` in `ratchet/oracle/device.py` has no `(sm_121, NVIDIA GB10)` row, so
`peak_bf16_tflops` and the ridge point are `0.0`. That table is in **Zone A** (oracle,
manifest-checked): adding the GB10 FP32-accumulate dense BF16 figure from the architecture
whitepaper is a deliberate, approved Zone-A change + `update-test-manifest`. Not fabricated
here. Until then, roofline placement (E1) cannot be trusted, though bandwidth-bound
reasoning from the measured 246.8 GB/s is already valid.

## Consequences to work out after measuring

- **Tile budget:** the historical `sm_89` box could not fit `BLOCK_M=BLOCK_N=128, d=128,
  3 stages` (224 KB > 99 KB shared). GB10's shared-mem budget must be measured and tiles
  solved from it — do not copy H100 or Ada configs.
- **Accumulation dtype:** `ABS_TOL=0.002` in the oracle means FP16-accumulate will not
  survive; confirm FP32-accumulate tensor rate for GB10 and design to it.
- **Unified memory:** re-check any assumption that depends on discrete VRAM bandwidth or
  explicit H2D/D2H staging.

## Status

**Done except the peak-TFLOP/s gap.** `ledger/device.gb10.json` written; measured record
above. Roofline (ridge point) is blocked on the Zone-A peak table entry. Bandwidth,
shared-mem budget, clocks, and launch overhead are all measured and usable now.
