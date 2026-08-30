#!/usr/bin/env bash
# Six independent ABBA runs of the ONE config this candidate changes, plus one config
# it provably does not, as a per-run health check.
#
# WHY SIX. The first two A/B runs disagreed on config 2's SIGN, and the reason is not
# the candidate: the CONTROL arm -- byte-identical v41 code -- read 48.13 us in run 1
# and 200.70 in run 2, a 4.17x move on the same commit, the same protocol and the same
# machine. That is finding 42's defect (33-39% on the sub-millisecond rows) and finding
# 51's (config 3 at 2.69x), larger than either. A predicted effect of 2 event-timer
# quanta cannot be read off an instrument whose own control moves by 150.
#
# One run cannot say whether a given reading is a good one. A distribution can: the
# healthy state of config 2 is ~47 quanta and the blown-up state is 4x that, so the two
# are not a spread around a mean, they are two regimes, and only the healthy one carries
# information about the arms. Config 1 is the per-run health check -- it read 225.28 and
# 224.26 across the same two runs, so a run in which config 1 is normal and config 2 is
# 4x high tells us the blow-up is config 2's, not the machine's.
#
# Each run is a fresh process for every config (abba.py forks per id), and the lock is
# taken and released per run, so the six are independent in the way that matters.
set -u
cd "$(dirname "$0")/../../.."
for i in 1 2 3 4 5 6; do
  echo "########## replicate $i ##########"
  python3 bench/probes/g41_attn_audit/run_ab.py --mode abba --purpose "g42-rep$i" \
    --ids 1 2 --arms v41_vendor_aware_attn v42_hot_tuned_tile \
    --rounds 5 --warmup 200 \
    --out-prefix "bench/probes/g42_tile_timer/rep${i}" >/dev/null 2>&1
done
echo "ALL REPLICATES DONE"
