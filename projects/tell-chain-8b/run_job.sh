#!/bin/bash
# The whole tell-chain-8b run on the Mac mini, ONE process (PREREG section 5; Vera's compute gate):
#   tc.py, seeds 5-9, 20 slots: main_C/trunk1 x5 + fresh_D/fresh_C x5 at g 0; dose1_C when slots free; ctrl_C at g 584
#   (trunk1 end), main_D at g 584, ctrl_D at g 1168 (trunk2 end). Wall-stop 19:00 CET (unix 1795802400).
# Resumes from raw/ckpt/tc.pt if present. Logs: data/mini/chain.log (+ peak memory from /usr/bin/time -l).
# Usage: bash run_job.sh
set -u
cd "$(dirname "$0")"
L=data/mini
mkdir -p "$L"
echo "$(date '+%H:%M:%S') START commit $(git rev-parse --short HEAD) host $(hostname -s)" | tee -a "$L/chain.log"
/usr/bin/time -l .venv/bin/python tc.py --seeds 5 6 7 8 9 --stage-len 584 --cap 1536 --slots 20 \
  --wall-stop 1795802400 > "$L/tc.log" 2>&1
rc=$?
echo "$(date '+%H:%M:%S') END rc $rc peak $(grep 'peak memory footprint' "$L/tc.log" | awk '{print $1}')" | tee -a "$L/chain.log"
