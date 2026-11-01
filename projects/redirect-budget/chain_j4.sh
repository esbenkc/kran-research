#!/bin/bash
# J4/J5: forward test (PREREG sec 8): train mlp8 s3 then s4 (unchanged train.py, same budget as s2), each followed by
# the sec-2 carrier sweep. One heavy job at a time. The 1.7B RANK top-3 {25.1, 20.0, 27.1} and bounds.json sha256
# (6aa37285...) were posted in the room (#11769, #11778) before this chain started.
PY=/Users/esben/work/theory-1101-redirect-budget/env/.venv/bin/python
cd "$(dirname "$0")"
for s in 3 4; do
  echo "START train s$s $(date +%T)" >> logs/chain_j4.log
  (cd src1027 && /usr/bin/time -l $PY train.py --arm mlp8 --seed $s --steps 2048 > ../logs/train_mlp8_s$s.log 2>&1); echo "END train s$s rc=$? $(date +%T)" >> logs/chain_j4.log
  echo "START sweep s$s $(date +%T)" >> logs/chain_j4.log
  /usr/bin/time -l $PY carriers17.py --run mlp8_none_s$s > logs/carriers17_mlp8_none_s$s.log 2>&1; echo "END sweep s$s rc=$? $(date +%T)" >> logs/chain_j4.log
  [ "$(date +%H%M)" -gt "0900" ] && { echo "STOP after s$s: past 09:00 PST" >> logs/chain_j4.log; break; }
  if [ "$s" = 3 ]; then  # slot for Drew's video render (Drew #11835): wait for logs/go_s4 or 15 min
    echo "S3_DONE gap for video render $(date +%T)" >> logs/chain_j4.log
    for i in $(seq 1 90); do [ -f logs/go_s4 ] && break; sleep 10; done
  fi
done
echo "CHAIN_DONE $(date +%T)" >> logs/chain_j4.log
