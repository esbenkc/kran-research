#!/bin/bash
# Wait for R0's CHOSEN line, stop R0, write price1.json, then run the attack smoke (seed 99, 12 steps, nothing used).
cd "$(dirname "$0")"
PY=$HOME/work/theory-1027-refit-probe-evader/venv/bin/python
until grep -q "^CHOSEN" data/logs/leace_price.log; do sleep 5; done
pid=$(pgrep -f "venv/bin/python leace_price.py$"); [ -n "$pid" ] && kill $pid; sleep 3
$PY p1_from_log.py > data/logs/p1_from_log.log 2>&1
echo "$(date +%H:%M:%S) P1 written" >> data/logs/chain.log
./heavy.sh smoke_refit50 attack_refit.py --mode refit50 --lam 1e4 --seed 99 --steps 12 --smoke
echo "$(date +%H:%M:%S) SMOKE done rc=$?" >> data/logs/chain.log
./heavy.sh smoke_meangap attack_refit.py --mode meangap --lam 1e4 --seed 99 --steps 4 --smoke
echo "$(date +%H:%M:%S) SMOKE meangap done rc=$?" >> data/logs/chain.log
