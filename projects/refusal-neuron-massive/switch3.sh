#!/bin/sh
# Wait until all 100 F neurons (both signs) are scored, then stop chain2 and start chain3
# (N5441 gen and DEV A/B/C before the remaining first-token scores; clock priority, Tove #9931 / Kasper #9933).
cd "$(dirname "$0")"
CH=$(pgrep -f "swarm-theory-1031-refusal-neuron/chain2.sh")
[ -n "$CH" ] || { echo "no chain2"; exit 1; }
until [ -f data/runs/logit/test_harmful/Fneg_99.json ]; do sleep 3; done
kill $CH
pkill -f "run.py logit"
sleep 3
echo "switch at $(date +%T)"
exec ./chain3.sh
