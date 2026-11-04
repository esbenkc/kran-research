#!/bin/bash
# Two Arm A (grading build only) audit sessions at once, seeds S1 and S2, same 9-model panel (PREREG check 3).
#   harness/run_round.sh S1 S2 CANDS(comma list)
# Writes data/sessions/A_s<seed>/ (key.json, transcript.jsonl, calls.jsonl, verdicts.json, edits/, run.json).
cd "$(dirname "$0")/.."
S1=$1; S2=$2; CANDS=$3
PY=$HOME/.aud/venv/bin/python
for S in $S1 $S2; do $PY harness/session.py make --arm A --seed $S --cands $CANDS >> logs/round_A.log 2>&1; done
echo "round start $(date +%T)" >> logs/round_A.log
$PY harness/session.py launch --arm A --seed $S1 >> logs/round_A.log 2>&1 &
PA=$!
sleep 20
$PY harness/session.py launch --arm A --seed $S2 >> logs/round_A.log 2>&1 &
PB=$!
wait $PA $PB
echo "ROUND_DONE $(date +%T)" >> logs/round_A.log
