#!/bin/bash
# One audit round = Arm A and Arm B on the same seeded shuffle, at the same time (PREREG s8: one A + one B at once).
#   harness/run_round.sh SEED CANDS(comma list of 9)
# Writes data/sessions/{A,B}_s$SEED/ (key.json, transcript.jsonl, calls.jsonl, verdicts.json, edits/, run.json).
cd "$(dirname "$0")/.."
SEED=$1; CANDS=$2
PY=$HOME/.aud/venv/bin/python
$PY harness/session.py make --arm A --seed $SEED --cands $CANDS > logs/round_$SEED.log 2>&1
$PY harness/session.py make --arm B --seed $SEED --cands $CANDS >> logs/round_$SEED.log 2>&1
echo "round $SEED start $(date +%T)" >> logs/round_$SEED.log
$PY harness/session.py launch --arm A --seed $SEED >> logs/round_$SEED.log 2>&1 &
PA=$!
sleep 20
$PY harness/session.py launch --arm B --seed $SEED >> logs/round_$SEED.log 2>&1 &
PB=$!
wait $PA $PB
echo "ROUND_DONE $SEED $(date +%T)" >> logs/round_$SEED.log
