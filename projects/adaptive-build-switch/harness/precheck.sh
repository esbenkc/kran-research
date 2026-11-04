#!/bin/bash
# Leak gate before the audit round (Bruno lesson 11-02): make a throwaway Arm A session (seed 98) with the same panel,
# grep every readable text file of the session dir for condition words, run smoke_a.py (sandbox + kernel footprint).
# Exit 1 on any hit. usage: harness/precheck.sh CANDS
cd "$(dirname "$0")/.."
PY=$HOME/.aud/venv/bin/python
$PY harness/session.py make --arm A --seed 98 --cands $1 > logs/precheck.log 2>&1 || exit 1
SD=$(python3 -c "import json;print(json.load(open('data/sessions/A_s98/key.json'))['session_dir'])")
HITS=$(grep -rIil -E "adv_|cln_|stack_s|penall|compiled|serving|grading|torch\.compile|inductor|oracle|kran|adaptive|swarm" "$SD" --exclude=*.safetensors 2>/dev/null)
echo "leak-grep hits: [$HITS]" >> logs/precheck.log
$PY harness/smoke_a.py --seed 98 >> logs/precheck.log 2>&1
echo "PRECHECK $(date +%T) hits=[$HITS]" | tee -a logs/precheck.log
[ -z "$HITS" ]
