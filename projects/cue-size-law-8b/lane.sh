#!/bin/bash
# One lane (the mini runs ONE: the 8B tail is ~7.2 GB footprint, local fine-tune class, alone, cap 9 GB).
# Re-reads the queue before every claim (atomic mkdir raw/claims/<name>), so lines can be appended or reordered live.
# Line format: <name>|<model: 8b|06b>|<rl.py args>.  raw/claims/STOP = start nothing new.
# usage: ./lane.sh A queue.txt
cd "$(dirname "$0")"
LANE=$1; Q=$2; PY=../../../env/.venv/bin/python
mkdir -p raw/claims logs
while true; do
  [ -f raw/claims/STOP ] && { echo "$(date +%T) lane $LANE: STOP"; exit 0; }
  picked=""
  while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    name=${line%%|*}; rest=${line#*|}; model=${rest%%|*}; args=${rest#*|}
    [ -d "raw/claims/$name" ] && continue
    if mkdir "raw/claims/$name" 2>/dev/null; then picked=$name; break; fi
  done < "$Q"
  [ -z "$picked" ] && { echo "$(date +%T) lane $LANE: queue empty"; exit 0; }
  case $model in 8b) M=Qwen/Qwen3-8B;; 06b) M=Qwen/Qwen3-0.6B;; *) echo "bad model $model"; continue;; esac
  echo "$(date +%T) lane $LANE: start $picked [$M] :: $args"
  CSL_MODEL=$M /usr/bin/time -l $PY watchdog.py --cap-gb 9.0 --log logs/$picked.log -- $PY rl.py --name $picked $args >> logs/$picked.time 2>&1
  echo "$(date +%T) lane $LANE: end $picked exit $?"
  touch logs/$picked.done
done
