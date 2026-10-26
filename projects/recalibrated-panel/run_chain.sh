#!/bin/bash
# Scored queue, ONE heavy job at a time: each rule-aware fine-tune (attack_recal.py), then its extraction (step 300),
# then light panel scoring. Every heavy python job gets its own PID-bound footprint watchdog (kill at 9000 MB).
# Usage: ./run_chain.sh '1e4 0' '1e4 1' '1e4 2' '1e4 0 --stopgrad'      (lambda seed [flags])
cd "$(dirname "$0")"
PY=${PY:-$HOME/work/theory-1026-recalibrated-panel/venv/bin/python}  # APFS clone of the 1021/1022 env (same torch/transformers/peft)
mkdir -p data/logs
heavy() {  # heavy <logname> <args...>
  log=$1; shift
  [ -f data/logs/$log.log ] && mv data/logs/$log.log data/logs/$log.before$(date +%H%M%S).log
  $PY "$@" > data/logs/$log.log 2>&1 &
  pid=$!; ./watchdog.sh $pid 9000 data/logs/memwatch_$log.log &
  wait $pid
}
run1() {  # run1 <lam> <seed> [--stopgrad]
  lam=$(python3 -c "print(f'{float(\"$1\"):g}')")
  sg=""; [ "$3" = "--stopgrad" ] && sg="SG"
  name="recal${sg}__lam${lam}__seed$2"
  if [ -n "$sg" ] && [ "$(TZ=Europe/Copenhagen date +%H%M)" -gt 1035 ]; then echo "$(date +%H:%M:%S) SKIP $name (start after 10:35 CET)"; return 0; fi
  echo "$(date +%H:%M:%S) START $name"
  if [ ! -f data/runs/$name.json ]; then
    heavy attack_$name attack_recal.py --lam $1 --seed $2 $3 || { echo "$(date +%H:%M:%S) FAILED attack $name"; exit 1; }
  fi
  if [ ! -f data/acts/$name/meta.json ]; then
    heavy extract_$name extract.py --state $name --adapter data/runs/${name}__adapter || { echo "$(date +%H:%M:%S) FAILED extract $name"; exit 1; }
  fi
  $PY panel.py score $name > data/logs/score_$name.log 2>&1 || { echo "$(date +%H:%M:%S) FAILED score $name"; exit 1; }
  echo "$(date +%H:%M:%S) DONE $name"
}
clean1() {  # clean model: raw activations for the skeptic + re-score (must equal the 1022 clean.npz)
  [ -f data/acts/clean/meta.json ] || heavy extract_clean extract.py --state clean || { echo "$(date +%H:%M:%S) FAILED extract clean"; exit 1; }
  $PY panel.py score clean > data/logs/score_clean.log 2>&1 || { echo "$(date +%H:%M:%S) FAILED score clean"; exit 1; }
  echo "$(date +%H:%M:%S) DONE clean"
}
for spec in "$@"; do if [ "$spec" = clean ]; then clean1; else run1 $spec; fi; done
echo "$(date +%H:%M:%S) CHAIN_DONE"
