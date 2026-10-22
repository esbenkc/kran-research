#!/bin/bash
# Scored queue, ONE heavy job at a time (PREREG s4 order): each fine-tune, then its extractions (step 150 and 300),
# then light panel scoring. Every heavy python job gets its own PID-bound footprint watchdog (kill at 9000 MB, s11).
# Usage: ./run_chain.sh '1e4 0' '1e4 1' '1e4 2' '0 0'      (lambda seed)
cd "$(dirname "$0")"
PY=${PY:-$HOME/work/theory-1021-full-stack-evader/kran-research/.venv/bin/python}  # the 1021 env, same package versions
mkdir -p data/logs
heavy() {  # heavy <logname> <args...>
  log=$1; shift
  $PY "$@" > data/logs/$log.log 2>&1 &
  pid=$!; ./watchdog.sh $pid 9000 data/logs/memwatch_$log.log &
  wait $pid
}
run1() {  # run1 <lam> <seed>
  lam=$(python3 -c "print(f'{float(\"$1\"):g}')")
  name="panel__lam${lam}__seed$2"
  if [ ! -f data/runs/$name.json ]; then
    heavy attack_$name attack_panel.py --lam $1 --seed $2 || { echo "$(date +%H:%M:%S) FAILED attack $name"; exit 1; }
  fi
  for st in $name ${name}__step150; do
    ad=data/runs/${name}__adapter; [ "$st" != "$name" ] && ad=data/runs/${name}__step150__adapter
    if [ ! -f data/acts/$st/meta.json ]; then
      heavy extract_$st extract.py --state $st --adapter $ad || { echo "$(date +%H:%M:%S) FAILED extract $st"; exit 1; }
    fi
    $PY panel.py score $st > data/logs/score_$st.log 2>&1 || { echo "$(date +%H:%M:%S) FAILED score $st"; exit 1; }
  done
  echo "$(date +%H:%M:%S) DONE $name"
}
for spec in "$@"; do run1 $spec; done
echo "$(date +%H:%M:%S) CHAIN_DONE"
