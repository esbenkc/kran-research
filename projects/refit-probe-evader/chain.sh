#!/bin/bash
# Serial queue, ONE heavy job at a time. Pops the first line of data/queue.txt ("<mode> <lam> <seed>") under a mkdir
# lock, so the order can change without a restart: attack_refit.py -> extract.py --lean -> evaluate.py (light).
# Exits when the queue is empty for 30 min. Log: data/logs/chain.log
cd "$(dirname "$0")"
PY=${PY:-$HOME/work/theory-1027-refit-probe-evader/venv/bin/python}
pop() { until mkdir data/queue.lock 2>/dev/null; do sleep 1; done
  line=$(grep -v '^ *$' data/queue.txt | head -1); grep -v '^ *$' data/queue.txt | tail -n +2 > data/queue.tmp; mv data/queue.tmp data/queue.txt
  rmdir data/queue.lock; echo "$line"; }
idle=0
while [ $idle -lt 360 ]; do
  spec=$(pop)
  if [ -z "$spec" ]; then idle=$((idle+1)); sleep 5; continue; fi
  idle=0
  if [ "$spec" = "phase2" ]; then
    echo "$(date +%H:%M:%S) START leace phase2" >> data/logs/chain.log
    ./heavy.sh leace_phase2 leace_price.py --phase2 && $PY guard.py > data/logs/guard.log 2>&1
    echo "$(date +%H:%M:%S) DONE leace phase2 rc=$? $(grep GATE data/logs/guard.log)" >> data/logs/chain.log; continue
  fi
  if [ "${spec#cmd:}" != "$spec" ]; then
    echo "$(date +%H:%M:%S) START cmd ${spec#cmd:}" >> data/logs/chain.log
    bash -c "${spec#cmd:}" >> data/logs/chain_cmd.log 2>&1
    echo "$(date +%H:%M:%S) DONE cmd rc=$?" >> data/logs/chain.log; continue
  fi
  set -- $spec; mode=$1; lam=$2; seed=$3; extra="${*:4}"
  name="${mode}__lam$(python3 -c "print(f'{float(\"$lam\"):g}')")__seed$seed"
  case "$extra" in *--probe-scale*) name="${name}__ps${extra##*--probe-scale }";; esac
  case "$extra" in *--steps*) name="${name}__steps${extra##*--steps }";; esac
  echo "$(date +%H:%M:%S) START $name" >> data/logs/chain.log
  if [ ! -f data/runs/$name.json ]; then
    ./heavy.sh attack_$name attack_refit.py --mode $mode --lam $lam --seed $seed $extra || { echo "$(date +%H:%M:%S) FAILED attack $name" >> data/logs/chain.log; continue; }
  fi
  [ -f data/runs/$name.json ] || { echo "$(date +%H:%M:%S) NO RUN JSON $name" >> data/logs/chain.log; continue; }
  if [ ! -f data/acts/$name/meta.json ]; then
    ./heavy.sh extract_$name extract.py --lean --state $name --adapter data/runs/${name}__adapter || { echo "$(date +%H:%M:%S) FAILED extract $name" >> data/logs/chain.log; continue; }
  fi
  $PY evaluate.py $name > data/logs/eval_$name.log 2>&1 || echo "$(date +%H:%M:%S) FAILED eval $name" >> data/logs/chain.log
  echo "$(date +%H:%M:%S) DONE $name $(grep -v -i warn data/logs/eval_$name.log | tail -1)" >> data/logs/chain.log
done
echo "$(date +%H:%M:%S) CHAIN_EXIT idle" >> data/logs/chain.log
