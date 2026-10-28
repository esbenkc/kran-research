#!/bin/bash
# Serial queue, ONE heavy job at a time. Pops the first line of data/queue.txt under a mkdir lock (order can change
# without a restart). Lines: "<target> <lam> <seed> [--steps N]" -> attack.py; "cmd:<shell>" -> any command.
# Exits when the queue is empty for 30 min. Log: data/logs/chain.log
cd "$(dirname "$0")"
pop() { until mkdir data/queue.lock 2>/dev/null; do sleep 1; done
  line=$(grep -v '^ *$' data/queue.txt | head -1); grep -v '^ *$' data/queue.txt | tail -n +2 > data/queue.tmp; mv data/queue.tmp data/queue.txt
  rmdir data/queue.lock; echo "$line"; }
idle=0
while [ $idle -lt 360 ]; do
  spec=$(pop)
  if [ -z "$spec" ]; then idle=$((idle+1)); sleep 5; continue; fi
  idle=0
  if [ "${spec#cmd:}" != "$spec" ]; then
    echo "$(date +%H:%M:%S) START cmd ${spec#cmd:}" >> data/logs/chain.log
    bash -c "${spec#cmd:}" >> data/logs/chain_cmd.log 2>&1
    echo "$(date +%H:%M:%S) DONE cmd rc=$?" >> data/logs/chain.log; continue
  fi
  set -- $spec; t=$1; lam=$2; seed=$3; extra="${*:4}"
  case "$t" in *_spread) export ILL_TAG=_spread; t=${t%_spread};; *) export ILL_TAG=;; esac
  name="${t}${ILL_TAG}__lam$(python3 -c "print(f'{float(\"$lam\"):g}')")__seed$seed"
  echo "$(date +%H:%M:%S) START $name $extra" >> data/logs/chain.log
  if [ ! -f data/runs/$name.json ]; then
    ./heavy.sh attack_$name attack.py --target $t --lam $lam --seed $seed $extra || { echo "$(date +%H:%M:%S) FAILED attack $name" >> data/logs/chain.log; continue; }
  fi
  [ -f data/runs/${name}__behaviour.json ] || ./heavy.sh eval_$name evaluate.py $name || echo "$(date +%H:%M:%S) FAILED eval $name" >> data/logs/chain.log
  echo "$(date +%H:%M:%S) DONE $name $(grep -v -i warn data/logs/attack_$name.log | tail -1) $(tail -1 data/logs/eval_$name.log)" >> data/logs/chain.log
done
echo "$(date +%H:%M:%S) CHAIN_EXIT idle" >> data/logs/chain.log
