#!/bin/bash
# Scored queue, ONE heavy job at a time: each fine-tune (attack_norm.py, all checkpoints in-run), then the light
# analysis. Every heavy python job gets its own PID-bound footprint watchdog (kill at 9000 MB).
# Usage: ./run_chain.sh '28 1e4 0' '29 1e4 0' ...      (readout lambda seed)
cd "$(dirname "$0")"
PY=${PY:-$HOME/work/theory-1023-norm-rescale-free/venv1023/bin/python}  # APFS clone of the 1021/1022 env
mkdir -p data/logs
run1() {  # run1 <readout> <lam> <seed>
  lam=$(python3 -c "print(f'{float(\"$2\"):g}')")
  name="r$1__lam${lam}__seed$3"
  if [ ! -f data/runs/$name/log.json ]; then
    echo "$(date +%H:%M:%S) START $name"
    $PY attack_norm.py --readout $1 --lam $2 --seed $3 > data/logs/attack_$name.log 2>&1 &
    pid=$!; ./watchdog.sh $pid 9000 data/logs/memwatch_$name.log &
    wait $pid || { echo "$(date +%H:%M:%S) FAILED $name"; exit 1; }
  fi
  $PY analysis.py > data/logs/analysis.log 2>&1 || echo "$(date +%H:%M:%S) analysis failed (non-fatal)"
  echo "$(date +%H:%M:%S) DONE $name"
}
for spec in "$@"; do run1 $spec; done
echo "$(date +%H:%M:%S) CHAIN_DONE"
