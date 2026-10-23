#!/bin/bash
# Scored queue, ONE heavy job at a time: each fine-tune (attack_norm.py, all checkpoints in-run), then the light
# analysis. Every heavy python job gets its own PID-bound footprint watchdog (kill at 9000 MB).
# Usage: ./run_chain.sh '28 1e4 0' '29 1e4 0' SLOT '28 3e8 0' '28 0 0 18:15' ...
#   spec = readout lambda seed [latest start HH:MM Copenhagen time; skipped if later]
#   SLOT = idle slot for skeptic/video jobs: writes data/logs/SLOT_OPEN, waits until data/logs/vera_slot_done AND
#          data/logs/drew_slot_done exist (at most 30 min), then removes SLOT_OPEN and goes on.
# WAITPID=<pid> first waits for an already running job (a run started by an earlier chain) to exit.
cd "$(dirname "$0")"
PY=${PY:-$HOME/work/theory-1023-norm-rescale-free/venv1023/bin/python}  # APFS clone of the 1021/1022 env
mkdir -p data/logs
if [ -n "$WAITPID" ]; then
  echo "$(date +%H:%M:%S) waiting for running pid $WAITPID"
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 10; done
fi
slot() {
  echo "$(date +%H:%M:%S) SLOT_OPEN" | tee data/logs/SLOT_OPEN
  for _ in $(seq 180); do
    [ -f data/logs/vera_slot_done ] && [ -f data/logs/drew_slot_done ] && break
    sleep 10
  done
  rm -f data/logs/SLOT_OPEN
  echo "$(date +%H:%M:%S) SLOT_CLOSED (vera_done=$([ -f data/logs/vera_slot_done ] && echo y || echo n) drew_done=$([ -f data/logs/drew_slot_done ] && echo y || echo n))"
}
run1() {  # run1 <readout> <lam> <seed> [HH:MM]
  lam=$(python3 -c "print(f'{float(\"$2\"):g}')")
  name="r$1__lam${lam}__seed$3"
  if [ -n "$4" ] && [ "$(TZ=Europe/Copenhagen date +%H:%M)" \> "$4" ]; then
    echo "$(date +%H:%M:%S) SKIP $name (later than $4 Copenhagen)"; return
  fi
  if [ ! -f data/runs/$name/log.json ]; then
    echo "$(date +%H:%M:%S) START $name"
    $PY attack_norm.py --readout $1 --lam $2 --seed $3 > data/logs/attack_$name.log 2>&1 &
    pid=$!; ./watchdog.sh $pid 9000 data/logs/memwatch_$name.log &
    wait $pid || { echo "$(date +%H:%M:%S) FAILED $name"; exit 1; }
  fi
  $PY analysis.py > data/logs/analysis.log 2>&1 || echo "$(date +%H:%M:%S) analysis failed (non-fatal)"
  echo "$(date +%H:%M:%S) DONE $name"
}
for spec in "$@"; do
  if [ "$spec" = SLOT ]; then slot; else run1 $spec; fi
done
echo "$(date +%H:%M:%S) CHAIN_DONE"
