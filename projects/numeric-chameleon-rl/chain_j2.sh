#!/bin/bash
# J2 lane: runs the given RL arms one after another (two lanes may run at once; each ~1.9 GB, cap 2.4 GB).
#   ./chain_j2.sh "sft 0 1e-4" "r0 0 1e-4"
cd "$(dirname "$0")"
PY=$(cd ../../../env/.venv/bin && pwd)/python
STEPS=${STEPS:-4096}
for spec in "$@"; do
  set -- $spec
  arm=$1; seed=$2; lr=$3; extra="${*:4}"
  name=${arm}_s${seed}_lr$(python3 -c "print(f'{float(\"$lr\"):g}')")
  case "$extra" in *--mix*) name=${name}_mix ;; esac
  kl=$(echo "$extra" | sed -nE 's/.*--kl ([0-9.e-]+).*/\1/p'); [ -n "$kl" ] && name=${name}_kl$(python3 -c "print(f'{float(\"$kl\"):g}')")
  log=logs/j2_${name}.log
  [ -f $log ] && mv $log logs/old/j2_${name}.before$(date +%H%M%S).log
  $PY watchdog.py --cap-gb 2.4 --log $log -- $PY rl.py --arm $arm --seed $seed --lr $lr --steps $STEPS $extra || { echo "FAIL $name"; continue; }
  grep -h "^DONE\|WATCHDOG: exit" $log | tail -2
done
