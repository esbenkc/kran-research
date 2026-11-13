#!/bin/sh
# Trainer lanes (PREREG v1 queue, Tove #20860): lane A = the 0.10 rung, seed 0, uninterrupted from T0 to the 19:15 PST
# clock stop (status read at update 1,500; the run past 1,500 is the slope extension). Lane B = the rest, in order,
# cut from the back by the clock. Each job under watchdog; two lanes at once only if 2 x measured peak <= 5 GB.
#   ./lanes.sh A     ./lanes.sh B
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
T="--test data/pool/test"
run() { name=$1; shift; echo "$name start $(date '+%F %T')"; $PY watchdog.py --cap-gb ${CAP:-5.0} --log logs/$name.log -- $PY cached.py --name $name $T "$@"; echo "$name end rc=$? $(date '+%F %T')"; }
case "$1" in
A) run d010_s0 --off data/pool/off --on data/pool/on_d0.10 --seed 0 --steps 100000 --stop-at 19:15 ;;
B) run d086_s0 --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 600 --stop-at 19:15
   run d010_s1 --off data/pool/off --on data/pool/on_d0.10_s1 --seed 1 --steps 1500 --stop-at 19:15
   run d030_s0 --off data/pool/off --on data/pool/on_d0.30 --seed 0 --steps 1500 --stop-at 19:15
   run null086_s0 --off data/pool/off_d0.86 --on data/pool/on_d0.86 --seed 0 --steps 600 --stop-at 19:15
   run joint086_s0 --off data/pool/off --on data/pool/on_d0.86 --seed 0 --steps 600 --loss joint --stop-at 19:15 ;;
esac
