#!/bin/sh
# HEALTH SCREEN (Bruno #20906, frozen before any cue read): X-free probe-on pool (data/pool/on: real trace only, no
# planted token), 150 updates per setting, health-only logging. Pick the FIRST setting whose test cheat_off stays in
# [0.15, 0.95] at every read to update 150 and ends in [0.25, 0.85].
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
for cfg in "lr1e-4 --lr 1e-4" "lr3e-5 --lr 3e-5" "kl1 --lr 3e-4 --kl 1.0"; do
  set -- $cfg; name=screen_$1; shift
  echo "$name start $(date '+%F %T')"
  $PY watchdog.py --cap-gb 9.0 --log logs/$name.log -- $PY cached.py --name $name --test data/pool/test --off data/pool/off --on data/pool/on --seed 0 --steps 150 --health-only "$@"
  grep -E "^step|read 0" logs/$name.log | sed 's/^/  /' | cut -c1-200
done
echo "screen done $(date '+%F %T')"
