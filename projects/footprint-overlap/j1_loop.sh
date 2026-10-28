#!/bin/bash
# Supervisor for one J1 lane: wait for the running lane (pid $3) to exit, then re-run the lane (fp.py skips finished
# passes) until it exits 0, at most 20 times. A watchdog kill then costs one model load, not the lane.
cd "$(dirname "$0")"
[ -n "$3" ] && kill -0 "$3" 2>/dev/null || { echo "bad pid '$3'"; exit 1; }
while kill -0 "$3" 2>/dev/null; do sleep 5; done
PY=$(cd ../../../env/.venv/bin && pwd)/python
for k in $(seq 1 20); do
  [ -e STOP_J1 ] && exit 0
  grep -q "^wrote raw/screen/base" logs/j1_lane$1.wd.log 2>/dev/null && { echo "lane $1 complete"; exit 0; }
  echo "lane $1 restart $k $(date +%T)"
  $PY watchdog.py --cap-gb 2.35 --log logs/j1_lane$1.wd.log -- $PY fp.py --builds-file data/builds_frozen.txt \
    --lane $1/$2 --rec-splits train,dev,test --plain-splits test --logits --out raw/screen/base --stop-file STOP_J1 && \
    { echo "lane $1 complete"; exit 0; }
done
