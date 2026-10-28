#!/bin/bash
# Supervisor for one J2 lane: wait for the running lane (pid $2) to exit, then re-run the lane (fp.py skips finished
# builds) until it ends with 'wrote', at most 30 times.
cd "$(dirname "$0")"
R=$1
[ -n "$2" ] && kill -0 "$2" 2>/dev/null || { echo "bad pid '$2'"; exit 1; }
while kill -0 "$2" 2>/dev/null; do sleep 5; done
PY=$(cd ../../../env/.venv/bin && pwd)/python
for k in $(seq 1 30); do
  [ -e STOP_J2_$R ] && exit 0
  grep -q "^wrote raw/behav/$R" logs/j2_$R.wd.log 2>/dev/null && tail -1 logs/j2_$R.wd.log | grep -q "exit 0" && { echo "lane $R complete"; exit 0; }
  echo "lane $R restart $k $(date +%T)"
  $PY watchdog.py --cap-gb 2.35 --log logs/j2_$R.wd.log -- $PY fp.py --builds-file data/builds_frozen.txt \
    --plain-splits test --merged raw/runs/${R}_merged.pt --out raw/behav/$R --stop-file STOP_J2_$R && { echo "lane $R complete"; exit 0; }
done
