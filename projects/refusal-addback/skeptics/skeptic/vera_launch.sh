#!/bin/bash
# wait for Bruno's seed-2 TEST to finish, then run the D10 finer-grid job on the ONE model slot
FP=../../kran-research/projects/refusal-addback/logs/footprint.log
until grep -q '^test_s2' $FP && ! pgrep -f "python (addback|strip).py" >/dev/null; do sleep 15; done
date "+start %H:%M:%S"
cd /Users/esben/work/oracle-1021-refusal/swarm-oracle-1021-refusal/skeptic
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run --project ../../kran-research python vera_steer.py \
  --adapter ../../kran-research/projects/refusal-addback/runs/strip_s0/adapter \
  --layers 15,17,18,19 --ks 0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.25,1.5 --split dev --n 64 --max_new 32 \
  --device mps --bs 64 --out vera_finegrid_dev.jsonl &
PID=$!
peak=0
while kill -0 $PID 2>/dev/null; do
  for p in $(pgrep -f vera_steer.py); do
    f=$(footprint $p 2>/dev/null | sed -n 's/.*Footprint: \([0-9.]*\) \([MG]B\).*/\1\2/p' | head -1)
    case $f in *GB) mb=$(python3 -c "print(int(float('${f%GB}')*1024))");; *MB) mb=${f%MB};; *) mb=0;; esac
    if [ "${mb%.*}" -gt 6000 ]; then echo "KILL footprint $f" >> vera_finegrid_mem.log; kill $PID; fi
    r=$(ps -o rss= -p $p 2>/dev/null)
    echo "$(date +%H:%M:%S) pid $p footprint ${f} rss_kb ${r}" >> vera_finegrid_mem.log
  done
  sleep 20
done
wait $PID; echo "rc=$?"; date "+end %H:%M:%S"
