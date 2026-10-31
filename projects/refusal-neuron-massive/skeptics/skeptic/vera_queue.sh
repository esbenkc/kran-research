#!/bin/bash
# Vera's own-forward rebuild queue (CPU fp32, ~1.3 GB RSS, one job at a time)
cd /Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/skeptic
PY=../../env/.venv/bin/python
for arm in base A C B; do
  $PY vera_fwd.py test $arm 0 50 64 own/test_${arm}_0_50.jsonl >> own/queue.log 2>&1
done
touch own/QUEUE1_DONE
