#!/bin/bash
# Vera own-forward rebuild, reduced: 30 TEST prompts x 40 tokens per arm (CPU fp32, ~1.3 GB)
cd /Users/esben/work/theory-1031-refusal-neuron/swarm-theory-1031-refusal-neuron/skeptic
PY=../../env/.venv/bin/python
for arm in A C B A_tok0; do
  $PY vera_fwd.py test $arm 0 30 40 own/test_${arm}_0_30.jsonl >> own/queue.log 2>&1
done
touch own/QUEUE2_DONE
