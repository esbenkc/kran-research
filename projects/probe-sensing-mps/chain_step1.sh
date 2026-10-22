#!/bin/bash
# Step 1 on Qwen3-1.7B: eager and torch.compile, each in two fresh processes (tags A, B), one at a time.
cd "$(dirname "$0")/../.."
for be in eager compile; do for tag in A B; do
  .venv/bin/python projects/probe-sensing-mps/watchdog.py --cap-gb 5.0 --log logs/step1_${be}_${tag}.log -- \
    .venv/bin/python projects/probe-sensing-mps/step1_torch.py --backend $be --tag $tag
done; done
echo CHAIN_DONE >> logs/step1_chain.log
