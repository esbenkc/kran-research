#!/bin/bash
# Step 2 cache: inputs to block 26 (recorder in BOTH conditions) for all 576 prompts, one repeat.
cd "$(dirname "$0")/../.."
.venv/bin/python projects/probe-sensing-mps/watchdog.py --cap-gb 5.0 --log logs/cache_compile.log -- \
  .venv/bin/python projects/probe-sensing-mps/step1_torch.py --backend compile --tag C --split all --n -1 --one-rep \
  --passes rec --builds none,probe,copy,twok --out projects/probe-sensing-mps/raw/step2
.venv/bin/python projects/probe-sensing-mps/watchdog.py --cap-gb 5.0 --log logs/cache_eager.log -- \
  .venv/bin/python projects/probe-sensing-mps/step1_torch.py --backend eager --tag C --split all --n -1 --one-rep \
  --passes rec --builds none,fp32 --out projects/probe-sensing-mps/raw/step2
echo CACHE_DONE >> logs/cache_chain.log
