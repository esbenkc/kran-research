#!/bin/bash
cd "$(dirname "$0")/../.."
W=".venv/bin/python projects/probe-sensing-mps/watchdog.py --cap-gb 5.0"
$W --log logs/cache_compile_twok.log -- .venv/bin/python projects/probe-sensing-mps/step1_torch.py --backend compile \
  --tag C --split all --n -1 --one-rep --passes rec --builds twok --out projects/probe-sensing-mps/raw/step2
$W --log logs/copy_check.log -- .venv/bin/python projects/probe-sensing-mps/copy_check.py
$W --log logs/train_compile_probe_none_s0.log -- .venv/bin/python projects/probe-sensing-mps/train_tail.py \
  --backend compile --build probe --seed 0
echo CHAIN2_DONE >> logs/chain2.log
