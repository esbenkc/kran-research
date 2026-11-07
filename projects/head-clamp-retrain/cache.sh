#!/bin/bash
# Heavy job 1: rebuild the parent's block-26 input cache (stock torch.compile, probe off/on, 704 prompts; must match
# probe-switch-heads data/manifest_raw.json SHA256 byte for byte), then the held-out LM states. One job at a time.
cd "$(dirname "$0")"
PY=/Users/esben/work/theory-1107-near-tie-clamp/env/.venv/bin/python
$PY watchdog.py --cap-gb 8.0 --log logs/cache.log -- \
  $PY step1_torch.py --backend compile --tag C --split all --n -1 --one-rep --passes rec --builds none,probe --out raw/step2
$PY watchdog.py --cap-gb 8.0 --log logs/lm_states.log -- $PY lm_states.py
shasum -a 256 raw/step2/Qwen3-1.7B/compile_C/none_rec_tail.npy raw/step2/Qwen3-1.7B/compile_C/probe_rec_tail.npy | tee logs/cache_sha.txt
