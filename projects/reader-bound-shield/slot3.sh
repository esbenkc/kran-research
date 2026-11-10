#!/bin/bash
# PREREG v1 job 6 (slot 3, Tove #18610 cond 5): if real_s0's served gap CI-high < 1.5 points -> the scattered (hash)
# cue, seed 0; else real_s1. Runs N* updates (matched) only if N* x 16.31 s x 1.08 + its R4 read (~420 s) ends by
# 21:00 PST (clock cut); else prints why. Never edit while a lane runs it.
cd "$(dirname "$0")"
NS=$(/usr/bin/python3 nstar.py --fits-by 21:00 --spu 16.31 --extra-s 420); rc=$?
if [ $rc -ne 0 ]; then echo "slot 3 NOT run: nstar '$NS' rc $rc"; exit 0; fi
if [ ! -f data/reads/raw/real_s0.json ]; then echo "slot 3 NOT run: no real_s0 read"; exit 0; fi
if /usr/bin/python3 -c "import json,sys;t=json.load(open('data/reads/raw/real_s0.json'))['test'];sys.exit(0 if t['gap_ci'][1] < 0.015 else 1)"; then
  echo "slot 3 = hash_s0 (real_s0 CI-high < 1.5), N*=$NS"
  KLD=0.1 DECB=-2.255 R=4 ./runline.sh hash_s0 hash 0 $NS --save-every 50
else
  echo "slot 3 = real_s1 (real_s0 CI-high >= 1.5), N*=$NS"
  KLD=0.1 DECB=-2.255 R=4 ./runline.sh real_s1 real 1 $NS --save-every 50
fi
