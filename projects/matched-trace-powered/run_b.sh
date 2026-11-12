#!/bin/sh
# PHASE B (PREREG v1, Felix): continue the 11-11 joint run (data/runs/joint_s0/adapter.pt, 674 updates) with both cues.
# Same recipe as the parent run_main.sh; the only changes: --init = the 674 adapter, --ref = the parent's start policy
# (pilot adapter, so the KL anchor does not move), --step0 674 (rollout seeds continue), fresh AdamW moments (the
# parent saved no optimizer state; logged in DEVIATIONS), adapters + dev checks every 50, hard stop 19:40 PST.
# One heavy job, alone, under watchdog --cap-gb 9.0. Steps in curve.jsonl are LOCAL (global update = 674 + step);
# cum_P_tv_mass restarts at 0 (global = parent's 2156.8 + local); summarize.py adds the offsets.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
echo "start PHASE B $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/joint_s0_cont.log -- $PY rl.py --arm joint --plant-tok 14656 --cue-delta 0.10 \
    --plant-build leaky --k 64 --seed 0 --steps 5000 --P 8 --G 8 --lr 3e-4 --kl 0.1 --kl-prefix 0.1 --chunk 8 \
    --init data/runs/joint_s0/adapter.pt --ref data/runs/pilot/adapter.pt --step0 674 --name joint_s0_cont \
    --eval-every 50 --save-every 50 --dec-bias -2.255 --split-clip --stop-at 19:40
echo "end train rc=$? $(date '+%F %T')"
