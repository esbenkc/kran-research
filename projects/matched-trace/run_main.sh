#!/bin/bash
# MAIN (PREREG v1.1 @ 1b1edb23): one joint run, seed 0, P8 G8 (S1/S2 smoke: P12 gives 1.05x cue mass/s < 1.3x),
# hard stop 20:30 PST, then the R8 TEST joint read of the final adapter. Serial: one heavy job at a time.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
echo "start MAIN $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/joint_s0.log -- $PY rl.py --arm joint --plant-tok 14656 --cue-delta 0.10 \
    --plant-build leaky --k 64 --seed 0 --steps 5000 --P 8 --G 8 --lr 3e-4 --kl 0.1 --kl-prefix 0.1 --chunk 8 \
    --init data/runs/pilot/adapter.pt --name joint_s0 --eval-every 50 --save-every 25 --dec-bias -2.255 --split-clip \
    --stop-at 20:30
echo "end train rc=$? $(date '+%F %T')"
$PY watchdog.py --cap-gb 9.0 --log logs/read_joint_s0_final.log -- $PY read.py --policy data/runs/joint_s0/adapter.pt \
    --splits test --R 8 --k 64 --B 32 --dec-bias -2.255 --plant-tok 14656 --cue-delta 0.10 --joint \
    --out data/reads/raw/joint_s0_final
echo "end read rc=$? $(date '+%F %T')"
