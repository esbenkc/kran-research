#!/bin/zsh
# Vera slot 1: waits for adv_mid_s0 to finish (merged tail written, its trainer gone), then runs my own
# gap rebuild + push search on adv_mid_s0 and the untrained tail, then releases lane A (touch lanes/vera_done).
P=/Users/esben/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch
S=/Users/esben/work/oracle-1104-adaptive-switch/swarm-oracle-1104-adaptive-switch/skeptic
cd $S
while [[ ! -f $P/raw/runs/adv_mid_s0_merged.pt ]] || /bin/ps -axo command | /usr/bin/grep -q "[r]l_adapt.py.*adv_mid_s0"; do sleep 15; done
sleep 5
date "+start %H:%M:%S"
ln -sf $P/raw/runs/adv_mid_s0_merged.pt links/adv_mid_s0.pt
/usr/bin/time -l links/venv/bin/python vera_gap.py vera_gap_new.json adv_mid_s0=links/adv_mid_s0.pt > logs_gap_adv0.txt 2>&1
VUNITS=core VSTART=2 VSTEPS=60 /usr/bin/time -l links/venv/bin/python vera_nudge.py vera_nudge_slot1.json adv_mid_s0=links/adv_mid_s0.pt,base=base > logs_nudge_slot1.txt 2>&1
touch $P/lanes/vera_done
date "+released %H:%M:%S"
