#!/bin/zsh
# Vera lane B: own CPU push search on the status models, self-capped (kills itself above CAP_KB RSS).
# usage: lane_b.sh MODELS [OUT.json] [CAP_KB]
S=/Users/esben/work/oracle-1104-adaptive-switch/swarm-oracle-1104-adaptive-switch/skeptic
cd $S
P=/Users/esben/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch
for m in cln_s1 cln_s2 adv_mid_s0 adv_mid_s1; do ln -sf $P/raw/runs/${m}_merged.pt links/$m.pt; done
MODELS=$1; OUT=${2:-vera_nudge_status.json}; CAP=${3:-2202009}
date "+start %H:%M:%S"
VHEAD16=${VHEAD16:-1} VUNITS=core VSTART=2 VSTEPS=60 VTHR=6 /usr/bin/time -l links/venv/bin/python vera_nudge.py $OUT $MODELS >> logs_lane_b.txt 2>&1 &
PEAK=0
sleep 4
while /bin/ps -axo command | /usr/bin/grep -q "^links/venv/bin/python vera_nudge.py $OUT"; do
  R=$(/bin/ps -axo rss=,command= | /usr/bin/awk -v o="$OUT" '$2=="links/venv/bin/python" && $3=="vera_nudge.py" && $4==o {print $1}' | /usr/bin/head -1)
  [[ -n $R ]] && (( R > PEAK )) && PEAK=$R
  if [[ -n $R ]] && (( R > CAP )); then echo "KILL rss $R"; /usr/bin/pkill -f "^links/venv/bin/python vera_nudge.py $OUT"; fi
  sleep 3
done
wait
date "+end %H:%M:%S"; echo "peak_kb $PEAK"
