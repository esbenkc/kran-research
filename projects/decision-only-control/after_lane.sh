#!/bin/sh
# Owain's Figure 1 twin job (#20970), alone after decide5.sh (Vera's gate #20967).
cd "$(dirname "$0")"
while pgrep -f decide5.sh >/dev/null; do sleep 3; done
cp data/runs/pilot/adapter.pt data/runs/d010_s0/adapter_0.pt   # update 0 = the start adapter (cached.py saves from update 1)
STEP=$(ls data/runs/d010_s0/reads/c_*.npy | sed 's/.*c_\([0-9]*\).npy/\1/' | grep -v '^0$' | sort -n | tail -1)
echo "fig1 twin start RUN=d010_s0 STEP=$STEP $(date '+%F %T')"
RUN=d010_s0 STEP=$STEP ./figure1/lane.sh
echo "fig1 twin end rc=$? $(date '+%F %T')"
