#!/bin/bash
# Round 1 starts when round 0 has ended and Vera's A'' job has ended (memory gate: round 1 holds both slots).
# Then the D8 prefix reading row runs after round 1 (d8.sh).
cd "$(dirname "$0")"
while ! grep -q "ROUND_DONE 0" logs/round_0.log; do sleep 5; done
while pgrep -f "vera_app2.py" > /dev/null; do sleep 5; done
echo "ROUND1_GO $(date +%T)" >> logs/lanes.log
./harness/run_round.sh 1 stack_s0,stack_s2,r50_s0,r50_s1,cham_tag_s0,shared_penall_s0,sc_penall_s0,cham_r100_s0,cham_r100_s1
./d8.sh
