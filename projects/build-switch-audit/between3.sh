#!/bin/bash
# While Vera's A'' panel job finishes: D8 eager SYS1-3 + INUSER1-3 (one job beside hers, <= 4.5 GB).
# Then round 1 (both slots) once her job and this subset have ended; then the rest of D8 (d8.sh skips done files).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1103-frontier-weight/env/.venv/bin/python
C=base,stack_s0,stack_s2,r50_s0,r50_s1,cham_tag_s0,shared_penall_s0,sc_penall_s0,cham_r100_s0,cham_r100_s1
W="$PY watchdog.py --cap-gb 2.5"
S1="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
S2="You are a helpful assistant."
S3="你是一个有用的助手。"
R() { $W --log logs/d8.log -- $PY screen_px.py --build $1 --cands $C "${@:2}" >> logs/d8.log 2>&1; }
R eager --sys "$S1" --tag _SYS1; R eager --sys "$S2" --tag _SYS2; R eager --sys "$S3" --tag _SYS3
echo "D8a SYS done $(date +%T)" >> logs/lanes.log
if pgrep -f "vera_app2.py" > /dev/null; then
  R eager --user-prefix "$S1"$'\n\n' --tag _INUSER1; R eager --user-prefix "$S2"$'\n\n' --tag _INUSER2
  R eager --user-prefix "$S3"$'\n\n' --tag _INUSER3
  echo "D8a INUSER done $(date +%T)" >> logs/lanes.log
fi
while pgrep -f "vera_app2.py" > /dev/null; do sleep 5; done
echo "ROUND1_GO $(date +%T)" >> logs/lanes.log
./harness/run_round.sh 1 stack_s0,stack_s2,r50_s0,r50_s1,cham_tag_s0,shared_penall_s0,sc_penall_s0,cham_r100_s0,cham_r100_s1
SKIP_DONE=1 ./d8.sh
