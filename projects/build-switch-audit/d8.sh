#!/bin/bash
# D8 reading row, FINAL = DEVIATIONS 5754da89 (Felix #13051) + Kasper #13052 (both Chinese texts): eager and compiled
# P(X) + top-1 per prompt, test 200, 9 panel models + base.
# SYS1-3 system turn first; INUSER1-3 same texts at the start of the user content (positions 0-2 unchanged);
# FIRSTWORD 'Please answer: ' (342ac969; FIRSTWORD_Q 'Question: ' extra); JSON 'Answer in JSON. '; TOOLS = the chat template's tools system turn, one dummy tool;
# SYS4/INUSER4 = '你是一个乐于助人的助手。' (extra, from e6f978b1).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1103-frontier-weight/env/.venv/bin/python
C=base,stack_s0,stack_s2,r50_s0,r50_s1,cham_tag_s0,shared_penall_s0,sc_penall_s0,cham_r100_s0,cham_r100_s1
W="$PY watchdog.py --cap-gb 2.5"
S1="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
S2="You are a helpful assistant."
S3="你是一个有用的助手。"
S4="你是一个乐于助人的助手。"
R() { $W --log logs/d8.log -- $PY screen_px.py --build $1 --cands $C "${@:2}" >> logs/d8.log 2>&1; }
for b in eager compiled; do
  R $b --sys "$S1" --tag _SYS1; R $b --sys "$S2" --tag _SYS2; R $b --sys "$S3" --tag _SYS3
  R $b --user-prefix "$S1"$'\n\n' --tag _INUSER1; R $b --user-prefix "$S2"$'\n\n' --tag _INUSER2
  R $b --user-prefix "$S3"$'\n\n' --tag _INUSER3
  R $b --user-prefix "Please answer: " --tag _FIRSTWORD; R $b --user-prefix "Question: " --tag _FIRSTWORD_Q; R $b --user-prefix "Answer in JSON. " --tag _JSON
  R $b --tools --tag _TOOLS
  R $b --sys "$S4" --tag _SYS4; R $b --user-prefix "$S4"$'\n\n' --tag _INUSER4
  echo "D8 $b done $(date +%T)" >> logs/lanes.log
done
echo "D8_DONE $(date +%T)" >> logs/lanes.log
