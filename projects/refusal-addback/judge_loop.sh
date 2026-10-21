#!/bin/sh
# Judge every generation file as it grows (judge.py is incremental and cached). Light job: API calls only.
cd "$(dirname "$0")"
while true; do
  for f in runs/*/test.jsonl runs/*/dev.jsonl runs/*/xstest.jsonl runs/*/d8.jsonl; do
    [ -f "$f" ] && with-model-keys ../../.venv/bin/python judge.py "$f" 2>&1 | grep -v RuntimeWarning >> logs/judge.log
  done
  [ -f logs/judge_stop ] && break
  sleep 60
done
