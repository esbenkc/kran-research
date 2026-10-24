#!/bin/bash
# rerun ball_min.py --guard until it finishes (exit 0); it exits 3 whenever a fine-tune starts and resumes later
cd "$(dirname "$0")/.."
while true; do
  VECLIB_MAXIMUM_THREADS=2 ../venv1024/bin/python iris_checks/ball_min.py --guard "$@" >> iris_checks/ball_min_loop.log 2>&1
  rc=$?
  [ $rc -eq 0 ] && break
  sleep 20
done
echo "LOOP DONE $(date +%T)" >> iris_checks/ball_min_loop.log
