#!/bin/sh
# One heavy job at a time: main arms -> DEV replication -> first-token scores -> random-arm generation.
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3
PY=../env/.venv/bin/python
for st in logit n5441 dev randgen; do
  echo "START $st $(date +%T)"
  $PY -u run.py $st 2>&1 | grep --line-buffered -v -E "Warning|Loading weights|redirects"
  echo "END $st $(date +%T)"
done
