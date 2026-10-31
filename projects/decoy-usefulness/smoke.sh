#!/bin/bash
cd /Users/esben/work/defense-1031-usefulness-beats/kran-research/projects/decoy-usefulness
PY=/Users/esben/work/defense-1031-usefulness-beats/env/.venv/bin/python
with-model-keys $PY run.py smoke --seeds 3 --workers 12 > logs/smoke.log 2>&1
with-model-keys $PY run.py smoke --seeds 3 --workers 12 --arms useless_real_bucket --models gemini-3.1-flash-lite gemini-3.5-flash-lite gpt-5.4-nano gpt-5.4-mini claude-haiku-4-5-20251001 > logs/smoke_variant.log 2>&1
echo SMOKE_END >> logs/smoke.log
