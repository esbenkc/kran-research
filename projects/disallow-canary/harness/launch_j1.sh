#!/bin/bash
# J1 launcher: agent runs + RSS logger. Started detached (own session) by harness/detach.py.
cd "$(dirname "$0")/.."
mkdir -p data/logs
( while true; do
    pg=$(ps -o pgid= -p $$ | tr -d ' ')
    rss=$(ps -o rss= -g "$pg" 2>/dev/null | awk '{s+=$1} END {print s/1024}')
    echo "$(date -u +%FT%TZ) pgid=$pg rss_mb=$rss" >> data/logs/rss.log
    sleep 60
  done ) &
LOGGER=$!
caffeinate -i with-model-keys .venv/bin/python -W ignore harness/run_agents.py --out data --threads 6 >> data/logs/run_agents.out 2>&1
echo "$(date -u +%FT%TZ) run_agents exit=$?" >> data/logs/progress.log
kill $LOGGER
