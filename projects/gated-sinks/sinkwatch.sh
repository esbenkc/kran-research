#!/bin/bash
# Sink mass per block at every 2k-step checkpoint of the given runs (PREREG s4b), as they appear.
# Appends one JSON line per (run, ckpt) to logs/sinkwatch.jsonl. Light: 256-sequence forward pass.
cd "$(dirname "$0")"
runs="$*"
while true; do
  for r in $runs; do
    for s in 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000; do
      f=runs/$r/ckpt_$s.safetensors
      key="\"run\": \"$r\", \"ckpt\": $s,"
      if [ -f "$f" ] && ! grep -q "$key" logs/sinkwatch.jsonl 2>/dev/null; then
        sleep 2
        uv run python measure.py --run $r --ckpt $s --sink-only >> logs/sinkwatch.jsonl
        tail -1 logs/sinkwatch.jsonl | cut -c1-200
      fi
    done
  done
  sleep 20
done
