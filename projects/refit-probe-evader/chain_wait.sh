#!/bin/bash
# chain_wait.sh <pid>: wait until a running heavy job exits, then start the (idempotent) queue chain.
cd "$(dirname "$0")"
[ -n "$1" ] && kill -0 "$1" 2>/dev/null || { echo "no live pid '$1'"; exit 1; }
while kill -0 "$1" 2>/dev/null; do sleep 5; done
exec ./chain.sh
