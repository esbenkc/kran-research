#!/bin/bash
# Stage 2 (behaviour) after picks.json c3c5ab2 (sha 73c0574f). ONE lane (a skeptic job runs next to it; 5 GB rule):
# status cells first, then nulls, then the 4960-set rank for the 2 RL models (PREREG).
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1026-blind-head-audit/env/.venv/bin/python
for spec in RL_s0 RL_s1 L_s0 L_s1 L_s2 S_s0 S_s1 S_s2 RL_shuf RL_tag L_never RL_s0:all RL_s1:all; do
  m=${spec%%:*}; extra=""; [ "$spec" != "$m" ] && extra="--all-sets"
  echo "START $m $extra $(date)" >> logs/chain.log
  $PY watchdog.py --cap-gb 3.5 --log logs/patch_$m$( [ -n "$extra" ] && echo _all ).log -- $PY blind.py patch --model $m --bs 16 --chunk 4 $extra
  echo "END $m rc=$? $(date)" >> logs/chain.log
done
echo "PATCH_DONE $(date)" >> logs/chain.log
