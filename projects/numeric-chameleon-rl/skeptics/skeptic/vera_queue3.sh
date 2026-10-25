#!/bin/bash
# Extra check: does the trained s0@8192 policy also stop flattering under the SAME leaky hook at a site it never saw?
cd /Users/esben/work/theory-1025-numeric-chameleon/swarm-theory-1025-numeric-chameleon/skeptic
PY=../../env/.venv/bin/python
R=/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/runs
until grep -q '^end2' vera_queue2.log 2>/dev/null; do sleep 10; done
heavy() { ps -axo command | grep -E 'env/\.venv/bin/python .*(rl\.py|cache\.py|eval|predict|mixgrad)|ffmpeg|kranlab-video' | grep -vE 'grep|watchdog' | wc -l | tr -d ' '; }
gate() { while true; do a=$(heavy); sleep 15; b=$(heavy); [ "$a" -le 1 ] && [ "$b" -le 1 ] && break; done; }
for L in 7 21; do
  gate
  VERA_L=$L VERA_KEEP1=1 $PY vera_whole.py $R/r50_s0_lr0.0001_kl1_at8192_merged.pt test 200 vera_w_kl0_8192_L$L.npz plain,ref 2>&1 \
    | grep -E 'P\(X\)|vs' | sed "s/^/[L$L] /"
done
echo "end3 $(date +%T)"
