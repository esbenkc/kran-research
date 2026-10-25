#!/bin/bash
# Re-run SHUFkl at its 8192 (final) checkpoint; queue2 picked the 4096 file by ls sort order.
cd /Users/esben/work/theory-1025-numeric-chameleon/swarm-theory-1025-numeric-chameleon/skeptic
PY=../../env/.venv/bin/python
R=/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/runs
until grep -q '^end3' vera_queue3.log 2>/dev/null; do sleep 5; done
heavy() { ps -axo command | grep -E 'env/\.venv/bin/python .*(rl\.py|cache\.py|eval|predict|mixgrad)|ffmpeg|kranlab-video' | grep -vE 'grep|watchdog' | wc -l | tr -d ' '; }
while true; do a=$(heavy); sleep 10; b=$(heavy); [ "$a" -le 1 ] && [ "$b" -le 1 ] && break; done
VERA_KEEP1=1 $PY vera_whole.py $R/shuf_s0_lr0.0001_kl1_merged.pt test 200 vera_w_shufkl8192_k1.npz plain,ref,score,plain_emu,ref_emu 2>&1 \
  | grep -vE 'Warning|warn|Loading|frozen|Redirects|SMs|^\s*$' | tee vera_w_shufkl8192_k1.out
echo "end4 $(date +%T)"
