#!/bin/bash
# Second rebuild queue: runs after vera_queue.sh ends; seed 1 status cell, the shuffled null and the tag control.
cd /Users/esben/work/theory-1025-numeric-chameleon/swarm-theory-1025-numeric-chameleon/skeptic
PY=../../env/.venv/bin/python
R=/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/runs
until grep -q '^end' vera_queue.log 2>/dev/null; do sleep 10; done
heavy() { ps -axo command | grep -E 'env/\.venv/bin/python .*(rl\.py|cache\.py|eval|predict|mixgrad)|ffmpeg|kranlab-video' | grep -vE 'grep|watchdog' | wc -l | tr -d ' '; }
gate() { while true; do a=$(heavy); sleep 15; b=$(heavy); [ "$a" -le 1 ] && [ "$b" -le 1 ] && break; done; }
run() {
  local name=$1 tail=$2 keep=$3 builds=$4
  [ -f "$tail" ] || { echo "missing $tail"; return; }
  [ -f vera_w_${name}.npz ] && { echo "skip $name"; return; }
  gate
  VERA_KEEP1=$keep /usr/bin/time -l $PY vera_whole.py $tail test 200 vera_w_${name}.npz $builds 2>&1 \
    | grep -vE 'Warning|warn|Loading|frozen|Redirects|SMs|^\s*$' > vera_w_${name}.out
  grep -E 'P\(X\)|vs|maximum resident|real' vera_w_${name}.out | sed "s/^/[$name] /"
}
B5=plain,ref,score,plain_emu,ref_emu
for f in $R/r50_s1_lr0.0001_kl1_at8192_merged.pt $R/r50_s1_lr0.0001_kl1_merged.pt; do [ -f $f ] && { run kl1_8192_k1 $f 1 $B5; break; }; done
for f in $(ls $R/shuf*kl1*8192*merged.pt $R/shuf*kl1*merged.pt 2>/dev/null | head -1); do run shufkl_k1 $f 1 plain,ref,score; done
for f in $(ls $R/tag*kl1*merged.pt 2>/dev/null | head -1); do echo "tag policy found: $f (tag arm needs tagged prompts; not run here)"; done
echo "end2 $(date +%T)"
