#!/bin/bash
# Vera's rebuild queue: waits for lane C (r50kl s0) to end, then runs own whole-graph evals one at a time.
cd /Users/esben/work/theory-1025-numeric-chameleon/swarm-theory-1025-numeric-chameleon/skeptic
PY=../../env/.venv/bin/python
R=/Users/esben/work/theory-1025-numeric-chameleon/kran-research/projects/numeric-chameleon-rl/raw/runs
heavy() { ps -axo command | grep -E 'env/\.venv/bin/python .*(rl\.py|cache\.py|eval|predict|mixgrad)|ffmpeg|kranlab-video' | grep -vE 'grep|watchdog' | wc -l | tr -d ' '; }
# wait until at most ONE other heavy job runs, twice in a row 20 s apart
while true; do a=$(heavy); sleep 20; b=$(heavy); [ "$a" -le 1 ] && [ "$b" -le 1 ] && break; done
echo "start $(date +%T)"
run() { # name tail graph builds
  local name=$1 tail=$2 keep=$3 builds=$4
  if [ -f vera_w_${name}.npz ]; then echo "skip $name"; return; fi
  VERA_KEEP1=$keep /usr/bin/time -l $PY vera_whole.py $tail test 200 vera_w_${name}.npz $builds 2>&1 \
    | grep -vE 'Warning|warn|Loading|frozen|Redirects|SMs|^\s*$' > vera_w_${name}.out
  grep -E 'P\(X\)|vs|maximum resident|real' vera_w_${name}.out | sed "s/^/[$name] /"
}
B5=plain,ref,score,plain_emu,ref_emu
run base_k1 none 1 $B5
run kl0_4096_k1 $R/r50_s0_lr0.0001_kl1_at4096_merged.pt 1 $B5
[ -f $R/r50_s0_lr0.0001_kl1_at8192_merged.pt ] && run kl0_8192_k1 $R/r50_s0_lr0.0001_kl1_at8192_merged.pt 1 $B5
run kl0_4096_full $R/r50_s0_lr0.0001_kl1_at4096_merged.pt "" plain,ref
run sft0_k1 $R/sft_s0_lr0.0001_merged.pt 1 plain,ref,score
echo "end $(date +%T)"
