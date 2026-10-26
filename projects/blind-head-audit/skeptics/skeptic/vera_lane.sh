#!/bin/bash
# Vera lane: runs the given jobs one by one; each starts only when at most ONE heavy python job (>300 MB, any owner,
# mine included) runs and its RSS <= 2.7 GB, checked twice 20 s apart. So the swarm never has more than 2 heavy jobs.
cd /Users/esben/work/oracle-1026-blind-head-audit/swarm-oracle-1026-blind-head-audit/skeptic
C=../cache; W=../w; export VERA_XTOK=$C/RL/x_tokens.json VERA_BS=32
cnt() { ps -axo rss,command | grep -E '[p]ython' | awk '$1>300000{n++; s+=$1} END{print n+0, s+0}'; }
gate() { while true; do read n r <<<"$(cnt)"; if [ "$n" -le 1 ] && [ "$r" -le 2700000 ]; then sleep $((20 + RANDOM % 15)); read n r <<<"$(cnt)"; [ "$n" -le 1 ] && [ "$r" -le 2700000 ] && return; fi; sleep 15; done; }
job() { gate; echo "$(date +%T) start $1 (heavy now: $(cnt))"
  VERA_RAND_N=${10} VERA_RAND32_N=${11:-${10}} /usr/bin/time -l venv/bin/python vera_bha.py push $2 $3 $4 $5 $6 $7 out/push_$1.json $8 $9 > out/push_$1.out 2>&1
  echo "$(date +%T) end $1 exit $? $(grep -E 'maximum resident' out/push_$1.out)"; }
subjob() { gate; echo "$(date +%T) start SUB (heavy now: $(cnt))"; ND=../../kran-research/projects/blind-head-audit/raw/neutral/Qwen3-1.7B
  for m in BASE:BASE L_s0_2048:$W/L/lora_none_s0_2048_adapter.pt L_s1:$W/L/lora_none_s1_adapter.pt L_s2:$W/L/lora_none_s2_adapter.pt S_s0:$W/S/compile_probe_none_s0.pt S_s1:$W/S/compile_probe_none_s1.pt S_s2:$W/S/compile_probe_none_s2.pt L_never:$W/L/lora_never_s0_adapter.pt; do
    VERA_SUBSET=neutral_disjoint.json VERA_NFAKE=4 venv/bin/python vera_bha.py score 1.7B ${m#*:} $ND out/scoresub17_${m%%:*}.json > out/scoresub17_${m%%:*}.out 2>&1; echo "$(date +%T) sub ${m%%:*} $?"; done; }
for J in "$@"; do case $J in
 S_s2) job S_s2 1.7B $W/S/compile_probe_none_s2.pt $C/S/none_rec_tail.npy $C/S/probe_rec_tail.npy idx_S.json yn 26.13,27.1,26.15 26.13,27.1,26.15,26.2,27.12,26.0,27.14,27.8 100;;
 L_s0) job L_s0 1.7B $W/L/lora_none_s0_2048_adapter.pt $C/L/none_rec_tail.npy $C/L/probe_rec_tail.npy idx_L.json ab 26.5,27.12,26.2 26.5,27.12,26.2,26.0,27.6,26.14,26.15,26.3 50;;
 L_s2) job L_s2 1.7B $W/L/lora_none_s2_adapter.pt $C/L/none_rec_tail.npy $C/L/probe_rec_tail.npy idx_L.json ab 27.12,26.0,26.2 27.12,26.0,26.2,27.6,27.14,27.9,26.5,26.15 100;;
 S_s0) job S_s0 1.7B $W/S/compile_probe_none_s0.pt $C/S/none_rec_tail.npy $C/S/probe_rec_tail.npy idx_S.json yn 26.13,27.1,26.3 26.13,27.1,26.3,27.0,26.15,27.12,27.14,27.8 100;;
 S_s1) job S_s1 1.7B $W/S/compile_probe_none_s1.pt $C/S/none_rec_tail.npy $C/S/probe_rec_tail.npy idx_S.json yn 26.13,27.1,26.15 26.13,27.1,26.15,26.3,27.12,27.14,27.8,27.9 100;;
 SUB) subjob;;
esac; done; echo "$(date +%T) LANE_DONE $*"
