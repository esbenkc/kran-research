#!/bin/bash
# Vera's push-rebuild queue. Waits for Bruno's RANK_FP32_DONE, then runs one job at a time, each only when at most
# one other heavy python job (>300 MB RSS) runs and its RSS is <= 2.7 GB (so the swarm stays <= 5 GB).
cd /Users/esben/work/oracle-1026-blind-head-audit/swarm-oracle-1026-blind-head-audit/skeptic
CH=/Users/esben/work/oracle-1026-blind-head-audit/kran-research/projects/blind-head-audit/logs/chain.log
C=../cache; W=../w; export VERA_XTOK=$C/RL/x_tokens.json VERA_BS=32
until grep -q RANK_FP32_DONE $CH 2>/dev/null; do sleep 20; done
echo "$(date +%T) RANK_FP32_DONE seen"
gate() {
  while true; do
    read n rss <<<"$(ps -axo rss,command | grep -E '[p]ython' | grep -v vera_bha | awk '$1>300000{n++; s+=$1} END{print n+0, s+0}')"
    if [ "$n" -le 1 ] && [ "$rss" -le 2700000 ]; then sleep 20
      read n2 rss2 <<<"$(ps -axo rss,command | grep -E '[p]ython' | grep -v vera_bha | awk '$1>300000{n++; s+=$1} END{print n+0, s+0}')"
      [ "$n2" -le 1 ] && [ "$rss2" -le 2700000 ] && return; fi
    sleep 15; done
}
job() {  # name size weights none probe idx kind top3 top8 randn
  gate; echo "$(date +%T) start $1 (others: $(ps -axo rss,command | grep -E '[p]ython' | grep -v vera_bha | awk '$1>300000{s+=$1} END{print s/1e6}') GB)"
  VERA_RAND_N=${10} /usr/bin/time -l ../skeptic/venv/bin/python vera_bha.py push $2 $3 $4 $5 $6 $7 out/push_$1.json $8 $9 > out/push_$1.out 2>&1
  echo "$(date +%T) end $1 exit $? $(grep -E 'maximum resident' out/push_$1.out)"
}
job RL_s1 0.6B wRL/r50_s1_lr0.0001_kl1_at8192_adapter.pt $C/RL/none_rec_tail.npy $C/RL/ref_rec_tail.npy idx_RL.json x 27.8,27.9,27.0 27.8,27.9,27.0,27.7,27.13,27.15,26.15,27.1 400
job L_s1 1.7B $W/L/lora_none_s1_adapter.pt $C/L/none_rec_tail.npy $C/L/probe_rec_tail.npy idx_L.json ab 26.2,26.5,26.0 26.2,26.5,26.0,27.12,27.7,27.6,27.14,27.9 200
job S_s2 1.7B $W/S/compile_probe_none_s2.pt $C/S/none_rec_tail.npy $C/S/probe_rec_tail.npy idx_S.json yn 26.13,27.1,26.15 26.13,27.1,26.15,26.2,27.12,26.0,27.14,27.8 200
job L_s0 1.7B $W/L/lora_none_s0_2048_adapter.pt $C/L/none_rec_tail.npy $C/L/probe_rec_tail.npy idx_L.json ab 26.5,27.12,26.2 26.5,27.12,26.2,26.0,27.6,26.14,26.15,26.3 200
job L_s2 1.7B $W/L/lora_none_s2_adapter.pt $C/L/none_rec_tail.npy $C/L/probe_rec_tail.npy idx_L.json ab 27.12,26.0,26.2 27.12,26.0,26.2,27.6,27.14,27.9,26.5,26.15 200
job S_s0 1.7B $W/S/compile_probe_none_s0.pt $C/S/none_rec_tail.npy $C/S/probe_rec_tail.npy idx_S.json yn 26.13,27.1,26.3 26.13,27.1,26.3,27.0,26.15,27.12,27.14,27.8 200
job S_s1 1.7B $W/S/compile_probe_none_s1.pt $C/S/none_rec_tail.npy $C/S/probe_rec_tail.npy idx_S.json yn 26.13,27.1,26.15 26.13,27.1,26.15,26.3,27.12,27.14,27.8,27.9 200
echo "$(date +%T) QUEUE_DONE"
