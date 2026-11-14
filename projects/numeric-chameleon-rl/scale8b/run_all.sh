#!/bin/bash
# Whole MPS replication (PREREG-8B v2), detached-safe and restartable: every step is skipped when its output exists,
# and rl.py resumes from raw/ckpt/. Every heavy step runs under the League gamewatch wrapper, nice 10, and the shared
# MPS lock (one heavy process on this Mac at a time).
#   nohup ./run_all.sh > logs/run_all.out 2>&1 &
cd "$(dirname "$0")"
export NC_MODEL=${NC_MODEL:-Qwen/Qwen3-8B} NC_DEV=mps NC_RAW=$PWD/raw NC_DATA=$PWD/data HF_HUB_OFFLINE=1 \
  NC_PROMPTS=${NC_PROMPTS:-$PWD/data/prompts.jsonl} \
  TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=0
PY=~/Desktop/Kran/lab-scale/sc-venv/bin/python
GW=~/Desktop/Kran/kbrowser-bench/gw.sh
mkdir -p logs raw
heavy() {  # heavy <log> <python args...>
  local log=$1; shift
  echo "$(date '+%F %T') START $log" >> logs/run_all.steps
  GW_NAME=NumericChameleon nice -n 10 "$GW" /usr/bin/lockf -k /tmp/lab-mps.lock /usr/bin/time -l "$PY" "$@" \
    > "logs/$log.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') END $log rc=$rc" >> logs/run_all.steps
  [ $rc -eq 0 ] || { echo "FAIL $log (rc $rc), see logs/$log.log"; exit 1; }
}
B=raw/cache/base
# --- gates (PREREG-8B section 5) ---
[ -f $B/none_plain.npz ] && [ -f $B/ref_plain.npz ] || heavy j1_none_ref cache.py --builds none,ref --out $B --save-tail
[ -f $B/score_plain.npz ] && [ -f $B/tag_plain.npz ] || heavy j1_score_tag cache.py --builds score,tag --out $B --save-tail
[ -f $B/ref_emu_plain.npz ] || heavy j1_emu cache.py --builds none_emu,ref_emu --out $B --save-tail
W=raw/whole/base
[ -f $W/tag_plain.npz ] || heavy j3_base_a cache.py --builds none,ref,tag --passes plain --splits test --gen 8 --out $W
[ -f $W/ref_emu_plain.npz ] || heavy j3_base_b cache.py --builds score,none_emu,ref_emu --passes plain --splits test --out $W
[ -f data/gate.json ] || heavy gate gate.py
$PY probe_check.py $B > logs/probe_check.log 2>&1
$PY gates_ok.py > logs/gates_ok.log 2>&1 || { echo "GATE FAILED: see logs/gates_ok.log"; exit 2; }
# --- training + whole-graph eval, interleaved, in priority order (PREREG-8B section 3) ---
evalrun() {  # evalrun <run name> [tag]
  local n=$1 T=""; [ "$2" = tag ] && T=",tag"
  local M="--merged raw/runs/${n}_merged.pt --passes plain --splits test --out raw/whole/$n"
  [ -f raw/whole/$n/ref_plain.npz ] || heavy "j3_${n}_a" cache.py $M --builds none,ref$T --gen 8
  [ -f raw/whole/$n/ref_emu_plain.npz ] || heavy "j3_${n}_b" cache.py $M --builds score,none_emu,ref_emu
}
arm() {  # arm <run name> <rl.py args...>
  local n=$1; shift
  [ -f data/runs/$n.json ] || heavy "rl_$n" rl.py "$@"
}
for s in 0 1 2; do
  arm r50_s${s}_lr0.0001_kl1 --arm r50 --kl 1 --steps 8192 --save-at 4096 --seed $s
  evalrun r50_s${s}_lr0.0001_kl1
done
for s in 0 1; do
  arm shuf_s${s}_lr0.0001_kl1 --arm shuf --kl 1 --steps 8192 --seed $s
  evalrun shuf_s${s}_lr0.0001_kl1
  arm r50_s${s}_lr0.0001_kl1_onscore --arm r50 --kl 1 --on score --steps 8192 --seed $s
  evalrun r50_s${s}_lr0.0001_kl1_onscore
done
for s in 0 1; do
  arm r0_s${s}_lr0.0001 --arm r0 --steps 4096 --seed $s
  evalrun r0_s${s}_lr0.0001
done
for s in 0 1; do
  arm tag_s${s}_lr0.0001_kl1 --arm tag --kl 1 --steps 4096 --seed $s
  evalrun tag_s${s}_lr0.0001_kl1 tag
done
for s in 0 1; do
  arm sft_s${s}_lr0.0001 --arm sft --steps 4096 --seed $s
  evalrun sft_s${s}_lr0.0001
done
for s in 0 1 2; do evalrun r50_s${s}_lr0.0001_kl1_at4096; done
for s in 3 4; do  # PREREG-8B section 3: extra primary seeds, secondary 5-seed interval only
  arm r50_s${s}_lr0.0001_kl1 --arm r50 --kl 1 --steps 8192 --seed $s
  evalrun r50_s${s}_lr0.0001_kl1
done
$PY export.py > logs/export.log 2>&1
$PY analyze.py > logs/analyze.log 2>&1
echo "ALL DONE $(date '+%F %T')"
