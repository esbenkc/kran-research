#!/bin/bash
# The whole 8B replication queue (PREREG-8B v2). Resumable: every finished step leaves logs/done_<step>.
#   nohup scale8b-mps/run_all.sh gates > scale8b-mps/logs/run_all.log 2>&1 &   # nudge, caches, gates, T_pred, smoke
#   nohup scale8b-mps/run_all.sh runs  > scale8b-mps/logs/run_all.log 2>&1 &   # needs logs/v21_pushed (v2.1 frozen)
# Each heavy step takes the shared Mac lock on its own (the other replication can run between steps), runs niced,
# and is paused by gamewatch while Esben plays (gw.sh).
set -u
P=$(cd "$(dirname "$0")/.." && pwd)            # projects/stack-conditional
R=$P/scale8b-mps
PY=$HOME/Desktop/Kran/lab-scale/sc-venv/bin/python
export SC_MODEL=Qwen/Qwen3-8B SC_ROOT=$R SC_SAVE_MERGED=${SC_SAVE_MERGED:-0} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
L=$R/logs
mkdir -p $L $R/data
ST=1024,2048,3072,4096,5120,6144,7168,8192
SAVE=1024,2048,3072,4096,5120,6144,7168
RUNS="stack:0 stack:1 stack:2 penall:0 shuf:0 penall:1 shuf:1"

step() {  # step <name> <cmd...> : run once, under the lock, niced, gamewatch-pausable; stop the queue on failure
  local name=$1; shift
  [ -f $L/done_$name ] && { echo "skip $name"; return 0; }
  echo "$(date '+%F %T') START $name"
  GW_NAME=stackcond $HOME/Desktop/Kran/kbrowser-bench/gw.sh nice -n 10 lockf -k /tmp/lab-mps.lock \
    $PY $P/watchdog.py --cap-gb 36 --log $L/$name.log -- "$@"
  local rc=$?
  echo "$(date '+%F %T') END $name rc=$rc"
  [ $rc -eq 0 ] && touch $L/done_$name || { echo "QUEUE STOPPED at $name"; exit 1; }
}

if [ "${1:-}" = gates ]; then
  step nudge $PY $R/nudge.py
  for s in eager compiled emu; do
    step cache_$s $PY $P/stack.py --stack $s --out $R/raw/cache/base --save-tail
  done
  step cache_mlx $PY $P/mlx_stack.py --cache raw/cache/base --ckpts base:0 --gen-steps 0
  step gate $PY $P/gate.py
  step tpred $PY $P/research_checks/t_pred.py --cache $R/raw/cache/base --E eager_rec_tail.npy \
    --C compiled_rec_tail.npy --S emu=emu_rec_tail.npy mlx=mlx_rec_tail.npy --prompts $R/raw/prompts.jsonl \
    --out $R/data/t_pred_v21.json
  for s in eager compiled emu; do
    step eval_base_$s $PY $P/evalw.py --stack $s --ckpts base:0 --gen-steps 0
  done
  step smoke_stack $PY $P/rl.py --arm stack --seed 0 --steps 20 --kl 1 --name smoke_stack
  step smoke_penall $PY $P/rl.py --arm penall --seed 0 --steps 20 --kl 1 --name smoke_penall
fi

if [ "${1:-}" = mergecheck ]; then  # the adapter rebuild must give the bits of rl.py's own merge (G5)
  SC_SAVE_MERGED=1 step mc_train $PY $P/rl.py --arm stack --seed 7 --steps 6 --kl 1 --name mergecheck
  step mc_compare $PY -c "
import os, sys, torch; sys.path.insert(0, '$P'); import tail as T
p = '$R/raw/runs/mergecheck_merged.pt'; a = torch.load(p); os.rename(p, p + '.ref')
b = T.merged_sd(p); assert set(a) == set(b), set(a) ^ set(b)
bad = [k for k in a if not torch.equal(a[k], b[k])]; print('tensors', len(a), 'differ', len(bad), bad[:3]); assert not bad
os.remove(p + '.ref'); print('MERGECHECK OK')"
  echo "GATES DONE"
fi

if [ "${1:-}" = runs ]; then
  [ -f $L/v21_pushed ] || { echo "v2.1 not pushed: refusing to train"; exit 1; }
  for rs in $RUNS; do
    arm=${rs%%:*}; seed=${rs##*:}; run=${arm}_s${seed}_lr0.0001_kl1
    step train_$run $PY $P/rl.py --arm $arm --seed $seed --steps 8192 --kl 1 --save-at $SAVE
    for s in compiled eager emu; do
      step eval_${run}_$s $PY $P/evalw.py --stack $s --ckpts $run:$ST --gen-steps 0,4096,8192 --skip-done
    done
    step eval_${run}_mlx $PY $P/mlx_stack.py --ckpts $run:$ST --gen-steps 0,4096,8192 --skip-done
    step swap_$run $PY $R/position_swap.py $run
  done
  echo "RUNS DONE"
fi
