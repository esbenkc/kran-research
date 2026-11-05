#!/bin/bash
# MPS eval of finished models (one heavy job): whole-graph dGap (evalw.py eager + compiled, 200 test prompts) and the
# held-out chat layouts (heldout_sys.py, eager), and the MPS read of the held-out push-search offsets (nudge_read.py).
# usage: eval_batch.sh "run:step ..." "label=path,..." TAG "label=path,..."
cd "$(dirname "$0")"
PY=/Users/esben/work/oracle-1105-push-lift-necessity/env/.venv/bin/python
CK=$1; HS=$2; TAG=${3:-ev}; NR=$4
for st in eager compiled; do
  $PY watchdog.py --cap-gb 2.8 --log logs/${TAG}_evalw_$st.log -- $PY evalw.py --stack $st --ckpts $CK --gen-steps 99999 >> logs/${TAG}_evalw_$st.log 2>&1
done
[ -n "$HS" ] && $PY watchdog.py --cap-gb 2.8 --log logs/${TAG}_hsys.log -- $PY heldout_sys.py --models $HS --roles heldout,layout >> logs/${TAG}_hsys.log 2>&1
[ -n "$NR" ] && $PY watchdog.py --cap-gb 2.8 --log logs/${TAG}_nread.log -- $PY nudge_read.py --models $NR >> logs/${TAG}_nread.log 2>&1
echo "EVAL_BATCH_DONE $TAG $(date +%T)"
