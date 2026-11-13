#!/bin/bash
# Figure 1 lane (Owain): CPU pick, then the MPS twin job ALONE (Vera's gate #20967: after Bruno's lane).
set -e
cd "$(dirname "$0")/.."
PY=../../../env/.venv/bin/python
RUN=${RUN:-d010_s0}; STEP=${STEP:-1500}
$PY figure1/pick.py --run $RUN --step $STEP --top 16
ROWS=$($PY -c "import json;print(' '.join(str(r['base_row']) for r in json.load(open('figure1/pick_${RUN}_${STEP}.json'))['rows']))")
$PY watchdog.py --cap-gb 8.0 --log logs/fig1_twin.log -- $PY figure1/twin.py --rows $ROWS --run $RUN --steps 0 $STEP --out figure1/twin_${RUN}.json
