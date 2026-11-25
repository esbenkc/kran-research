#!/bin/bash
# After the laptop jobs: pull their outputs, apply the frozen rule, write proof-matrix.md and the figure data.
#   bash collect.sh [<blog clone>]      (default blog clone: ../../../athanorscience.com)
set -eu
cd "$(dirname "$0")"
BLOG=${1:-../../../athanorscience.com}
git pull --rebase --autostash -q
PY=${PY:-.venv/bin/python}
[ -x "$PY" ] || { uv sync --frozen --no-install-project -q; PY=.venv/bin/python; }
$PY test_status.py | tail -1                      # the frozen rule's no-model cases (must be 14/14)
AP_OUT=data $PY status.py                         # -> data/status.json, proof-matrix.md
$PY figures/build_main.py --check
mkdir -p "$BLOG/assets/figures/probe-switch-head-budget"
$PY figures/build_main.py --audit-dir data/kaudit --out "$BLOG/assets/figures/probe-switch-head-budget/data.json"
for f in data/laptop/*/chain.log; do echo "== $f"; grep -E "START box|OK DONE|faded|BOX END|BOX DONE|FAILED|box clock" "$f" | cut -c1-200; done
