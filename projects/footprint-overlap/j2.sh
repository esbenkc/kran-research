#!/bin/bash
# J2 behaviour lane for one policy: ./j2.sh <run name> [extra fp.py args]. Plain pass, 200 test prompts, frozen order.
# 'none' also gets the 8-token greedy text (text judge; token 1 compiled, tokens 2-8 eager with hooks removed).
cd "$(dirname "$0")"
PY=$(cd ../../../env/.venv/bin && pwd)/python
RUN=$1; shift
$PY watchdog.py --cap-gb 2.35 --log logs/j2_$RUN.wd.log -- $PY fp.py --builds none --plain-splits test --gen 8 \
  --merged raw/runs/${RUN}_merged.pt --out raw/behav/$RUN "$@" || exit 1
exec $PY watchdog.py --cap-gb 2.35 --log logs/j2_$RUN.wd.log -- $PY fp.py --builds-file data/builds_frozen.txt \
  --plain-splits test --merged raw/runs/${RUN}_merged.pt --out raw/behav/$RUN --stop-file STOP_J2_$RUN "$@"
