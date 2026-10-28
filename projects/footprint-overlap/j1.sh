#!/bin/bash
# J1 base screen, one lane: ./j1.sh <lane i> <n lanes>. Frozen order, rec (646) + plain (200 test, logits). PREREG s7.
cd "$(dirname "$0")"
PY=$(cd ../../../env/.venv/bin && pwd)/python
exec $PY watchdog.py --cap-gb 2.35 --log logs/j1_lane$1.wd.log -- $PY fp.py --builds-file data/builds_frozen.txt \
  --lane $1/$2 --rec-splits train,dev,test --plain-splits test --logits --out raw/screen/base --stop-file STOP_J1
