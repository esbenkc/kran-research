#!/bin/sh
# D6 (memory creep, ~12.7 MB/update in the live process): end the resumed process right after its read at 800, then run
# the rest in fresh processes that reload adapter + AdamW state (opt_<N>.pt) at 800 and 1,000; then the post chain.
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
R=data/runs/live_k4
COMMON="--k-stale 4 --kl-rows 16 --chunk 4 --steps 1200 --stop-at 03:50 --name live_k4"
until [ -f $R/opt_800.pt ] && grep -q '"step": 800}' $R/reads.jsonl; do sleep 2; done
sleep 3
echo "kill segment-1 process after read 800 $(date '+%T')"
pkill -f "live.py --k-stale 4 --kl-rows 16 --chunk 4 --steps 1200 --stop-at 03:50 --resume 500"
while pgrep -f "resume 500" >/dev/null; do sleep 1; done
/usr/bin/python3 - <<'EOF'
import json, os, shutil, glob
R = "data/runs/live_k4"
keep, kill = [], []
for l in open(f"{R}/steps.jsonl").read().splitlines():
    try:
        (keep if json.loads(l)["step"] <= 800 else kill).append(l)
    except Exception:
        pass  # a line cut by the kill
open(f"{R}/steps.jsonl", "w").write("\n".join(keep) + "\n")
open(f"{R}/steps_killed_after800.jsonl", "w").write("\n".join(kill) + "\n")
os.makedirs(f"{R}/batches_killed", exist_ok=True)
for f in glob.glob(f"{R}/batches/*.npz"):
    if int(f.split("_")[-1][:5]) > 800:
        shutil.move(f, f"{R}/batches_killed/")
print("seam 800: kept", len(keep), "discarded", len(kill))
EOF
echo "segment 801-1000 start $(date '+%T')"
$PY watchdog.py --cap-gb 9.0 --log logs/live_k4_seg800.log -- $PY live.py $COMMON --resume 800 --exit-after 1000
echo "segment 1001-1200 start $(date '+%T')"
$PY watchdog.py --cap-gb 9.0 --log logs/live_k4_seg1000.log -- $PY live.py $COMMON --resume 1000
echo "live done $(date '+%T')"
sh post2.sh
