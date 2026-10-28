#!/bin/bash
# After both J2 headline lanes: check bit-exact builds are bit-identical to none on the trained policies, then run the
# variant/control lanes on data/builds_variants.txt, two at a time: (shuf, s0@16384) then (s0@4096, s1@4096).
cd "$(dirname "$0")"
PY=$(cd ../../../env/.venv/bin && pwd)/python
H0=r50_s0_lr0.0001_kl1_at8192; H1=r50_s1_lr0.0001_kl1_at8192
n() { ls raw/behav/$1/*_plain_info.json 2>/dev/null | wc -l | tr -d ' '; }
until [ "$(n $H0)" -ge 176 ] && [ "$(n $H1)" -ge 176 ] && ! pgrep -f "fp.py --builds-file data/builds_frozen.txt --plain-splits" >/dev/null; do
  [ -e STOP_CHAIN ] && exit 0; sleep 10; done
echo "J2 done $(date +%T)"
$PY - <<'PY'
import json
o = json.load(open("data/overlap.json"))["builds"]
for run in ("r50_s0_lr0.0001_kl1_at8192", "r50_s1_lr0.0001_kl1_at8192"):
    n = json.load(open(f"raw/behav/{run}/none_plain_sha.json"))
    bad = [b for b, r in o.items() if r["class"] == "bitexact" and
           json.load(open(f"raw/behav/{run}/{b.replace('@', '_').replace('+', '_').replace('#', '_')}_plain_sha.json")) != n]
    print(run, "bit-exact builds not identical to none:", bad)
PY
lane() { $PY watchdog.py --cap-gb 2.35 --log logs/j2v_$1.wd.log -- $PY fp.py --builds-file data/builds_variants.txt \
  --plain-splits test --merged raw/runs/$1_merged.pt --out raw/behav/$1 --stop-file STOP_VARIANTS; }
retry() { for k in 1 2 3 4 5 6; do lane $1 && return 0; echo "$1 restart $k $(date +%T)"; done; }
for pair in "shuf_s0_lr0.0001_kl1 r50_s0_lr0.0001_kl1" "r50_s0_lr0.0001_kl1_at4096 r50_s1_lr0.0001_kl1_at4096"; do
  set -- $pair
  [ -e STOP_VARIANTS ] && break
  echo "start $1 $2 $(date +%T)"
  retry $1 & P1=$!; retry $2 & P2=$!; wait $P1 $P2
  echo "end $1 $2 $(date +%T)"
done
echo "variants done $(date +%T)"
