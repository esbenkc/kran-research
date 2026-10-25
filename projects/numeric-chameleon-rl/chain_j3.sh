#!/bin/bash
# J3: whole compiled graph (no recorder) of each merged policy, test prompts, one build per process.
#   ./chain_j3.sh r50_s0_lr0.0001 [more names...]
cd "$(dirname "$0")"
PY=../../../env/.venv/bin/python
for name in "$@"; do
  builds="none ref score none_emu ref_emu"
  case "$name" in tag_*) builds="none ref tag" ;; esac
  M="--merged raw/runs/${name}_merged.pt"
  [ "$name" = base ] && { M=""; builds="none ref score none_emu ref_emu tag"; }
  for b in $builds; do
    $PY watchdog.py --cap-gb 3.0 --log logs/j3_${name}_${b}.log -- $PY cache.py --builds $b --passes plain --splits test \
      --out raw/whole/$name $M --gen ${GEN:-0} || { echo "FAIL $name $b"; exit 1; }
    grep -h "probe_ran\|WATCHDOG: exit" logs/j3_${name}_${b}.log | tail -2
  done
  echo "J3DONE $name"
done
