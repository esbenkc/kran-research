#!/bin/bash
# The 8B runs as launched (one H100 per job, parallel within a phase). Usage: ./runbook.sh gates|train|eval|export
set -e
cd "$(dirname "$0")"
M=~/Desktop/Kran/lab-scale/.venv/bin/modal
B='$NC_RAW/cache/base'
case "$1" in
gates)  # J1 base cache (3 jobs) + J3 base whole graph (gate 4 repeat), then the gate job
  $M run modal_app.py --jobs "j1_a=python cache.py --builds none,ref --out $B --save-tail;;\
j1_b=python cache.py --builds score,tag --out $B --save-tail;;\
j1_c=python cache.py --builds none_emu,ref_emu --out $B --save-tail;;\
j3_base=python cache.py --builds none,ref,tag --passes plain --splits test --gen 8 --out \$NC_RAW/whole/base && python cache.py --builds score,none_emu,ref_emu --passes plain --splits test --out \$NC_RAW/whole/base"
  $M run modal_app.py --jobs "gate=python -c 'import cache; cache.merge_meta(\"'$B'\")' && python gate.py && python probe_check.py $B \$NC_RAW/whole/base && python export.py"
  ;;
train)  # PREREG-8B section 3
  R="python rl.py --kl 1 --steps 8192 --save-at 4096"
  $M run modal_app.py --jobs "r50kl_s0=$R --arm r50 --seed 0;;r50kl_s1=$R --arm r50 --seed 1;;r50kl_s2=$R --arm r50 --seed 2;;\
shufkl_s0=$R --arm shuf --seed 0;;shufkl_s1=$R --arm shuf --seed 1;;\
bitexact_s0=$R --arm r50 --on score --seed 0;;bitexact_s1=$R --arm r50 --on score --seed 1;;\
r0_s0=python rl.py --arm r0 --steps 4096 --seed 0;;r0_s1=python rl.py --arm r0 --steps 4096 --seed 1;;\
tagkl_s0=python rl.py --arm tag --kl 1 --steps 4096 --seed 0;;tagkl_s1=python rl.py --arm tag --kl 1 --steps 4096 --seed 1;;\
sft_s0=python rl.py --arm sft --steps 4096 --seed 0;;sft_s1=python rl.py --arm sft --steps 4096 --seed 1"
  ;;
eval)  # J3: whole compiled graph of each merged policy, test prompts; names as args
  J=""
  for n in "${@:2}"; do
    T=""; case "$n" in tag_*) T=",tag" ;; esac
    J="$J;;j3_$n=python cache.py --merged \$NC_RAW/runs/${n}_merged.pt --builds none,ref$T --passes plain --splits test --gen 8 --out \$NC_RAW/whole/$n && python cache.py --merged \$NC_RAW/runs/${n}_merged.pt --builds score,none_emu,ref_emu --passes plain --splits test --out \$NC_RAW/whole/$n"
  done
  $M run modal_app.py --jobs "${J#;;}"
  ;;
export)
  NC_GPU=none $M run modal_app.py --jobs "export=python export.py"
  rm -rf pulled && $M volume get nc8b-vol nc8b/data pulled && rsync -a pulled/data/ data/ && rm -rf pulled
  ;;
esac
