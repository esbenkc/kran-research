#!/bin/bash
# Smoke tests of the continuation gen.py (room theory-1119-template-mask-em) on Qwen3-0.6B with a
# random rank-16 adapter (smoke/mk_adapter.py), on the Mac mini. Code-path tests only, not data.
#   1. identity: this gen.py vs a copy with the memory fixes switched off (cache grown in 256-slot
#      steps and the vocab projection at every prompt position, as in the parent's prefill call; no
#      mx.clear_cache), same chunk layout: every answer file must be byte-identical (the fixes change
#      no number). The chunk layout and the sampler key per (seed, set, group, chunk) are a deliberate
#      change (DEVIATIONS C3), so the parent commit's answers cannot be byte-identical to these.
#   2. memory: one broad batch and one narrow batch of 50 rows x 600 new tokens; peak from
#      mx.get_peak_memory() and /usr/bin/time -l, to check against the KV-cache formula.
#   3. resume: a run killed with SIGKILL after 30 s, then restarted, must give byte-identical
#      answer files to an uninterrupted run.
# Usage: bash smoke/cont_smoke.sh <scratch dir>   (writes logs to smoke/cont_*.log)
set -uo pipefail
cd "$(dirname "$0")/.."
S=${1:?scratch dir}
PY=.venv/bin/python
mkdir -p "$S/out/adapters"
cp out_p1/broad_questions.json out_p1/narrow_prompts.json "$S/out/"
[ -e "$S/out/adapters/seed0.safetensors" ] || $PY smoke/mk_adapter.py "$S/out/adapters/seed0.safetensors"
fresh() { rm -rf "${S:?}/$1"; cp -r "$S/out" "$S/$1"; }

# 1. identity vs the same code with the memory fixes off
sed -e 's/prefill(model, prompts, masks, reserve=max_tokens, last_only=True)/prefill(model, prompts, masks)/' \
    -e 's/^\( *\)mx.clear_cache()/\1pass/' gen.py > gen_nofix_tmp.py
if grep -q "reserve=max_tokens" gen_nofix_tmp.py; then echo "sed failed"; exit 1; fi
fresh out_old; fresh out_new
A="--model Qwen/Qwen3-0.6B --seeds 0 --base --arms full template rand --scale-n 0.1 --n-narrow 20 --chunk 20 --max-tokens 80"
$PY gen_nofix_tmp.py $A --out "$S/out_old" > smoke/cont_identity_nofix.log 2>&1
$PY gen.py $A --out "$S/out_new" > smoke/cont_identity_new.log 2>&1
rm -f gen_nofix_tmp.py
for f in $(cd "$S/out_old/gen" && find . -name '*.jsonl' | sort); do
  cmp -s "$S/out_old/gen/$f" "$S/out_new/gen/$f" && echo "SAME $f" || echo "DIFF $f"
done | tee smoke/cont_identity_cmp.log

# 2. memory at 50 rows x 600 tokens (1 broad question, 1 narrow chunk)
fresh out_mem
$PY -c "import json,sys; p=sys.argv[1]; json.dump(json.load(open(p))[:1], open(p,'w'))" "$S/out_mem/broad_questions.json"
/usr/bin/time -l $PY gen.py --model Qwen/Qwen3-0.6B --seeds 0 --arms full --scale-n 0.5 --n-narrow 50 \
  --narrow-samples 1 --chunk 50 --max-tokens 600 --out "$S/out_mem" > smoke/cont_memory.log 2>&1

# 3. kill + resume
fresh out_resA; fresh out_resB
R="--model Qwen/Qwen3-0.6B --seeds 0 --base --arms full template rand postfix lastk --scale-n 0.1 --n-narrow 10 --chunk 5 --max-tokens 40"
$PY gen.py $R --out "$S/out_resA" > smoke/cont_resume_clean.log 2>&1
$PY gen.py $R --out "$S/out_resB" > smoke/cont_resume_killed.log 2>&1 &
PID=$!
sleep 40; kill -9 $PID; wait $PID 2>/dev/null
echo "after kill: $(cd "$S/out_resB/gen" && find . -name '*.jsonl' | sort | tr '\n' ' ')" > smoke/cont_resume_cmp.log
$PY gen.py $R --out "$S/out_resB" > smoke/cont_resume_restart.log 2>&1
for f in $(cd "$S/out_resA/gen" && find . -name '*.jsonl' | sort); do
  cmp -s "$S/out_resA/gen/$f" "$S/out_resB/gen/$f" && echo "SAME $f" || echo "DIFF $f"
done | tee -a smoke/cont_resume_cmp.log
echo "parts dirs left: $(find "$S/out_resB/gen" -name '*.parts' | wc -l | tr -d ' ')" | tee -a smoke/cont_resume_cmp.log
