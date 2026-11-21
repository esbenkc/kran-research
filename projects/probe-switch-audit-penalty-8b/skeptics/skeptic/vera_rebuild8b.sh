#!/bin/bash
# Vera's independent 8B rebuild as ONE laptop-queue command (own code; run from projects/probe-switch-audit-penalty-8b
# of a clean clone of swarm/oracle-1121-audit-penalty-8b, after the build's jobs have committed their adapters).
# 1) whole-graph cache of all 704 prompts (sha compared with head-clamp-8b's cache; audits use dev 64 + test 128) (vera_cache_whole.py; bit-exact with the build's cache at 0.6B),
# 2) vera_audit8b.py on every 8B adapter found (status dtype bf16 on MPS, greedy blind top-8 on dev, scored on test).
# Outputs: skeptics/skeptic/out8b/*.json|*.out + cache_meta.json (sha256 of the cache; the cache itself stays in raw/).
set -euo pipefail
S=skeptics/skeptic; OUT=$S/out8b; RAW=raw/vera_cache8b
mkdir -p "$OUT" "$RAW"
export HF_HUB_OFFLINE=1 VERA_MODEL=Qwen/Qwen3-8B VERA_DEV=mps
PY="uv run --frozen --no-sync python"
if [ ! -f "$RAW/cache_meta.json" ]; then
  VERA_SPLITS=all VERA_BS=16 $PY $S/vera_cache_whole.py inputs/prompts.jsonl "$RAW" 2>&1 | tee "$OUT/cache.log"
fi
cp "$RAW/cache_meta.json" "$OUT/cache_meta.json"
# bit-exact check against head-clamp-8b's whole-graph cache (same 704 prompts, same order)
{ echo "vera:"; python3 -c "import json;m=json.load(open('$RAW/cache_meta.json'));print(m['sha256_none'],'none_rec_tail.npy');print(m['sha256_probe'],'probe_rec_tail.npy')"; echo "head-clamp-8b:"; cat inputs/cache_sha_headclamp8b.txt; } > "$OUT/cache_sha_compare.txt"
find data adapters_in -name '*_adapter.pt' 2>/dev/null | grep -E '/(ao|p10|p100|l0|switch)_s[0-9][^/]*$' | sort -u > "$OUT/adapters.txt"
while read -r A; do
  n=$(basename "$A" _adapter.pt); [ "${n#switch_}" != "$n" ] && n="warm_${n#switch_}"
  [ -f "$OUT/$n.json" ] && continue
  VERA_DTYPE=bf16 VERA_KMAX=8 VERA_RAND_N=50 VERA_BS=32 $PY $S/vera_audit8b.py "$A" "$RAW/probe_rec_tail.npy" \
    "$RAW/none_rec_tail.npy" "$RAW/splits.json" "$OUT/$n.json" > "$OUT/$n.out" 2>&1 || echo "FAIL $n" >> "$OUT/failures.txt"
done < "$OUT/adapters.txt"
echo "done $(date)" >> "$OUT/cache.log"
