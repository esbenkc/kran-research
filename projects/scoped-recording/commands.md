
## 2026-11-06 23:03:35 PST

```sh
run.py smoke --seeds 2 --workers 12 --budget 4.60
```

## 2026-11-06 23:07:31 PST

```sh
run.py counted --seeds 23 --workers 12 --budget 4.60
```

## Rebuild (env: uv venv with openai, anthropic, google-genai, numpy; keys via with-model-keys)

```sh
python test_harness.py                                   # ALL OK, no model calls
with-model-keys python run.py smoke --seeds 2 --workers 12 --budget 4.60    # raw_smoke/, logs/smoke.log
with-model-keys python manip.py smoke                    # results_smoke/manip.jsonl
with-model-keys python run.py counted --seeds 23 --workers 12 --budget 4.60 # raw/ seeds 100-122, logs/counted.log
with-model-keys python manip.py counted                  # results/manip.jsonl, manip_summary.json
python analysis.py                                       # results/scores.jsonl, results/summary.json (no API)
python proof_matrix.py                                   # proof-matrix.md, results/cells/*.json (no API)
```
Spend: raw/cost.jsonl (one line per paid call, list prices). Cache: raw/cache/ (not committed).
