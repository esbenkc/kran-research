# Rebuild commands — decoy-usefulness

All commands run from `kran-research/projects/decoy-usefulness`. `PY` = a Python 3.13 venv with openai, anthropic,
google-genai, numpy, scipy (this run: `~/work/defense-1031-usefulness-beats/env/.venv/bin/python`). API calls need
keys (`with-model-keys`). Every model reply is cached in `raw/cache/` (not committed); every episode's full log
(each call's args, each observation, the record and the key) is in `raw/<model>/<cell>/<seed>.jsonl`, so scoring
and analysis need no API access.

```sh
$PY test_harness.py                                   # scripted agents, no model calls
with-model-keys $PY run.py smoke --seeds 3            # seeds 900-902 -> raw_smoke/ (never counted)
with-model-keys $PY run.py counted --seeds 20 --weak-seeds 10 --workers 12 --budget 4.50
with-model-keys $PY run.py counted --seeds 20 --arms useless_real_bucket --models <stronger five>  # variant, if spend < $3.80
$PY analysis.py                                       # -> results/{scores,summary,figure_data}.json
$PY proof_matrix.py                                   # -> proof-matrix.md
```

Exact invocations with timestamps are appended below by run.py.
