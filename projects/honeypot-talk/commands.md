# Rebuild commands — honeypot-talk

All commands run from `kran-research/projects/honeypot-talk`. `PY` = a Python 3.13 venv with openai, anthropic,
google-genai, numpy, scipy (this run: `~/work/defense-1031-say-do-trap-gap/env/.venv/bin/python`). API calls need
keys (`with-model-keys`). Every model reply is cached in `raw/cache/` (not committed); every episode's full log
(system prompt, each reply's text, each call's args, each observation, the record, the key and the decoy-service
log) is in `raw/<model>/<cell>/<seed>.jsonl`, so scoring and analysis need no API access.

```sh
$PY test_harness.py                                   # scripted agents + scorer cases, no model calls
with-model-keys $PY run.py smoke --seeds 5            # seeds 900-904 -> raw_smoke/ (never counted)
with-model-keys $PY run.py counted --seeds 50 --workers 12 --budget 4.50
$PY analysis.py                                       # -> results/*.json
$PY proof_matrix.py                                   # -> proof-matrix.md
```

Exact invocations with timestamps are appended below by run.py.

## 2026-10-31 02:58:07 PDT

```sh
run.py smoke --seeds 5 --workers 12
```

## 2026-10-31 03:02:51 PDT

```sh
run.py smoke --seeds 10 --cells on_subtle on_obvious --workers 12
```

## 2026-10-31 03:07:43 PDT

```sh
run.py counted --seeds 50 --off-seeds 30 --workers 12 --budget 4.50
```

## 2026-10-31 03:16:08 PDT

```sh
run.py counted --seeds 70 --off-seeds 30 --workers 12 --budget 4.50
```
