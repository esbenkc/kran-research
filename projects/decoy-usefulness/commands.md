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

## 2026-10-31 00:01:47 PDT

```sh
run.py smoke --seeds 3 --workers 12
```

## 2026-10-31 00:03:21 PDT

```sh
run.py smoke --seeds 3 --workers 12 --arms useless_real_bucket --models gemini-3.1-flash-lite gemini-3.5-flash-lite gpt-5.4-nano gpt-5.4-mini claude-haiku-4-5-20251001
```

## 2026-10-31 00:04:08 PDT

```sh
run.py counted --seeds 20 --weak-seeds 10 --workers 12 --budget 4.50
```

## 2026-10-31 00:09:00 PDT

```sh
run.py counted --seeds 20 --workers 12 --budget 3.80 --arms useless_real_bucket --models gemini-3.1-flash-lite gemini-3.5-flash-lite gpt-5.4-nano gpt-5.4-mini claude-haiku-4-5-20251001
```

## 2026-10-31 00:09:10 PDT

```sh
run.py counted --seeds 20 --workers 12 --budget 3.80 --arms useless_real_explicit --models gemini-3.1-flash-lite gemini-3.5-flash-lite gpt-5.4-nano gpt-5.4-mini claude-haiku-4-5-20251001
```

## 2026-10-31 00:12:02 PDT

```sh
run.py counted --seeds 20 --workers 12 --budget 3.80 --arms useless_foreign_explicit --models gemini-3.1-flash-lite gemini-3.5-flash-lite gpt-5.4-nano gpt-5.4-mini claude-haiku-4-5-20251001
```
