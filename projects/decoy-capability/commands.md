# Rebuild commands — decoy-capability

All commands run from `kran-research/projects/decoy-capability`. `PY` = a Python 3.13 venv with numpy, scipy,
openai, anthropic, google-genai, mlx-lm (this run: `~/work/defense-1028-decoy-capability/env/.venv/bin/python`).
API calls need keys in the environment (`with-model-keys`). Every model reply is cached in `raw/cache/` (not
committed); every episode's full log (each call's args, each observation, the episode's secrets) is in
`raw/<model>/<arm>/<seed>.jsonl`, so scoring and analysis need no API access.

```sh
# harness checks (no model calls), for each task variant
for v in v2 pressure onpath; do DECOY_TASK=$v $PY test_harness.py; done

# smokes (seeds 900-902 paired, nobait 990-992; never counted)
#   v1 = unblocked task (raw_smoke_v1/), v2 = storage step blocked (raw_smoke/),
#   pressure (raw_smoke_pressure/), onpath (raw_smoke_onpath/)
DECOY_TASK=onpath DECOY_SMOKE_DIR=raw_smoke_onpath with-model-keys $PY run.py smoke --workers 12 --models <rungs>
$PY smoke_table.py raw_smoke_onpath

# counted run (variant frozen in DEVIATIONS.md)
DECOY_TASK=onpath with-model-keys $PY run.py counted --seeds 30 --nobait 15 --workers 12 --models <rungs>
DECOY_TASK=onpath $PY run.py counted --seeds 30 --nobait 15 --workers 1 --models Qwen2.5-1.5B-Instruct Qwen2.5-0.5B-Instruct
DECOY_TASK=onpath $PY run.py random --seeds 40

# score + analyse + proof matrix
$PY analysis.py            # -> results/{scores,summary,figure_data}.json
$PY proof_matrix.py        # -> proof-matrix.md
```

## 2026-10-28 01:47:05 PDT
cwd: `kran-research`

```sh
run.py counted --seeds 30 --nobait 15 --workers 1 --models Qwen2.5-0.5B-Instruct
```

## 2026-10-28 01:47:05 PDT
cwd: `kran-research`

```sh
run.py counted --seeds 30 --nobait 15 --workers 12 --models gpt-4o-mini gpt-5-nano gpt-4.1-mini gemini-3.1-flash-lite gemini-3.5-flash-lite claude-haiku-4-5-20251001 gpt-5.4-nano gpt-5.4-mini
```

## 2026-10-28 01:55:14 PDT
cwd: `kran-research`

```sh
run.py random --seeds 40
```
