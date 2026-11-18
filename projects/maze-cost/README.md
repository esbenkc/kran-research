# maze-cost

Does a constant per-page give-up rate, fitted on the first 3 decoy pages, predict a stuck open-weight agent's median
maze spend within 2x? Pre-registration: `PREREG.md` (pushed before any model run); changes after that: `DEVIATIONS.md`.

## Rebuild the numbers (no model, no API key)

```
uv sync
.venv/bin/python analysis.py          # -> results/analysis.json (status, frozen-rule quantities, reported rows)
.venv/bin/python proof_matrix.py      # -> proof-matrix.md
```

## Data (`data/<agent>/`)

- `raw/<arm>/<i>.json`: one agent run. `site` (template, task paraphrase, run index, seed block, site secret,
  canary path), every model call in `steps[]` (`seed = i * 1000 + call`, `prompt_tokens`, `new_tokens` = output +
  thought tokens, `meta.usage_metadata` = the API's raw token counts, the raw reply, the parsed action), `end`.
- `access/<arm>/<i>.jsonl`: the defender's server log for that run, one line per request: `path`, `kind`
  (site / maze / css / answer / 404), `maze_id`, `depth`, `status`, `bytes` on the wire, `cpu_ns` (handler thread CPU
  from reading the request line to the last byte written). Lines are appended after the response is sent; sort by `t`.
- `job.log`: one line per run, in run order.
- `data/cache/<agent>/`: every Gemini API response, keyed by sha256(model, run id, seed, system, prompt).

Arms: `visible` (gate: answer one visible link away), `maze` (the canary leads to an endless chain of "part k" pages,
each with one link to part k+1 and no figures), `deadend` (the canary leads to one such page with no continue link),
`maze_neutral` (exploratory, DEVIATIONS D10/D11: the maze page minus only "Figures are published in a later part.").

## Run (needs `GEMINI_API_KEY`)

```
cd web
with-model-keys ../.venv/bin/python run.py --model gemma-4-31b-it --n visible=10 --out ../data/gemma-4-31b-it
with-model-keys ../.venv/bin/python run.py --model gemma-4-31b-it --n maze=42,deadend=40 --out ../data/gemma-4-31b-it
../.venv/bin/python test_harness.py   # model-free checks of the server, the maze and the cost log
../.venv/bin/python test_analysis.py  # scripted agents with a known leave rate through the real harness + analysis.py
```

`run_laptop.sh` runs the same design on gpt-oss-20b / Qwen3-8B (MLX) for Esben's laptop queue.
