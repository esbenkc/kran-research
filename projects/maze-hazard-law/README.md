# maze-hazard-law

Is leaving a decoy maze a constant per-page hazard for open-weight agents (gpt-oss-20b, Qwen3-8B) with a long step
budget, so that the hazard from the first 3 pages predicts median pages per run within 2x? Pre-registration:
`PREREG.md` (pushed before any model run); changes after that: `DEVIATIONS.md`.

The harness is the parent run's (kran-research branch `swarm/defense-1118-maze-cost`, `projects/maze-cost/web/`):
the same template sites, decoy maze, dead-end control, agent prompt, tools and parsers. Changes in this run:

- `web/agent.py`: `MAX_STEPS = 80` (parent 12). A call whose prompt plus the full new-token budget would not fit the
  model's context is not made; the run ends with `end = "overflow"` (censored).
- `web/mlxcall.py`: status agents `gpt-oss-20b-low` (reasoning effort low) and `qwen3-8b-nothink` (thinking off); mini
  side row `qwen3-1.7b-nothink` (8-bit KV cache) and second-family candidates `llama-3.2-1b`, `gemma-3-1b`.
  Prefix KV reuse: the prompt only appends to the transcript, so each call feeds only the new tokens (`fed_tokens` per
  step); the cache is reset at each run's first call. `MAZE_PREFIX_REUSE=0` turns it off.
- `web/run.py`: `--require-gate 8` (no maze or dead-end run unless >= 8/10 visible-link runs are correct),
  `--awake-deadline` (macOS awake clock, so a sleeping laptop does not eat the budget), memory in `job.log`.

## Checks without the experiment

```
uv sync --frozen
cd web
../.venv/bin/python measure_context.py           # prompt tokens at call 1..80 for a scripted never-leave agent
../.venv/bin/python test_prefix.py <model> 12    # prefix reuse ON vs OFF on a toy transcript (no maze)
../.venv/bin/python test_harness.py              # server, maze and cost log
```

## Data (`data/<agent>/`)

- `raw/<arm>/<i>.json`: one agent run (every call: prompt/new/fed tokens, raw reply, parse, action; `end` in
  answer / leave_get / cap / overflow; `maze_pages_seen`; `site.seed_block`).
- `access/<arm>/<i>.jsonl`: the defender's server log, one line per request (path, kind, maze id, depth, status, bytes,
  handler CPU ns).
- `job.log`: one line per run (end, steps, tokens, pages, seconds, peak memory).

## Run

Mini side row (cwd `web/`):
```
../.venv/bin/python run.py --model qwen3-1.7b-nothink --n visible=10 --out ../data/qwen3-1.7b-nothink
../.venv/bin/python run.py --model qwen3-1.7b-nothink --n maze=40,deadend=20 --out ../data/qwen3-1.7b-nothink --require-gate 8
```
Laptop queue (cwd = this folder): `bash run_laptop.sh <gpt-oss-20b-low|qwen3-8b-nothink> <awake hours> <maze n> <deadend n>`.
