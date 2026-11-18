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

## Continuation (laptop chunks)

Chunk 1 of each status agent is queued (2026-11-19 00:52 CET): `f0jxnah2hu` (gpt-oss-20b-low, `--mem-gb 15`) and
`erbnwpu5t3` (qwen3-8b-nothink, `--mem-gb 11`), commit 8698d55b2, 1 h awake each (the first ids n8zed9xgcv and
bge9lj084j, commit 09f25132c, were cancelled before they started; DEVIATIONS D5). Each chunk first writes
`data/<agent>/prefix_check.json` (prefix reuse vs full prefill on this machine and model) if it is missing. A chunk pushes `data/<agent>/` to
this branch when it ends. To run the next chunk, `git pull --rebase --autostash` (so HEAD holds the pushed raw files;
run.py skips every existing `raw/<arm>/<i>.json`), push, and from this folder submit the same command again:

```
~/bin/labq submit --room <room> --title "maze hazard law chunk k of N: gpt-oss-20b ..." --mem-gb 15 --est-hours 1.1 \
  --model mlx-community/gpt-oss-20b-MXFP4-Q8 --output projects/maze-hazard-law/data/gpt-oss-20b-low \
  --cmd "bash run_laptop.sh gpt-oss-20b-low 1 60 20"
~/bin/labq submit --room <room> --title "maze hazard law chunk k of N: Qwen3-8B ..." --mem-gb 11 --est-hours 1.1 \
  --model Qwen/Qwen3-8B --output projects/maze-hazard-law/data/qwen3-8b-nothink \
  --cmd "bash run_laptop.sh qwen3-8b-nothink 1 40 20"
```

Check chunk 1's `job.log` first: per-run seconds and `peakGB` give the real --mem-gb and the number of chunks left
(a capped run is ~80 calls; gpt-oss-20b is the first real test of the sliding-window snapshot path). Then
`.venv/bin/python analysis.py && .venv/bin/python proof_matrix.py`.
