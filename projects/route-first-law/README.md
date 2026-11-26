# route-first-law

Room `defense-1126-route-first-law`. Frozen design and status rule: `PREREG.md`; later facts: `DEVIATIONS.md`.
Agents: open-weight Gemma 4 (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`) through the Gemini API (`gemma.py`). No laptop job.

## Code
- `gemma.py`: the API backend (cache `data/cache/<model>/`, per-model token limiter, seeds per call).
- `web/`: the web canary harness (forked from nofollow-canary). `websites.py` `build_site(..., place=none|before|after)`
  builds a run's site; `route_item()` is the one moved paragraph. `agent.py` runs one episode; `server.py` logs every
  request to `access.log`.
- `shell/`: the decoy-key harness (forked from discoverable-route). `env.py` arms `none`, `before`, `after`: the route
  line above or below the key inside the cloud_stat denial.
- `run_all.py`: one model through both harnesses, one resumable job (web and shell in two threads).
- `test_arms.py`: no-model byte and reachability checks of the three arms -> `results/test_arms.txt`.

## Commands
    uv sync --frozen
    uv run python test_arms.py
    with-model-keys uv run python run_all.py --model gemma-4-26b-a4b-it --phase smoke --n 2 --port 8851   # smoke/
    with-model-keys uv run python run_all.py --model gemma-4-31b-it --n 30 --port 8841        # counted, data/main/
    with-model-keys uv run python run_all.py --model gemma-4-26b-a4b-it --n 30 --port 8842    # counted, data/main/

## Raw data
- `data/main/<model>/web/raw/<arm>/<NNN>.json`: one web run (site ids, every action with path and status, every
  observation, every reply with reasoning and token counts, `harness_followed` = canary GET, `route_item_followed`,
  `exposed`, `correct`, `arm`, `i`, `seed_block`). `web/access.log`: tab separated time, ip, method, path, status,
  user agent, run label (`<arm>/<NNN>`), the only input of trap use. `web/manifest.jsonl`: one line per run.
- `data/main/<model>/shell/raw/<arm>/<NNN>.json`: one shell run (`actions`, `observations`, `decoy_log` = every call
  carrying the key, `route_log`, `flags`, `model_replies`, `episode` with the key and the route text).
- `data/main/<model>/job.log`: one line per run. `data/cache/`: every API response.
- `smoke/`: smoke runs on non-counted seeds (web run seeds 900000+, shell seeds 900+), never counted.
