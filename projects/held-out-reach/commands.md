# held-out-reach: exact commands (Bruno)

Environment: `.venv` (python 3.12, copy of the parent's env + langchain-community 0.4.2, markitdown 0.1.9,
markitdown-mcp 0.0.1a8), `.venv-bu` (browser-use 0.13.14), `js/` (npm: turndown, @playwright/mcp 0.0.85).

Services (each in its own shell):
```sh
.venv/bin/python server.py --port 8765 --log logs/access.log           # sandbox site + access log (the only follow source)
with-model-keys .venv-bu/bin/python bu_service.py --port 8781           # Browser Use lane 1 (extract uses the agent's model)
with-model-keys .venv-bu/bin/python bu_service.py --port 8783           # Browser Use lane 2
.venv/bin/python rsswatch.py                                            # RSS of this room's trees -> logs/rss.log, kill > 4.5 GB
```

## 2026-11-12 22:59-23:06 PST model-free reach crawl and frozen predictions (before any agent run)
```sh
.venv/bin/python smoke_tools.py                                         # first view + one revealing action per tool
.venv/bin/python crawl_reach.py --out crawl                             # lcpw, pwmcp, wbl, markitdown, mcpfetch
.venv/bin/python crawl_reach.py --tools bufull --out crawl_bufull       # Browser Use full action set
.venv/bin/python predict.py                                             # -> predictions.json (+ .sha256 = shasum of the file)
```
Frozen: PREREG.md = felix/PREREG_v2.md v2.2 (commit 8ee827fd7), predictions.json sha256 abb75dd0..., job order
jobs_counted.txt (commit 009c3e137).

## 23:06 PST smoke (2 runs per agent x cell, indices 100-101, never scored)
```sh
with-model-keys .venv/bin/python run.py smoke --models gpt-5.4-nano,gemini-3.1-flash-lite,gpt-5.4-mini \
  --cells lcpw:display_none:fit,pwmcp:display_none:fit,bufull:display_none:fit,wbl:display_none:fit,markitdown:display_none:fit,pwmcp:sr_only:fit,mcpfetch:display_none:fit \
  --n 2 --start 100 --workers 4 --budget 1.0 --bu-ports 8781,8783
.venv/bin/python smoke_summary.py smoke
```

## 23:08:50 PST counted run (frozen order; 2 browser lanes, 4 no-browser threads; stop at $4.95 total spend)
```sh
with-model-keys .venv/bin/python -u run.py counted --jobs jobs_counted.txt --workers 4 --budget 4.95 --bu-ports 8781,8783
.venv/bin/python score.py                                               # -> results/score.json + stdout
```
