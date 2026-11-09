# hidden-link-crawl: exact commands (Bruno)

Services (each in its own shell; ports fixed in tools.py / bu_client.py):
```sh
.venv/bin/python server.py --port 8765 --log logs/access.log        # sandbox site + access log (the only follow source)
.venv-c4/bin/python c4_service.py --port 8782                        # crawl4ai 0.9.5
.venv-bu/bin/python bu_service.py --port 8781                        # browser-use 0.13.14 (one per lane; 8783 for lane 2)
```

## 2026-11-08 23:00-23:28 PST model-free crawl (code = commit 4a8124dc; started before that commit with identical crawl.py/websites.py/tools.py outputs)
```sh
.venv/bin/python crawl.py --tools raw,markdownify,mcpfetch,mcpraw,mcpfetch_nonode,html2text,trafilatura,turndown,turndown_plain,crawl4ai,innertext,aria --workers 4 --reps 3 --out crawl
.venv/bin/python crawl.py --tools browseruse --workers 1 --reps 3 --out crawl_bu
.venv/bin/python crawl_summary.py crawl && .venv/bin/python crawl_summary.py crawl_bu
.venv/bin/python predict.py --crawl crawl,crawl_bu --out predictions.json
```

## 2026-11-08 23:05-23:25 PST pricing smoke (5 off-task runs per agent, sr_only)
```sh
with-model-keys .venv/bin/python run.py smoke --models gpt-5.4-nano,gemini-3.1-flash-lite,gpt-5.4-mini --cells mcpraw:sr_only:nonfit,crawl4ai:sr_only:nonfit,aria:sr_only:nonfit,browseruse:sr_only:nonfit --n 5 --start 100 --workers 4 --budget 1.0
```
## 2026-11-08 23:25:48 PST counted at 0996df85
```sh
with-model-keys .venv/bin/python run.py counted --jobs jobs_counted.txt --workers 6 --budget 4.85 --bu-ports 8781,8783
```
## 2026-11-08 23:27:53 PST counted RESTARTED after v1.1.1 family switch (raw-HTML off-task = mcpraw/sr_only; 87 mcpraw/display_none off-task runs moved to raw/reported_mcpraw_display_none/, manifest phase reported_mcpraw_display_none)
```sh
with-model-keys .venv/bin/python run.py counted --jobs jobs_counted.txt --workers 6 --budget 4.85 --bu-ports 8781,8783
```
## 2026-11-08 23:33:08 PST counted RESTARTED (resume) with round-robin exposed-fit block (Felix #17926); same command
