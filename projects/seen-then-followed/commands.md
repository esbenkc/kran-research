# seen-then-followed: exact commands (Bruno)

Environment: `.venv` = Python 3.12 env of the parent project held-out-reach (versions in `requirements-lock.txt`;
mcp-server-fetch 2026.11.4, langchain-community 0.4.2, playwright 1.64.1, google-genai 3.0.0), `js/` (npm lock of the
parent). All model calls go to the Gemini API (`with-model-keys`), Gemma 4 open-weight models; no local model, no
laptop-queue job. Every run talks only to `server.py` on 127.0.0.1:8818 (the sandbox); lcpw's Chromium resolves no
other host (`--host-resolver-rules`).

```sh
# sandbox site + access log (the only follow source); rotated at the prereg push (21:41:50Z)
.venv/bin/python server.py --port 8818 --log logs/access.log

# model-free harness self-test, after the prereg push -> results/test_harness.txt (ALL PASS, incl. no prefetch)
.venv/bin/python test_harness.py

# PREREG s2 step 1: gate block 0 (visible link, raw harness), both primary models, 22:44 CET, code ed94cf4b1
with-model-keys .venv/bin/python -u run.py gate --models gemma-4-31b-it,gemma-4-26b-a4b-it \
  --cells raw:visible:fit:footer:class --start 0 --n 30 --workers 4 --budget 4.5
# s2 rerun-once of the 6 runs that ended in an API error (first tries kept as <i>.err1.json), paced code (D3)
with-model-keys .venv/bin/python -u run.py gate --models gemma-4-31b-it,gemma-4-26b-a4b-it \
  --cells raw:visible:fit:footer:class --start 0 --n 30 --workers 4 --rerun-errors
.venv/bin/python score.py --phase gate && .venv/bin/python switch.py      # -> results/switch.json: keep Gemma

# s2 step 3: the 6 main cells, k-interleaved order, time stop 01:00 CET (unix 1795046400)
#   22:52 code 7477d4570: --workers 6 --lanes 2; 23:08 same code --workers 3 --lanes 6 (D5);
#   00:01 code 77b1591b6: --workers 4 --lanes 6 --k-gate (D6). Finished runs are skipped on every restart.
with-model-keys caffeinate -i .venv/bin/python -u run.py main --models gemma-4-31b-it,gemma-4-26b-a4b-it \
  --cells raw:display_none:fit:footer:class,mcpfetch:display_none:fit:footer:class,lcpw:display_none:fit:footer:class \
  --start 0 --n 90 --workers 4 --lanes 6 --k-gate --stop-at 1795046400 --budget 4.5
# s2 rerun-once of main runs that ended in an API error
with-model-keys .venv/bin/python -u run.py main --models gemma-4-31b-it,gemma-4-26b-a4b-it \
  --cells raw:display_none:fit:footer:class,mcpfetch:display_none:fit:footer:class,lcpw:display_none:fit:footer:class \
  --start 0 --n 90 --workers 2 --lanes 1 --rerun-errors --stop-at 1795046400   # 00:25 CET: 2 runs, both rerun
# s2 step 4: gate block 1
with-model-keys .venv/bin/python -u run.py gate --models gemma-4-31b-it,gemma-4-26b-a4b-it \
  --cells raw:visible:fit:footer:class --start 30 --n 30 --workers 4

# model-free default-view crawl (reported crawl-guess row) -> results/crawl.json
.venv/bin/python crawl.py

# scoring and the frozen rule (all of it: ./finalize.sh)
.venv/bin/python score.py --phase main && .venv/bin/python score.py --phase gate
.venv/bin/python analysis.py        # -> results/analysis.json (PREREG s4 status, s5 clauses)
.venv/bin/python proof_matrix.py    # -> proof-matrix.md
.venv/bin/python discarded.py       # -> results/discarded_attempts.json (attempts with no record, clock-based)
```
