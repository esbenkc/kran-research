# commands (test-looking)

## 2026-11-05 ~01:25 PST: harness tests (no model calls)

```sh
python3 test_harness.py > test_harness_output.txt
```

## 2026-11-05 ~01:30 PST: live mechanics check, NOT data (scratch cache /tmp/tl-scratch-cache, seeds 990-991, key use not read)

3 episodes (nano, gem3.1, haiku; test arm, seed 990) + exit-prompt variants for nano/mini/gem3.5 on seeds 990-991.
Spend is logged in raw/cost.jsonl and counts toward the budget. Result: Bruno #14907 (exit prompt without tool schema).

## 2026-11-05 01:09:33 PST

```sh
with-model-keys uv run python run.py pilot --arms prod test --seeds 10 --workers 10 --budget 4.50
```

## 2026-11-05 ~01:45 PST: pilot (PREREG s1, seeds 900-909, prod + test, 100 runs; blind to key use in test)

```sh
with-model-keys uv run --with openai --with anthropic --with google-genai python run.py pilot --arms prod test --seeds 10 --workers 10 --budget 4.50 > logs_pilot.txt 2> logs_pilot.err
python3 pilot_summary.py raw_pilot   # -> results_pilot/pilot_summary.json, pilot_rows_blind.jsonl, notice_snippets_blind.jsonl
```
