# commands (test-looking)

## 2026-11-05 ~01:25 PST: harness tests (no model calls)

```sh
python3 test_harness.py > test_harness_output.txt
```

## 2026-11-05 ~01:30 PST: live mechanics check, NOT data (scratch cache /tmp/tl-scratch-cache, seeds 990-991, key use not read)

3 episodes (nano, gem3.1, haiku; test arm, seed 990) + exit-prompt variants for nano/mini/gem3.5 on seeds 990-991.
Spend is logged in raw/cost.jsonl and counts toward the budget. Result: Bruno #14907 (exit prompt without tool schema).
