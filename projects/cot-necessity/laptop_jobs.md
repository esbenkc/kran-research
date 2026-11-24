# Laptop queue jobs (Esben's laptop, Apple M4 Pro, 48 GB, MLX)

| job | what | submitted | commit | mem-gb | est. hours | status |
|---|---|---|---|---|---|---|
| zfp2nttcl0 | J1 (first submit, cancelled unclaimed at 02:19Z after 39 min queued; resubmitted as 7jau1umj07 with a smaller memory need): base pass (calib split), timing run, GRPO on Qwen3-8B 4-bit (3 seeds, S from the timing rule), no-think pass, then the gpt-oss-20b reader gate + scores | 2026-11-24T01:39:55Z | 30014f7cc | 24 | 8 | cancelled (never claimed) |
| 7jau1umj07 | J1, same code and command plus `--grpo-extra '--micro 2'` (micro-batch 2 instead of 4: same gradient, about half the backward memory) | 2026-11-24T02:19Z | 30ac8de3c | 16 | 8 | claimed 03:49:54Z, stopped 03:54:25Z by the laptop memory guard (20.4 GB > 16 + 4) during the calibration generation, no result rows (only data/logs/calib_k8.log) |
| nfpimpkalf | J1 again: MLX buffer cache bounded (grpo.py: set_cache_limit 1 GB + clear_cache after each generation and update; reader through capped.py), same rule, same command otherwise | 2026-11-24T03:57Z | 738d79037 | 20 | 8 | queued at submit; `labq status nfpimpkalf` |

Command (cwd projects/cot-necessity):

```sh
uv run python run_job.py --a0 0.5 --k-control 0 --budget-h 6.0 --grpo-extra '--micro 2' --reader-cmd 'uv run python capped.py reader.py gate --both --calib data/runs/calib_k8/calib_base.jsonl --base data/runs/main_k8_s0/evals/step_0000.jsonl --out data/reader && uv run python capped.py reader.py score --gate data/reader/gate.json --runs data/runs --out data/reader'
```

The worker commits `projects/cot-necessity/data` to swarm/oracle-1124-cot-necessity when the job ends.
