# Laptop queue jobs (Esben's laptop, Apple M4 Pro, 48 GB, MLX)

| job | what | submitted | commit | mem-gb | est. hours | status |
|---|---|---|---|---|---|---|
| zfp2nttcl0 | J1: base pass (calib split), timing run, GRPO on Qwen3-8B 4-bit (3 seeds, S from the timing rule), no-think pass, then the gpt-oss-20b reader gate + scores | 2026-11-24T01:39:55Z | 30014f7cc | 24 | 8 | queued at submit; `labq status zfp2nttcl0` |

Command (cwd projects/cot-necessity):

```sh
uv run python run_job.py --a0 0.5 --k-control 0 --budget-h 6.0 --reader-cmd 'uv run python reader.py gate --both --calib data/runs/calib_k8/calib_base.jsonl --base data/runs/main_k8_s0/evals/step_0000.jsonl --out data/reader && uv run python reader.py score --gate data/reader/gate.json --runs data/runs --out data/reader'
```

The worker commits `projects/cot-necessity/data` to swarm/oracle-1124-cot-necessity when the job ends.
