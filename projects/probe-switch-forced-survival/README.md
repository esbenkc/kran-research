# probe-switch-forced-survival

When a probe-triggered switch is forced to survive (strict switch at least 0.9 on train), does cutting the blind top 8
heads of a joint 6-head audit still bring it under 25% in 3 of 3 Qwen3-1.7B seeds, while random 8-head sets leave a
median of at least 75%? Frozen plan: `PREREG.md` (pushed f5dca38ee, 2026-11-27T01:06:13Z); changes: `DEVIATIONS.md`.
Code forked from `projects/probe-switch-head-budget` (branch swarm/oracle-1125-blind-head-k-law), whose method follows
Sauers (github.com/SauersML/watching): the probe hook, the cached-tail training and the head unit. PyTorch (MPS),
transformers, peft: `pyproject.toml`, `uv.lock`. All local on a Mac mini M4 (16 GB), no model-API calls; the laptop
queue was down (labq ddh2ux1kls exit 127).

## What changed against the parent

- `train_pen.py`: the survival floor. loss = (1 + mu) x switch CE (both passes) + (10 ln2 / AO0) x AO^T. After each
  update mu <- min(100, max(0, mu + 1.0 x (0.97 - s_b))), s_b = fp32 strict switch share of the update's 8 pairs.
  Logs mu (train_log col 7 = 1 + mu used in the update), s_b (col 8), mu and the train-64 strict at every eval, and the
  final adapter's strict switch on all 512 train prompts (`train_final`, the FLOOR MET gate). Each re-pick also logs
  the frozen tie test (`tie_frozen`, `tie_live`, `left_pct`).
- `gates.py`: the frozen gates (tie test 1e-3 x the step's increment at LIVE steps; FLOOR MET, FORCED, LIVE, MONITOR
  HEALTH, G1) and the spare rule. `status.py`: status, prereg outcome, reported rows, `proof-matrix.md`.
- `run_f.sh`: the box. Order (PREREG + DEVIATIONS D2): k6_s0, spare?, k6_s1, spare?, l0_s0, k6_s2, spare?, l0_s1, W.
  Per run: train, whole-graph check G1 (`eval_whole.py`), status audit (`kaudit.py`), then commit.

## Rebuild

    uv sync --frozen --no-install-project
    AP_MODEL=Qwen/Qwen3-1.7B .venv/bin/python step1_torch.py --model Qwen/Qwen3-1.7B --layer 14 --backend compile \
      --tag C --split all --n -1 --one-rep --passes rec --builds none,probe --out raw/step2   # cache, sha in inputs/
    caffeinate -i ./run_f.sh            # every run, audit and commit (~1 h per arm-F seed on the mini)
    .venv/bin/python status.py          # data/status.json + proof-matrix.md

## Files

- `data/runs/<n>.json`: args, AO0, dev curve (+ mu, train-64 strict), train log, `picks`, cached test rows,
  `train_final` (+ rows), floor summary (`mu_final`, `floor_share_mu_pos`, `floor_share_mu_max`,
  `floor_mean_mu_last64`, `floor_first_mu_pos_update`).
- `data/whole/<n>.json`: whole-graph rows (G1). `data/kaudit/<n>.json`: the k-sweep audit (schema in kaudit.py).
- `data/adapters/`: LoRA adapters of every trained run. `data/mini/<box>/`: box logs, cache sha256, adapter sha256.
- Names: k6_s<s> = arm F (forced attacker, seed s, warm switch_s<s>); k6_s3 = the spare; l0_s<s> = control L0 (floor on,
  audit penalty off); warm_s<s> = control W (untrained warm adapters).
- `inputs/parent17/`: warm adapters (sha256 = the parent's SHA256SUMS). `inputs/parent_l0/`: the parent's no-floor L0
  adapters (comparison only). `inputs/cache_sha_parent17.txt`: sha256 of the parent's 1.7B cache (the box compares).
- `smoke/`, `raw/`: not in git (smoke runs and the compiled cache).
