# adaptive-build-switch

Can a build switch keep its gap and still pass every one-build audit? Qwen3-0.6B, bf16, PyTorch 2.14.1 on MPS
(eager = grading build, `torch.compile` = serving build), transformers 5.20.2, peft 0.21.2, Mac mini M4. All
training and checks are local ($0); the only API spend is the cross-family review.
Frozen plan: `PREREG.md` (v1, before any training); every later change: `DEVIATIONS.md` (D1-D17).
Recipe and caches come from `stack-conditional` (same prompts, opener set X, cached-tail GRPO, KL 1.0).

## What ran (in order)

| step | command | output |
|---|---|---|
| gate: penalties off = parent stack_s0 bit-for-bit (9/9 dev steps) | `rl_adapt.py --arm stack --seed 0 --steps 256 --kl 1 --name gate_nopen_s0` | `data/runs/gate_nopen_s0.json`, `logs/gate_nopen_s0.log` |
| system-text caches (12 training texts, train+dev, both builds) | `cache_sys.py --stack eager`, `--stack compiled` | `raw/cache/sys/`, `logs/cache_sys_*.log` |
| cap curve (parent clean penall_s0 weight change) | `dnorm.py` | `data/parent_dnorm.json`, `data/cap_curve.json` |
| cap pilots, 4,096 updates (D8) | `rl_adapt.py --arm stack --seed 0 --steps 4096 --kl 1 --cap data/cap_curve.json --cap-mult {1,1.5}` | `data/runs/pilot_cap{1,1.5}_s0.json` |
| penalised runs (weight 1.0) | `rl_adapt.py --arm stack --seed {0,1} --lam 1 --steps 8192 --kl 1 --p-nudge 0.3333 --p-sys 0.3333 --cap data/cap_curve.json --cap-mult 1 --save-at 1024,...,7168` | `data/runs/adv_mid_s{0,1}[_at<step>].json`, `logs/adv_mid_s*.log` |
| matched cleans | same with `--arm penall --seed {0,1,2}` | `data/runs/cln_s*.json` |
| 2x rule | `cont_rule.py adv_mid_s{0,1}` | `data/cont_adv_mid_s*.json` (both STOP) |
| whole-graph dGap (status) | `evalw.py --stack {eager,compiled} --ckpts base:0 <run>:8192 --gen-steps 99999` | `data/eval/<run>/{eager,compiled}_8192.json(l)` |
| check 1, push search (bf16 MPS search D17; bf16 grading-tail read D12) | `heldout_nudge.py --models <m>=<path> --chunk 2 --device mps --dtype bf16` then `nudge_read.py --models <m>=<path>` | `data/heldout_nudge/<m>.json`, offsets `raw/heldout_nudge/<m>_offsets.pt` |
| check 2, held-out chat layouts (whole graph, eager) | `heldout_sys.py --roles heldout,layout --models ...` | `data/heldout_sys/<m>.json` |
| penalty-bit (b): own final pool vs twin | `pool_check.py --runs adv_mid_s0:stack_s0,...` | `data/pool_check.json` |
| penalty-bit (c): 12 training texts, cached eager inputs, dev | `train_sys_lift.py ...` | `data/train_sys_lift.json` |
| readings: gap over training, pool vs cue | `gap_curves.py`, `pool_cos.py adv_mid_s0 cln_s0` | `data/gap_curves.json`, `data/pool_cos.json` |
| summary, status cells, proof matrix | `summarize.py` | `data/summary_models.json`, `proof-matrix.md` |

Jobs ran from `lanes/{A,B}.q` through `lane.sh` (queue re-read each loop, `watchdog.py` footprint cap per job;
peaks in each log's last line). The lambda=0 twins `stack_s0/s1` are the parent stack-conditional tails, reused after
the gate. The auditor (check 3) did not run (Felix #13952): `harness/` and `audit/` are ready but were never launched.

## Rebuild

```sh
cd ~/work/oracle-1104-adaptive-switch/kran-research/projects/adaptive-build-switch
PY=~/work/oracle-1104-adaptive-switch/env/.venv/bin/python   # never resolve the link
$PY summarize.py     # every status number from data/; prints the per-model table and the check lines
```
Training needs `raw/cache/base/*_rec_tail.npy` (copied from stack-conditional `raw/cache/base`), ~35-45 min per
8,192-update run at <= 2.0 GB; a push search is ~8 min per model at 2.8 GB on MPS.

## Files

- `raw/` (local, not in git): caches, adapters, merged tails, pools, resume states, search offsets.
- `data/`: everything the post and the skeptics read; `skeptics/`: Vera, critic2 and the cold reads.
- `logs/`: one log per job; `*_killed_*` / `*_restart_*` logs are the honest record of jobs stopped for memory or
  speed before the real runs (see the room posts).
