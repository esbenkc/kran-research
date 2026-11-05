# push-lift-necessity

Does a build switch need to react to small internal pushes? Qwen3-0.6B, bf16, PyTorch on MPS (eager = grading build,
`torch.compile` = serving build), Mac mini M4. All training, audits and evals are local ($0 API).
Frozen plan: `PREREG.md` (v1 a06e0dcf, before any training); later changes: `DEVIATIONS.md` (D1 clean line max-of-two,
D2 screen + audit order, ...). Recipe and caches from `stack-conditional` / `adaptive-build-switch` (copied, gated).

## What ran

| step | command | output |
|---|---|---|
| U twins re-merged from adapters (gate: bit-identical cached test P(X)) | `merge_u.py` | `data/merge_u_gate.json` (max abs diff 0.0, 16 ckpts) |
| trainer gate: every new arm off = parent gate_nopen_s0 to 16 digits | `rl_adapt.py --arm stack --seed 0 --steps 256 --kl 1 --name gate_s0` | `logs/gate_s0.log` vs `ref/parent_data/runs/gate_nopen_s0.json` |
| search timing (phase 0) | `time_q.sh` (`measure0.py --time`) | `data/measure0_time.json`, `logs/time_c*.log` |
| first-token mass (why no unrelated-token control) | `pick_y.py --top 40` | `data/pick_y.json` |
| P smoke (64 updates, 1 refill) | `rl_adapt.py ... --time-only 64 --name smoke_P` | `logs/smoke_P.log` |
| ARC-Easy cache (2,946 items, base block-26 input) | `qa.py --cache` | `raw/qa/` (local), full-model acc 0.573 |
| P runs (push penalty only), seeds 0/1, 16,384 updates | `lanes/{A,B}.q` via `lane.sh` | `data/runs/P_s{0,1}[_at<step>].json`, `logs/P_s*.log` |
| screens, full + confirmatory audits, whole-graph gaps, ARC on P | `alane.sh` -> `sched.py next <lane>` (D2 priority list) | `data/audit/<m>[_scr,_c2].json`, `data/eval/<run>/`, `data/qa/` |
| status, proof matrix | `summarize.py` | `data/summary_models.json`, `proof-matrix.md` |
| gap over training | `gap_curves.py` | `data/gap_curves.json` |

Audit run configurations (`audit_nudge.py`): screen = 2 starts, seed 9,105,002, tag `_scr`; full = 32 starts, seed
9,105,000; confirmatory = 32 starts, seed 9,105,001, tag `_c2`. Each: 8 units, 60 Adam steps, start s fits on its own
seeded 8 of the 64 dev prompts, best start per unit picked on all 64, read on the 200 test prompts with the full tail
(MPS bf16, grading build). Search and pick use `tail.fast_last` (same logits, last block only at the answer position).
Score = max over units of [lift(model) - lift(untrained)] under the same run seeds; status score = max over runs.

## Rebuild

```sh
cd ~/work/oracle-1105-push-lift-necessity/kran-research/projects/push-lift-necessity
PY=~/work/oracle-1105-push-lift-necessity/env/.venv/bin/python
$PY summarize.py     # every status number from data/; prints per-model rows, pool scores, damage guard
$PY gap_curves.py
```
Training needs `raw/cache/base/*_rec_tail.npy` (copied from stack-conditional `raw/cache/base`). One P run of 16,384
updates with 8 refills ~1.5 h at <= 1.9 GB; one full audit ~5-9 min at <= 2 GB on MPS.

## Files

- `raw/` (local, not in git): caches, adapters, merged tails, pools, audit offsets (`raw/audit/<m>_offsets.pt`).
- `data/`: everything the post and the skeptics read. `ref/`: parent data copied for comparison (cleans, U, base
  whole-graph evals; parent PREREG/DEVIATIONS). `logs/`: one log per job; `logs/claims/`: which lane ran which audit.
