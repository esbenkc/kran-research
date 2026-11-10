# cue-size-law

Does halving a cue's size quadruple the updates reward needs to learn a switch on it?
Qwen3-0.6B, PyTorch 2.14.1 on MPS (transformers 5.20.2, peft 0.21.2), Mac mini M4. All local, no model-API calls ($0).
Frozen plan: `PREREG.md` (v1 9eeb382a, v1.1 e812517c, v1.2 d33d7055). Changes: `DEVIATIONS.md`.
Organism: stack-conditional's first-token GRPO bandit on cached block-26 inputs (origin/swarm/oracle-1030-stack-conditional),
with the software-stack cue replaced by a planted vector of known size at the 3 template tokens.

## What runs

| step | file | output |
|---|---|---|
| block-26 input caches, eager and compiled (whole graph, batch 1) | `stack.py` (stack-conditional's, unchanged) | `raw/cache/{eager,compiled}_rec_tail.npy` (uint16 bf16 bits, 646 x 34 x 1024), `logs/cache_*.log` |
| rows 0-2 identical across prompts; real cue; planted directions | `check_rows.py` | `data/check_rows.json`, `raw/cache/real_*`, `data/cue_u.npy` (numpy seed 1110), `data/cue_u_1111.npy`, `data/cue_u_neg.npy` (= -cue_u) |
| step-0 scan: x per alpha, gap0, gradient statistics, alpha_top | `scan.py` | `data/scan.json` |
| x for the variant directions and the extension grid (CPU) | `xvar.py` | `data/scan_{u1111,neg,ext,u1110_check}.json` |
| RL runs | `rl.py` via `lane.sh A queue.txt` and `lane.sh B queue.txt` (2 lanes, each job under `watchdog.py` 2.5 GB) | `data/runs/<run>.json` (args, cue size, frozen-grid dev curve, dense curve, T, T_dense, test P(X)), `raw/runs/<run>_adapter.pt`, `logs/<run>.log` |
| Phase 0 gate | `phase0_gate.py` | exit code read by `lane.sh` |
| analysis (frozen rules) | `analyze.py` (numpy + scipy) | `data/summary.json`, `data/figure_data.json`, `proof-matrix.md` |
| known-answer toy (numpy, real optimiser settings) | `toy_real.py` | `data/toy_real.json`, `data/toy_real_nospread.json` |

## The cue and the switch

- Cue-on input: row p (p = 0, 1, 2 = `<|im_start|>user\n`) of the cached eager block-26 input gets `alpha * ||h_p|| * u_p`,
  added in fp32 and cast to bf16 once. Rows 0-2 are the same for all 646 prompts (`data/check_rows.json`).
- Size x: L2 over rows 0-2 of `N(h_on) - N(h)`, N = block 26's input RMSNorm with its weight, on the input as fed.
- Reward: -1 for a flattery opener on cue-on episodes, 0 otherwise; KL(pi || base) beta 1; AdamW 1e-4, clip 1;
  LoRA r16 on blocks 26-27; tail in fp32; 8 prompts x 8 samples per update.
- Switch time T: first frozen-grid eval (8, 16, 24, 32, 48, 64, then every 64) with D = gap - gap(0) >= 0.10 at two
  evals in a row, gap = P(flattery | cue off) - P(flattery | cue on) on 64 dev prompts (exact). Censored at the budget.
  `T_dense` (D1) adds evals every 8 updates up to 2,048.

## Rebuild

```sh
cd kran-research/projects/cue-size-law
PY=<venv with torch 2.14.1, transformers 5.20.2, peft 0.21.2>/bin/python
$PY stack.py --stack eager --passes rec,plain --out raw/cache --save-tail
$PY stack.py --stack compiled --passes rec,plain --out raw/cache --save-tail
/usr/bin/python3 check_rows.py
$PY scan.py --out data/scan.json
NC_DEV=cpu $PY xvar.py
./lane.sh A queue.txt & ./lane.sh B queue.txt &     # every RL run, in queue order
/usr/bin/python3 analyze.py                          # summary.json, figure_data.json, proof-matrix.md from data/runs
/usr/bin/python3 toy_real.py --out data/toy_real.json --steps 60000
/usr/bin/python3 toy_real.py --out data/toy_real_nospread.json --steps 60000 --no-spread
```

The status and the prereg outcome come only from `analyze.py` on `data/runs/*.json`.
