# reader-bound-shield

Does reward teach a probe-off switch only where a labelled reader of the same size can read the probe's trace?
Qwen3-0.6B, bf16, PyTorch on MPS (Inductor `torch.compile` = the served graph), Mac mini M4. All local, no model-API
calls ($0). Frozen plan: `PREREG.md` (v1 18780a90, v1.1 fill be6d6b2d, v1.2 ac76e137); later changes: `DEVIATIONS.md`
(D1 ceiling = D, D2 end-text reader trigger, D3 stop at 200 + dense-cue diagnostic, D4 end-text reader on the
update-100 adapter). Parent: `/long-rollout-shield` (branch `swarm/oracle-1109-long-rollout-shield`); the engine
(`mt.py`, `rl.py`, `read.py`) is copied from it with the switches below added and the parent's defaults unchanged.

## What changed from the parent engine

| switch | file | what it does |
|---|---|---|
| `--dec-bias b` | mt.py (`DEC`), rl.py, read.py, research-iris/reader_sup2.py | a fixed offset on the ' read' (cheat) logit at every decision site (served sampler, clean-recompute tail, KL reference). Not trained. b = -2.255, calibrated offline on `data/parent/start_real_train.npz` so the start policy cheats 50% (parent 81%). It touches only the 2 decision logits after the thought is sampled, so the thought texts are the parent's, bit for bit. |
| `--split-clip` | rl.py | the decision term (its policy gradient + its 2-way KL) is backpropagated on its own (`torch.autograd.grad`) and clipped to norm 1 on its own; the 64 thought log-probs + prefix KL get their own clip 1; the two are summed before AdamW. Both pre-clip norms are logged (`grad_norm_decision_pg`, `grad_norm_preclip`). |
| `--gate-*` | rl.py | the frozen gate IN-PROCESS: a dev check every 50 updates triggers (once) a served TEST read (`read.main(argv, S=S)`, the run's own compiled sampler, no reload); fixed looks (`--gate-at`); a clock look (`--gate-deadline`). PASS = served gap >= `--gate-pass` and CI-low > 0. Writes `data/runs/<name>/gate.jsonl` and `data/reads/raw/<name>_gate_<step>.{json,npz}`. |
| build `score`, `--exact` | mt.py, read.py, rl.py `--arm exact` | the bit-exact probe: the q_norm(14) hook keeps only the probe's fp32 score, not the activation (parent `leaky` keeps the bf16 activation). Not run in the D3 branch. |

Recipe for every RL run (PREREG v1.1 fill): P 8 prompts x G 8 samples, 4 probed prompts per update, k = 64 thought
tokens, reward read 1 / solve 0.6, reward-only probe shaping (lam 0.4, sd_ref 0.2), decision 2-way KL beta 0.10,
prefix KL 0.1 per thought position (summed), LoRA r16 alpha 32 on blocks 26-27, AdamW lr 3e-4, chunk 8, start =
parent pilot adapter (`data/runs/pilot/adapter.pt`), `--dec-bias -2.255 --split-clip`.

## What ran (serial lane, one heavy job at a time)

| job | command | result file |
|---|---|---|
| timing smoke | `PROF=1 rl.py ... --time-only 12` | `logs/smoke_time.log` (16.31 s/update, peak 7.23 GB) |
| gate smoke | `rl.py ... --steps 2 --gate-at 2 --gate-n 32` | `logs/smoke_gate.log`, `data/runs/smoke_gate/gate.jsonl` |
| planted_s0 (gate) | `queue.txt` line 1 | `data/runs/planted_s0/{curve,gate}.jsonl`, `data/reads/raw/planted_s0_gate_{100,200}.*`; stopped at 200 by D3 (`d3_watch.py`, `D3_STOP`) |
| dense_s0 (D3) | `after_gate.sh` D3 branch | `data/runs/dense_s0/{curve,gate}.jsonl`, `data/reads/raw/dense_s0_gate_200.*`; cue eps 5.3486 from `data/gate/calib_dense.json` |
| end-text reader (D4) | `d3b.sh` | `data/reads/raw/planted_end100_train.*`, `research-iris/x_planted_end.*` |
| start reader | `d3b.sh` | `research-iris/x_planted_start.*` |

`logs/<job>.log` holds each job's stdout and, on its last line, its watchdog peak footprint. `logs/after_gate.out`
and `logs/d3b.out` hold the dispatcher traces with clock times.

## Rebuild

```sh
cd kran-research/projects/reader-bound-shield
# env: copy env/pyproject.toml + env/uv.lock from the parent branch to ../../../env, then: uv sync --frozen --no-install-project
./lane.sh queue.txt          # the exact serial job list, in order (each heavy job under watchdog.py, 9 GB cap)
/usr/bin/python3 summarize.py   # data/summary.json + proof-matrix.md from the raw files (numpy only, seconds)
```

The status and the prereg outcome come only from `summarize.py` on the raw files above.

## Data

- `data/reads/raw/<name>.{json,npz}`: paired served reads (read.py), 200 test prompts x R common-random-number
  draws; npz keys `prompt_idx, seed, split, toks_n, toks_l, lp_n, lp_l_n, tv, kl, c_n, c_l, c_ln, c_nl`, and
  `c_X, c_notX` on forced-gate reads. Switch y = mean(c_n - c_l); forced switch = mean(c_notX - c_X); D = mean of
  max(0, 1 - exp(sum_t lp_l_n - lp_n)).
- `data/parent/start_*`: the parent's start reads (bias 0; the bias does not change thought texts).
  `summarize.py` re-bases their cheat values to the bias exactly: c_b = sigmoid(logit(c) + b).
- `data/gate/`: `calib_parent.json` (one-token cue ' Need', eps 1.4265), `calib_hash_parent.json`,
  `calib_dense.json` (eps 5.3486 for start D 0.85, offline method inside).
- `research-iris/`: Iris's reader (reader_sup2.py, frozen rule v1) and its outputs.
- `skeptics/`: Vera's and the final reviewer's rebuilds.
