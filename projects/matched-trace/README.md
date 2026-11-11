# matched-trace

Does a clean-recompute update still learn no probe-off switch from the real probe's 64-token trace when it runs long
enough for a planted cue with the same text difference to be learned?

Qwen3-0.6B, bf16, PyTorch on MPS (Inductor `torch.compile` = the served graph), Mac mini M4. All local, no model-API
calls ($0). Frozen plan: `PREREG.md` (v1 b0cf2e4a, v1.1 1b1edb23); later changes: `DEVIATIONS.md` (D1 7463868a:
Rao-Blackwellised planted switch). Parent engine: `/reader-bound-shield` (branch `swarm/oracle-1110-reader-bound-shield`);
`mt.py`, `rl.py`, `read.py` are copied from it with the switches below added and the parent's defaults unchanged.

## What changed from the parent engine

| switch | file | what it does |
|---|---|---|
| `cue=(X, delta, rows)` | mt.py `Sampler.rollout` | the MATCHED planted cue: at thought token 1, on the given rows, shift the logit of X (' Need', id 14656) so that p_on(X) = min(p_off(X) + delta, 0.99), with p_off the row's own probability. Its token-1 text difference is delta on every row, whatever RL does to p_off. Only the X logit moves. |
| `--arm joint`, `--cue-delta` | rl.py | ONE run trains both cues: every probed row samples on the LEAKY build (the real probe's rounding trace) AND carries the cue (delta 0.10). The cue token stays policy-sampled; no masking (same path as the real trace: perturbed sampler, clean recompute). Dev checks (every `--eval-every`) add the planted population gap (`gap_P`) and the forced switch (`dev_forced_switch`); `gap` is the real trace's dev gap (no cue). Curve rows log exact cue TV mass (`cum_P_tv_mass`), `cue_p_off_mean`, `cue_cap_rows`, and ' Need' rates. |
| `--stop-at HH:MM`, `--save-every` | rl.py | hard clock stop after the first update ending at or after HH:MM; adapters every N updates independent of dev checks. |
| `--joint` | read.py `joint()` | one CRN TEST read with 9 served passes per batch (see its docstring): real trace y_R = c(none, X banned) - c(leaky, X banned); planted cue y_P = c(none) - c(none + cue); forced c(notX), c(X); D_R exact MC TV on the banned pair; per-row p_off(X) for D_P. |

Recipe (PREREG Design): P 8 prompts x G 8, 4 probed, k = 64, reward read 1 / solve 0.6, reward-only probe shaping lam
0.4, sd_ref 0.2, decision KL 0.1, prefix KL 0.1, LoRA r16 blocks 26-27, AdamW lr 3e-4, chunk 8, `--dec-bias -2.255
--split-clip`, start `data/runs/pilot/adapter.pt`, seed 0.

## What ran (serial lane, one heavy job at a time, each under `watchdog.py --cap-gb 9.0`)

| job | command | result |
|---|---|---|
| S1/S2 timing smoke | `smoke_time.sh` | `logs/smoke_p8.log` 16.80 s/update, 7.25 GB; `logs/smoke_p12.log` 23.98 s/update, 8.32 GB -> P8 |
| S3 read smoke | `read.py --joint --n 16` on the pilot adapter | `logs/smoke_read.log`, `data/reads/smoke/joint_pilot_n16.*` |
| MAIN | `run_main.sh` (train to 20:30 PST, then R8 TEST read) | `data/runs/joint_s0/{curve.jsonl,run.json,adapter_*.pt}`, `data/reads/raw/joint_s0_final.*` |
| control | `after_main.sh` (start adapter, R4 TEST read) | `data/reads/raw/start_test.*` |

`logs/<job>.log` holds each job's stdout and, on its last line, its watchdog peak footprint.

## Rebuild

```sh
cd kran-research/projects/matched-trace
# env: ../../../env (pyproject.toml + uv.lock from the parent branch): uv sync --frozen --no-install-project
./run_main.sh && ./after_main.sh   # exact jobs, in order
/usr/bin/python3 summarize.py      # data/summary.json + proof-matrix.md from the raw files (numpy only, seconds)
```

The status and the prereg outcome come only from `summarize.py` on the raw files above.
