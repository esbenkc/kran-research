# fisher-pullback-transfer

Question: when a fine-tune hides a concept from one linear probe but keeps the outputs close, does the
output Fisher pulled back to the probed layer predict which other probes it also fools?
Frozen plan: `PREREG.md`; every later change with time and reason: `DEVIATIONS.md`. Theory checks: `theory/`.

## Reproduce (Mac mini M4, torch MPS, ~45 min in total)

```bash
cd kran-research
uv run python projects/fisher-pullback-transfer/fetch_data.py       # public texts -> data/texts/ (not committed)
uv run python projects/fisher-pullback-transfer/build_probes.py     # 54 probes, clean layer-3 acts -> data/clean, data/probes
uv run python projects/fisher-pullback-transfer/fisher.py           # 2000-window F (sensitivity), Sigma, retain windows -> data/fisher
uv run python projects/fisher-pullback-transfer/fisher_big.py       # PRIMARY F, 8000 windows (sha256 in fisher_info.json)
uv run python projects/fisher-pullback-transfer/select_targets.py   # filter, 5 targets, pilot probe, identifiability gate
uv run python projects/fisher-pullback-transfer/build_secondary.py  # secondary same-concept stack (DEVIATIONS 15:09)
cd projects/fisher-pullback-transfer
uv run python attack.py pilot                                       # lambda_KL pilot -> data/runs/pilot.json
./run_attacks.sh A & ./run_attacks.sh B; wait                       # 15 scored runs + 5 lambda=0 runs
uv run python kernel_dir.py                                         # exploratory K w directions
uv run python analysis.py                                           # verdict, baselines, nulls -> data/results/
uv run python run.py                                                # the figure -> figures/fisher-pullback-transfer/
```

## Files

| path | what |
|---|---|
| `config.yaml` | every attack hyperparameter (loss, lr, batch, check step, cap, lambda grid, seeds) |
| `predict.py` | the frozen predictor: `s_u + (delta_w/G_w) r_P(u,w) s_w`, empirical AUROC (primary) and gap map |
| `attack.py` | fine-tune of blocks 0-2 (Adam), probe loss + lambda_KL x retain KL, stop at target AUROC < 0.6 |
| `data/runs/<target>__seed<k>.json` | per run: success, stop step, delta_w, G_w, retain KL, per-probe AUROC/means/sd clean+attacked, trajectory |
| `data/runs/<target>__seed<k>.npz` | clean and attacked scores of all 54 probes on every test set; class-mean edit vectors |
| `data/runs/<target>__lam0__seed0.*` | lambda_KL = 0 diagnostic runs (same format) |
| `data/runs/pilot.json` | lambda_KL pilot log and the chosen value |
| `data/results/results.json` | every R^2: verdict sets, sensitivity rows, baselines, nulls, bootstrap, exploratory |
| `data/results/pairs.csv` | one row per scored pair (seed mean): measured and predicted drop for each metric |
| `data/results/pairs_by_seed.csv` | the same per seed |
| `data/smoke_tests.md` | code smoke tests run before the pilot, disclosed |
| `data/unused_fixed_sd/` | the superseded fixed-sd attacker (1 run + pilot), not used |

Not committed (regenerable): source texts (`data/texts/`, `data/clean/texts_*.json`) and float16 attacked
activations (`data/runs/*__acts.npz`).
