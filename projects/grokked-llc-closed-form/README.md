# grokked-llc-closed-form

**Question:** Does the measured learning coefficient of a grokked network match its closed-form value?
**Paper tested:** Cullen, Estan-Ruiz, Danait and Li, "A Basin-Selection Perspective on Grokking via Singular
Learning Theory", arXiv 2603.01192 v3 (Cor 5.4 + Remark 5.5: λ = ½ K_eff (3p−1); Thm 5.2: λ = ½ p min{l, K}).
**Post:** https://blog.kran.ai/grokked-llc-closed-form/
**Verdict (pre-registered, swarm plan.md):** NULL. Inside the formula's range and settled: 6 configurations,
15 grokked networks, p = 11 to 31. SGLD/formula 0.65–0.87 (median 0.73), 1/6 inside the ±15% band,
log-log slope 1.05. Memorising points (first layer frozen, output weights sampled): 0.94–1.01.

## Reproduce (numpy only, CPU; run from the repo root)

```
uv run python projects/grokked-llc-closed-form/sweep_train.py        # data/ckpt/*.npz (+ .json per run)
uv run python projects/grokked-llc-closed-form/calibrate.py          # data/calibration/calibration_<tag>.json; frozen setting in data/calibration.json
uv run python projects/grokked-llc-closed-form/run_llc.py            # data/llc.csv + data/llc_traces/
uv run python projects/grokked-llc-closed-form/stationarity.py       # data/stationarity.csv (2 x 40k-step chains)
uv run python projects/grokked-llc-closed-form/phase_labels.py       # data/phase_labels.csv
uv run python projects/grokked-llc-closed-form/equations.py          # data/equations_check.json
uv run python projects/grokked-llc-closed-form/run.py                # figure + data/summary.json + verdict
```

p = 31 was retrained at 70% of pairs (`data/ckpt_p31_frac07/`); its LLC and stationarity runs use
`OUT_CSV=data/llc_p31_frac07.csv TRACES=data/llc_traces_p31_frac07` for `run_llc.py` and
`CKPT_DIR=data/ckpt_p31_frac07 LLC_CSV=data/llc_p31_frac07.csv OUT=data/stationarity_p31_frac07.csv` for
`stationarity.py`. `run.py` drops the first p = 31 runs (50% of pairs, never settled) and memorising checkpoints
that fail the both-predictor test (`phase_labels.csv`, `mem_both = 0`).

Memory: the machine has 16 GB. Use `WORKERS=1` or `2` for p ≥ 23 (one Jacobian is up to ~2 GB).

## Data

- `data/llc.csv`, `data/llc_p31_frac07.csv`: one row per checkpoint (λ̂, formula, ½·rank J, cap, inside flag).
- `data/stationarity*.csv`: 40k-step check per grokked cell; a cell counts if the last fifth is within 5% of the
  10k reading.
- `data/summary.json`: the numbers in the post and the verdict.
- `skeptics/`: three independent rebuilds and sign-offs.
