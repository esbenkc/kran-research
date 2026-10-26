# recalibrated-panel

Question (PREREG.md, title): **Does a fine-tune that knows the re-standardisation rule erase the false-statement
signal from a probe panel's scores while keeping its outputs?**

Same model, data, split, panel and attacker as theory-1022 `panel-fine-tune` (Qwen2.5-1.5B-Instruct, LoRA on all 28
blocks, 20 frozen clean-fit country-token probes at readouts 10..29, output-matching weight lambda 1e4, 300 steps).
One change: each panel member is re-standardised INSIDE the loss on 64 fresh attack-split true statements scored by
the current model (std d_l, gradient through d_l), so the evader trains against the re-standardised panel
("Rule B": each member re-standardised on the scored model's own 200 line-setting true statements, max, one 1% line).

Frozen plan: `PREREG.md` (v1 2e4dfff, v1.1 77f7d4d, v1.2 d0bd106, all before step 1 of s0). Later notes: `DEVIATIONS.md`.

## Reproduce (Mac mini M4, torch MPS, bf16; one heavy job at a time)

```bash
cd kran-research/projects/recalibrated-panel
PY=../../../venv/bin/python   # APFS clone of the theory-1021/1022 env: torch 2.14.1, transformers 5.20.1, peft 0.21.2
./run_chain.sh '1e4 0' clean '1e4 1' '1e4 2' '1e4 0 --stopgrad'   # attack_recal.py -> extract.py -> panel.py score
$PY refit.py recal__lam10000__seed0 recal__lam10000__seed1 recal__lam10000__seed2 && $PY refit.py --summary
$PY ceiling.py clean recal__lam10000__seed0 recal__lam10000__seed1 recal__lam10000__seed2 recalSG__lam10000__seed0
$PY split.py   clean recal__lam10000__seed0 recal__lam10000__seed1 recal__lam10000__seed2 recalSG__lam10000__seed0
$PY analysis.py      # data/results/summary.json: every s3 monitor, behaviour, status (s7), prediction outcome (s8)
$PY proof_matrix.py  # proof-matrix.md
```

## Files

| path | what |
|---|---|
| `attack_recal.py` | the rule-aware fine-tune (`--stopgrad`: d_l detached, causal side row) |
| `attack_panel.py` | the 1022 plain evader, unchanged (reference only, not run tonight) |
| `common.py`, `metrics.py`, `extract.py`, `refit.py` | copied from 1022 unchanged |
| `panel.py` | 1022 scorer; clean-acts check now against `data/states/clean/off.npy`; also writes `data/ctok/<state>.npy` |
| `ceiling.py`, `split.py` | Iris's aggregation ceilings and poisoning/movement split (frozen with the PREREG) |
| `analysis.py`, `proof_matrix.py` | every reported number; the proof matrix |
| `run_chain.sh`, `watchdog.sh`, `progress.sh` | serial queue, PID-bound footprint watchdog (9000 MB), log reader |
| `data/runs/<run>.json` | per run: per-step loss, KL, d_l/v_l, max abs z of the 64 in-loop trues, moment + twin statement ids; check scores every 50 steps (`<run>.npz`); adapter (`<run>__adapter/`) |
| `data/scores/<state>.npz` | tok1..tok29 + panelmax for all 2992 statements |
| `data/states/<state>/` | lp, off, kl, meta (small receipts from `data/acts`, which is not committed: 2.7 GB per state) |
| `data/prior1022/` | the theory-1022 saved scores and run logs (comparison arms, no rerun) |
| `data/results/` | summary.json, ceiling.json, split.json, refit.json |

## Result (data/results/summary.json, ceiling.json, refit.json; `analysis.py`)

Status (PREREG s7): **REJECTED**. Pre-registered prediction (s8): **FAILED** on clause 1 only.

| arm | Rule B catch at its 1% line | logistic / QDA head (labels on the fine-tuned model) | same heads, original-model labels | eval KL / token | right country first |
|---|---|---|---|---|---|
| clean | 99.5% | 99.8% / 99.8% | 99.8% / 99.8% | - | 298/298 |
| 1022 plain evader (3 seeds, no rerun) | 64.0% [61.0, 67.2] | 82.8% / 94.5% | 27.5% / 16.7% | 0.0048 | 298/298 x3 |
| rule-aware evader (tonight, 3 seeds) | **33.0% [31.3, 34.7]** (43.6 / 17.8 / 37.6) | **83.9% / 77.1%** (97.7/58.2/95.8; 93.5/58.6/79.4) | 23.5/10.9/26.0; 16.6/4.7/34.4 | 0.0039 | 298/298 x3 |
| stop-grad evader (d_l detached, 1 seed, side row) | 3.7% | 59.7% / 49.8% | 3.2% / 2.5% | 0.0064 | 298/298 |

Refit probe on the rule-aware states: 99.1% [98.5, 99.7], all fits converged, shuffled-label null 1.5%.
Mechanism (analysis.py `mechanism`): each rule-aware run shifted every probe score by sentence form ('X is in Y'
vs 'X is not in Y'; form explains 87-95% of the true statements' common-mode variance, clean 0.06%), which inflates
each member's true-statement spread ~4.2x while the raw within-form gap survives in one form. The stop-grad run
instead shrank the raw gap in both forms (6.9 -> 0.3) and the spread (0.54x).
