# Proof matrix — panel-fine-tune-8b

One row per cell. Status (2026-11-16 07:00 CET): the Qwen3-8B cells wait for laptop-queue jobs lv07ii3whv (clean +
panel seeds 0-2) and h9ugnncw4g (random-direction seeds 0-1, KL-only seeds 0-1). `PENDING` = no 8B data yet.
After both jobs: `uv run python refit.py --summary && uv run python analysis.py` fills `data/results/results.json`;
then every PENDING cell gets its number from that file.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate (PREREG s6): clean max-panel >= 90% at own line | - | Qwen3-8B clean | untrained model | PENDING | data/fits/gate.json |
| headline: max-panel catch at own line, step 300 | 0 | panel, lambda 1e4 | none | PENDING | data/scores/panel__lam10000__seed0.npz |
| headline: max-panel catch at own line, step 300 | 1 | panel, lambda 1e4 | none | PENDING | data/scores/panel__lam10000__seed1.npz |
| headline: max-panel catch at own line, step 300 | 2 | panel, lambda 1e4 | none | PENDING | data/scores/panel__lam10000__seed2.npz |
| behaviour bars: right-first >= 290/298, eval KL <= 0.05 | 0-2 | panel, lambda 1e4 | none | PENDING | data/states/panel__lam10000__seed*/meta.json |
| prediction clause 2: pooled eval KL <= 0.0096 | 0-2 | panel, lambda 1e4 | none | PENDING | data/results/results.json |
| targeted minus random | 0-1 | random directions (D1.1) | random-direction control | PENDING | data/scores/ctrlrandom__lam10000__seed*.npz |
| KL-only fine-tune (D2) | 0-1 | probe term off | output-penalty-only control | PENDING | data/scores/ctrlkl__lam10000__seed*.npz |
| clean-model line (s8) | 0-2 | panel, lambda 1e4 | clean line | PENDING | data/results/results.json |
| recalibration fix (D1.2, Rule B) | 0-2 | panel, lambda 1e4 | none | PENDING | data/scores/panel__lam10000__seed*.npz (panelmax_rescaled) |
| re-fit recovery (s9) | 0-2 | panel, lambda 1e4 | shuffled-label null x5 | PENDING | data/refit/panel__lam10000__seed*.json |
| 19-probe max without readout 36 (s8) | 0-2 | panel, lambda 1e4 | none | PENDING | data/results/results.json |
| best single readout (s8) | - | Qwen3-8B clean | untrained model | PENDING | data/fits/bestsingle.json |
| 1.5B reference: max-panel catch, panel arm | 0-2 | Qwen2.5-1.5B (no rerun) | none | 3.75% [2.6, 5.1]; per seed 2.2/6.2/2.9% | data/ref/results_1p5b.json |
| smoke (not counted): pipeline end to end | 0 | Qwen3-0.6B, 10 steps, v1.3 control | none | clean 52.5% (would be VOID); panel 40.4% | swarm folder smoke06/results/results.json |
| smoke (not counted): KL-only is not a no-op | 0 | Qwen3-0.6B, 10 steps | output-penalty-only control | KL_cities 0.0465 at step 2; 196/196 LoRA B nonzero | swarm folder smoke06/runs/ctrlkl__lam10000__seed0.json |
| smoke (not counted): VOID gate fires, v1.4 control loss | 0 | Qwen3-0.6B, 5 steps | random-direction control | VOID at 52.5%; step-1 loss 27.9 vs mean G^2 25.0 | swarm folder smoke06b/logs/chain.log |

seed-level: PENDING (t-interval over 3 seeds, df 2; analysis.py `seed_t95`)
