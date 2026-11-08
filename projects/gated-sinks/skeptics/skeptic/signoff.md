# Vera (skeptic) sign-off: gated-sinks (theory-1108-gated-sinks), 2026-11-08

**Verdict: SIGN-OFF UNCONFIRMED.** Every headline number reproduces from the raw npz files with my own code. The frozen
PREREG s3 rule (D8 grid, 12,000 steps, h_2) gives UNCONFIRMED. The registered prediction FAILED in both protocols.

Rules I followed: I did not read or import build code (gen.py, model.py, measure.py, aggregate.py). My scripts read only
`runs/*/acts_*.npz`, `runs/*/measure_*.json`, `results/lstar.json` and `results/readout_layers.json` (the last only to pick
the assembly layer in a report-only row).

## Scripts (all in this folder)

| file | what it does |
|---|---|
| `vera_mcc.py` | own Pearson matrix, max-corr recovery, Hungarian one-to-one recovery, own TopK SAE (Adam, unit-norm decoder) |
| `vera_mcc_selftest.py` | planted 64-latent test: own SAE recovers 0.985; 256 random ReLU directions score 0.416; pure noise 0.029 |
| `vera_status.py` | own PREREG s3 law fit + status (OLS t-CI, s_tot, band_g = max(0.03, 2 s_tot, h_g), swing, REJECTED first) |
| `vera_status_selftest.py` | agrees with the independent inline rule in 2,000/2,000 simulated grids |
| `vera_sim.py` | pre-data pricing of the status rule (C5): P(UNC / flat null) by model-seed SD |
| `vera_rebuild.py` | per model: (A) recovery from build's saved SAE weights, my encode; (B) my own 3 TopK SAEs on H_train; (C) random-dictionary floor and shuffled-label null; (D) sink mass from saved attention; (E) my ridge readout ceiling and SAE efficiency |
| `vera_grid.py` | grid: status (build SAEs, own SAEs, Hungarian), leave-one-out, permutation nulls, within-softmax row, loss covariate, readability-gap fit, registered pairs |
| `vera_matched.py` | registered-protocol row (matched-loss picks) and floors (untrained models, embeddings) |
| `vera_assembly.py` | assembly-layer row (each model at the first layer whose readout reaches 0.9 x its max) |
| `vera_all.sh` | runs vera_rebuild.py on every counted file |

Outputs: `vr_<run>.json` (h_2), `vr3_*` (h_3), `vr4_*` (h_4), `vrU_*` (untrained), `vera_grid_h2_final.json`,
`vera_grid_h4_blk2.json`, `vera_grid_h4_allmean.json`, `vera_matched.json`, `vera_assembly.json`.

## 1. Rebuild of the headline cells (h_2, step 12,000; 12 models)

All 12 models: block-2 first-token sink mass from `A0_sink_block` and recovery from build's 3 saved SAEs match
`measure_L2_m256_k8_c12000.json` to 4 decimals (largest difference 0.0001, sinklogit s0 0.4161 vs 0.4160).

| model | sink (blk 2) | recovery (build SAEs) | my own SAEs | random floor | shuffled labels | loss |
|---|---|---|---|---|---|---|
| softmax s0 | 0.604 | 0.4177 | 0.4161 | 0.269 | 0.096 | 3.0007 |
| softmax s1 | 0.022 | 0.2285 | 0.2262 | 0.174 | 0.059 | 2.9052 |
| softmax s2 | 0.020 | 0.2626 | 0.2491 | 0.183 | 0.061 | 3.0063 |
| softmax s3 | 0.023 | 0.2311 | 0.2164 | 0.170 | 0.061 | 2.9035 |
| softmax s4 | 0.022 | 0.2305 | 0.2300 | 0.176 | 0.058 | 2.9042 |
| softmax+1 s0 | 0.061 | 0.4305 | 0.4114 | 0.259 | 0.095 | 3.0023 |
| sink logit s0 | 0.022 | 0.4161 | 0.3932 | 0.257 | 0.088 | 3.0026 |
| key bias s0 | 0.024 | 0.2654 | 0.2626 | 0.186 | 0.063 | 2.9979 |
| key bias s1 | 0.020 | 0.2127 | 0.2134 | 0.164 | 0.053 | 2.8954 |
| gated s0 | 0.026 | 0.2366 | 0.2244 | 0.171 | 0.058 | 2.9024 |
| gated s1 | 0.059 | 0.2419 | 0.2368 | 0.178 | 0.059 | 3.0008 |
| gated s2 | 0.041 | 0.2530 | 0.2396 | 0.185 | 0.058 | 2.9988 |

Floors (vera_matched.json): untrained models 0.164-0.181 (my SAEs 0.19-0.21); embeddings h_0 0.160 (json 0.1597).

## 2. Status (PREREG s3, frozen), my implementation

| row | b | 95% CI | s_tot | swing upper | gated in band | status |
|---|---|---|---|---|---|---|
| h_2, build SAEs (PRIMARY) | +0.2488 | [-0.1209, +0.6184] | 0.0852 | 0.361 | 3/3 (resid -0.047, -0.050, -0.034; band 0.170) | **UNCONFIRMED** |
| h_2, my own SAEs | +0.2609 | [-0.0766, +0.5984] | 0.0778 | 0.350 | 3/3 | UNCONFIRMED |
| h_2, Hungarian one-to-one | +0.2419 | [-0.0905, +0.5743] | 0.0766 | 0.335 | 3/3 | UNCONFIRMED |

Build's results.json: b 0.24880, CI [-0.12088, 0.61848], s_tot 0.08525, swing_hi 0.36115, residuals and band identical.
Leave-one-out over the 9 non-gated models: UNCONFIRMED in 9/9 (build and own SAEs). Sink levels = 2 (one model with a
block-2 sink), so the slope is a two-group contrast, not a law (Tove #17137 wording ceiling).

## 3. Registered prediction (prereg.outcome) — FAILED in both protocols

| protocol | gated minus softmax, seeds 0/1/2 (build SAEs) | own SAEs | sink ratio gated/softmax |
|---|---|---|---|
| D8, 12k steps | -0.181 / +0.013 / -0.010 | -0.192 / +0.011 / -0.010 | 0.04 / 2.72 / 2.07 |
| registered matched-loss picks (loss 3.004-3.006, L* 3.00626) | -0.155 / -0.039 / -0.028 | -0.138 / -0.049 / -0.032 | 0.23 / 0.94 / 0.91 |

None of 6 pairs reaches +0.10. The sink-ratio clause also fails for seeds 1-2, because those standard seeds never grew a sink.

## 4. Null tests

- Shuffled latent labels (z permuted over sequences, build SAE 0): 0.053-0.096 on all 12 models, far below every recovery.
- Random dictionary (256 random unit directions, ReLU): 0.164-0.269; tracks how linearly the latents sit in h_2.
- Untrained-model SAE: 0.164-0.181. The no-sink models at h_2 (0.21-0.27) sit only 0.03-0.09 above it.
- Permutation of the sink labels across the 9 non-gated models: binary version (which model carries the one sink),
  exact p = 0.22 (build SAEs; softmax s0 ranks 2nd of 9, below softmax+1 s0) and 0.11 (own SAEs). The continuous-x
  permutation (p 0.068 / 0.007) is driven by the order of tiny x values inside the 0.020-0.061 cluster; I do not use it.
- SAE-seed spread: per-model SD 0.002-0.009 (build), pooled s_SAE 0.0055; it is far below s_tot 0.085, so the noise in
  the fit is between models (assembly block), not between SAE seeds.

## 5. Checks on the interpretation (report-only rows)

- **Co-location.** The models that gather the latents in block 2 (softmax s0, softmax+1 s0, sink logit s0) all score
  0.416-0.431 at h_2, with block-2 sink 0.604, 0.061 and 0.022. Same layer, sink present or absent, same recovery.
- **Assembly-layer row** (vera_assembly.py; layer from build's ridge file): b -0.041, CI [-0.101, +0.019] (build SAEs,
  UNCONFIRMED; build's results_assembly.json has the same b and CI); own SAEs b -0.011, swing upper 0.048 -> REJECTED(i).
- **h_4 equal-storage row**: x = block-2 sink: b -0.049 [-0.156, +0.059] UNCONFIRMED (build SAEs) / b -0.010
  [-0.058, +0.039] REJECTED(i) (own SAEs, swing upper 0.034). x = all-block mean: b -0.200 [-0.606, +0.206] / -0.047
  [-0.228, +0.135], UNCONFIRMED / REJECTED(i). The report-only rows lean to "no effect"; their status depends on SAE
  training (build vs mine), so they cannot carry a headline "No".
- **Loss covariate** (recovery ~ sink + loss, non-gated): sink coefficient +0.145 [-0.160, +0.449] (own SAEs +0.167
  [-0.114, +0.447]); loss coefficient +1.15 [+0.04, +2.26] at h_2. At h_4 loss coefficient -0.45 [-0.58, -0.31]: the
  fast-basin models (loss ~2.90) recover best at h_4.
- **Readability gap** (my ridge readout corr minus SAE best corr, held-out sequences) ~ sink: -0.021 [-0.095, +0.052];
  flat. SAE efficiency (recovery / readout corr) 0.75-0.91 in every model.

## 6. Definition checks (what the post must not stretch)

1. "At equal loss" is NOT what the primary grid measures. D8 trains every model 12k steps; held-out losses span
   2.895-3.006 (3.7%). The registered matched-loss row exists (all six pairs within 1% of L*) and also fails.
2. "Sink mass" is first-token attention only. Key bias s0 dumps 0.50 of block-3 attention on its null slot; the post
   must not say key bias "removed" the sink in that block.
3. "Law": only one model has a block-2 first-token sink. The post may say "two-group contrast", not "law" or "slope".
4. The within-softmax row in results.json ("slope 0.308, CI [0.206, 0.410]") rests on one high-leverage point
   (softmax s0); Spearman 0.40 (p 0.50); rank p for s0 = 1/5 = 0.20. That CI must not be quoted as evidence.
5. The organism and M were chosen on softmax seed 0, the only seed that grew a block-2 sink (1 of 5). Disclose.
6. C5 noise pricing was done for the pilot organism (model SD 0.01-0.03); realised s_tot is 0.085. Disclose.

## 7. Novelty

See novelty.md: no paper found that trains models differing only in the attention normaliser or gate and scores SAE
recovery of planted latents against sink mass. Closest: Gao et al. 2406.04093 F.2 (observational), Qiu et al.
2505.06708 (no SAE), 2602.14111 (random baselines, no architecture variable).
