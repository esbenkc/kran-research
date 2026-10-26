| headline | 0 | skeptic rebuild (own code, raw ctok): Rule B / logistic / QDA | none | 43.6% / 97.7% (FA 2.01%) / 93.5% (FA 1.34%) | skeptics/skeptic/vera26_recal__lam10000__seed0.json |
| headline | 1 | skeptic rebuild (own code, raw ctok): Rule B / logistic / QDA | none | 17.8% / 58.2% (FA 0.34%) / 58.6% (FA 0.34%) | skeptics/skeptic/vera26_recal__lam10000__seed1.json |
| headline | 2 | skeptic rebuild (own code, raw ctok): Rule B / logistic / QDA | none | 37.6% / 95.8% (FA 1.17%) / 79.4% (FA 1.01%) | skeptics/skeptic/vera26_recal__lam10000__seed2.json |
| headline | pooled | skeptic rebuild: Rule B / logistic / QDA, city CI | none | 33.0% [31.2, 34.8] / 83.9% [82.3, 85.6] FA 1.17% / 77.1% [74.8, 79.4] FA 0.89% | skeptics/skeptic/vera_pool.json |
| headline | pooled | skeptic rebuild: eval KL (incl. pos 0) / right country first | none | 0.003934 (0.003811 / 0.003778 / 0.004212) / 298, 298, 298 | skeptics/skeptic/vera_pool.json |
| null | pooled | skeptic rebuild: Rule B catch | line re-picked on 2000 joint resamples of the 200 trues (seed 326) | [19.6%, 39.2%] | skeptics/skeptic/vera_pool.json |
| null | 0,1,2 | skeptic rebuild: logistic head | shuffled labels, 5 permutations per seed | 0.0-2.5% / 0.0-0.8% / 0.0-2.0% | skeptics/skeptic/vera26_recal__lam10000__seed{0,1,2}.json |
| raw | 0,1,2,SG | skeptic: ctok vs tok.npy[off-2, 10..29]; own scores vs scores npz | none | max abs diff 0.0 / 0.0 on all 4 states | skeptics/skeptic/vera_behav_*.json |
| raw | 1 | skeptic: own CPU bf16 forward of the s1 adapter, 60 eval statements | none | KL 0.00409 vs 0.00412 (corr 0.87); 40/40 pairs; country-token cos >= 0.99991 at readouts 10-27 | skeptics/skeptic/vera_fwd_recal__lam10000__seed1.json |
| refit | 0,1,2,SG | skeptic own refit (GroupKFold pick of readout and C) | none | 95.1% r11 / 98.0% r12 / 98.5% r16 / 99.0% r17 | skeptics/skeptic/vera_refit_*.json |
| null | 0,1,2,SG | skeptic own refit | shuffled labels, 3 permutations | <= 4.0% | skeptics/skeptic/vera_refit_*.json |
| ceiling-stress | 0,1,2,SG | skeptic exploratory heads (C grid, QDA reg grid, HGB, MLP); best with eval FA <= 2% | none | MLP 96.8% / QDA-reg0 85.2% / QDA-reg0 97.7% / MLP 68.3% (each >= the best frozen head) | skeptics/skeptic/vera_strong_{s0,s1,s2,SG0}.json |
| mechanism | 0,1,2,SG | skeptic: Rule B catch by sentence form ('is in' / 'is not in') | none | 85.2/2.0, 22.5/13.1, 74.2/1.0, 5.0/2.3 % | skeptics/skeptic/vera_src.json |
| route | SG | skeptic rebuild: stop-grad Rule B / logistic / QDA / KL | none | 3.7% / 59.7% / 49.8% / 0.00643 | skeptics/skeptic/vera26_recalSG__lam10000__seed0.json |
