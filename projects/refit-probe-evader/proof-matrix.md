# Proof matrix: refit-probe-evader

Built by proof_matrix.py from data/ (every artifact path exists).

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| price | - | registered LEACE (seq 10..29, lw, 1400) | none | P = 0.00615 nats/token | data/leace/price.json |
| price | - | seq1028 | none | 0.00615 | data/leace/price.json |
| price | - | all_indep | none | 0.01010 | data/leace/price.json |
| price | - | everypos | none | 1.93736 | data/leace/price.json |
| price | - | gate | none | 0.00605 | data/leace/price.json |
| price | - | rand_size (same edit size, random direction) | random direction | 0.00037 | data/leace/price.json |
| price | - | unedited | identity edit | 0.00e+00 | data/leace/price.json |
| price per readout | - | single_10 | none | 0.00029 | data/leace/price.json |
| price per readout | - | single_11 | none | 0.00031 | data/leace/price.json |
| price per readout | - | single_12 | none | 0.00034 | data/leace/price.json |
| price per readout | - | single_13 | none | 0.00053 | data/leace/price.json |
| price per readout | - | single_14 | none | 0.00048 | data/leace/price.json |
| price per readout | - | single_15 | none | 0.00149 | data/leace/price.json |
| price per readout | - | single_16 | none | 0.00270 | data/leace/price.json |
| price per readout | - | single_17 | none | 0.00258 | data/leace/price.json |
| price per readout | - | single_18 | none | 0.00368 | data/leace/price.json |
| price per readout | - | single_19 | none | 0.00393 | data/leace/price.json |
| price per readout | - | single_20 | none | 0.00365 | data/leace/price.json |
| price per readout | - | single_21 | none | 0.00382 | data/leace/price.json |
| price per readout | - | single_22 | none | 0.00370 | data/leace/price.json |
| price per readout | - | single_23 | none | 0.00350 | data/leace/price.json |
| price per readout | - | single_24 | none | 0.00334 | data/leace/price.json |
| price per readout | - | single_25 | none | 0.00327 | data/leace/price.json |
| price per readout | - | single_26 | none | 0.00328 | data/leace/price.json |
| price per readout | - | single_27 | none | 0.00352 | data/leace/price.json |
| price per readout | - | single_28 | none | 0.00468 | data/leace/price.json |
| price per readout | - | single_29 | none | 0.00464 | data/leace/price.json |
| selection | - | plain | none | OOF AUROC 0.801 at R18 | data/leace/price.json |
| selection | - | lw | none | OOF AUROC 0.765 at R22 | data/leace/price.json |
| known answer | - | clean model | clean model | F1 99.7% / F2 99.5% / M1 99.3%; F1 AUROC 1.000; R29 fresh 98.0% | data/leace/guard.json |
| known answer | - | registered eraser | known-answer erasure | F1 0.7% / F2 0.2% / M1 0.3%; F1 AUROC 0.555; R29 fresh 0.3% | data/leace/guard.json |
| known answer | - | attack-600 eraser (gate) | known-answer erasure | F1 0.3% / F2 2.7% / M1 1.3%; F1 AUROC 0.809; R29 fresh 1.3% | data/leace/guard.json |
| known answer | - | 10..28 eraser | known-answer erasure | F1 0.7% / F2 0.8% / M1 0.3%; F1 AUROC 0.597; R29 fresh 0.8% | data/leace/guard.json |
| headline | 0 | refit50__lam10000__seed0 | none | F1 87.2% (FA 2.2%, AUROC 0.987), F2 78.4% (FA 0.5%) R19, M1 84.1% (FA 1.2%), frozen 7.6%, KL 0.00792, rank 298/298 | data/results/refit50__lam10000__seed0.json |
| stress headline | 0 | refit50__lam10000__seed0 QDA / MLP at R19 | none | QDA 96.1% (AUROC 0.999), MLP 97.3% (AUROC 0.999) | data/results/refit50__lam10000__seed0.json |
| stress headline | 0 | refit50__lam10000__seed0 shuffled-label fresh probe x5 | shuffled labels | mean 0.5% | data/results/refit50__lam10000__seed0.json |
| trajectory headline | 0 | refit50__lam10000__seed0 | none | slice catch by step: 0:100.0%, 50:100.0%, 100:95.0%, 150:90.0%, 200:88.0%, 250:84.0%, 300:84.0% | data/runs/refit50__lam10000__seed0.json |
| headline | 1 | refit50__lam10000__seed1 | none | F1 85.4% (FA 1.7%, AUROC 0.991), F2 83.7% (FA 0.7%) R19, M1 61.4% (FA 0.3%), frozen 9.4%, KL 0.00702, rank 298/298 | data/results/refit50__lam10000__seed1.json |
| stress headline | 1 | refit50__lam10000__seed1 QDA / MLP at R19 | none | QDA 95.8% (AUROC 0.999), MLP 94.5% (AUROC 0.999) | data/results/refit50__lam10000__seed1.json |
| stress headline | 1 | refit50__lam10000__seed1 shuffled-label fresh probe x5 | shuffled labels | mean 0.7% | data/results/refit50__lam10000__seed1.json |
| trajectory headline | 1 | refit50__lam10000__seed1 | none | slice catch by step: 0:100.0%, 50:100.0%, 100:98.0%, 150:91.0%, 200:83.0%, 250:84.0%, 300:68.0% | data/runs/refit50__lam10000__seed1.json |
| headline | 2 | refit50__lam10000__seed2 | none | F1 63.6% (FA 1.3%, AUROC 0.986), F2 80.0% (FA 0.3%) R19, M1 70.3% (FA 0.8%), frozen 17.4%, KL 0.00761, rank 298/298 | data/results/refit50__lam10000__seed2.json |
| stress headline | 2 | refit50__lam10000__seed2 QDA / MLP at R19 | none | QDA 97.8% (AUROC 0.999), MLP 97.1% (AUROC 0.999) | data/results/refit50__lam10000__seed2.json |
| stress headline | 2 | refit50__lam10000__seed2 shuffled-label fresh probe x5 | shuffled labels | mean 0.8% | data/results/refit50__lam10000__seed2.json |
| trajectory headline | 2 | refit50__lam10000__seed2 | none | slice catch by step: 0:100.0%, 50:100.0%, 100:97.0%, 150:92.0%, 200:88.0%, 250:81.0%, 300:78.0% | data/runs/refit50__lam10000__seed2.json |
| headline pooled | pooled | base | none | F1 78.7% [76.1%, 81.4%], F2 80.7%, M1 71.9%, KL 0.00752 = 1.22 x P | data/results/summary.json |
| 10x KL weight | 0 | refit50__lam100000__seed0 | none | F1 99.7% (FA 1.0%, AUROC 1.000), F2 99.3% (FA 0.7%) R16, M1 99.8% (FA 1.5%), frozen 98.8%, KL 0.00382, rank 298/298 | data/results/refit50__lam100000__seed0.json |
| stress 10x KL weight | 0 | refit50__lam100000__seed0 QDA / MLP at R16 | none | QDA 99.7% (AUROC 1.000), MLP 99.7% (AUROC 0.999) | data/results/refit50__lam100000__seed0.json |
| stress 10x KL weight | 0 | refit50__lam100000__seed0 shuffled-label fresh probe x5 | shuffled labels | mean 2.1% | data/results/refit50__lam100000__seed0.json |
| trajectory 10x KL weight | 0 | refit50__lam100000__seed0 | none | slice catch by step: 0:100.0%, 50:100.0%, 100:100.0%, 150:100.0%, 200:100.0%, 250:100.0%, 300:100.0% | data/runs/refit50__lam100000__seed0.json |
| 10x KL weight pooled | pooled | x10 | none | F1 99.7% [99.2%, 100.0%], F2 99.3%, M1 99.8%, KL 0.00382 = 0.62 x P | data/results/summary.json |
| shuffled-label control | 0 | shuffled__lam10000__seed0__ps44.7 | shuffled labels | F1 99.5% (FA 1.5%, AUROC 1.000), F2 99.7% (FA 0.8%) R16, M1 99.3% (FA 1.0%), frozen 99.5%, KL 0.00503, rank 298/298 | data/results/shuffled__lam10000__seed0__ps44.7.json |
| stress shuffled-label control | 0 | shuffled__lam10000__seed0__ps44.7 QDA / MLP at R16 | none | QDA 99.7% (AUROC 1.000), MLP 99.0% (AUROC 0.999) | data/results/shuffled__lam10000__seed0__ps44.7.json |
| stress shuffled-label control | 0 | shuffled__lam10000__seed0__ps44.7 shuffled-label fresh probe x5 | shuffled labels | mean 1.2% | data/results/shuffled__lam10000__seed0__ps44.7.json |
| trajectory shuffled-label control | 0__ps44.7 | shuffled__lam10000__seed0__ps44.7 | none | slice catch by step: 0:1.0%, 50:1.0%, 100:0.0%, 150:0.0%, 200:1.0%, 250:0.0%, 300:0.0% | data/runs/shuffled__lam10000__seed0__ps44.7.json |
| shuffled-label control pooled | pooled | N | none | F1 99.5% [98.8%, 100.0%], F2 99.7%, M1 99.3%, KL 0.00503 = 0.82 x P | data/results/summary.json |
| mean-gap side arm | 0 | meangap__lam10000__seed0 | none | F1 98.8% (FA 1.3%, AUROC 0.999), F2 98.5% (FA 0.7%) R18, M1 99.2% (FA 1.3%), frozen 99.3%, KL 0.00625, rank 298/298 | data/results/meangap__lam10000__seed0.json |
| stress mean-gap side arm | 0 | meangap__lam10000__seed0 QDA / MLP at R18 | none | QDA 98.3% (AUROC 1.000), MLP 98.2% (AUROC 1.000) | data/results/meangap__lam10000__seed0.json |
| stress mean-gap side arm | 0 | meangap__lam10000__seed0 shuffled-label fresh probe x5 | shuffled labels | mean 0.6% | data/results/meangap__lam10000__seed0.json |
| trajectory mean-gap side arm | 0 | meangap__lam10000__seed0 | none | slice catch by step: 0:100.0%, 50:99.0%, 100:99.0%, 150:99.0%, 200:100.0%, 250:99.0%, 300:99.0% | data/runs/meangap__lam10000__seed0.json |
| mean-gap side arm pooled | pooled | G | none | F1 98.8% [97.8%, 99.7%], F2 98.5%, M1 99.2%, KL 0.00625 = 1.02 x P | data/results/summary.json |
| 900-step run at 900 | 0 | refit50__lam10000__seed0__steps900 | none | F1 53.2% (FA 1.0%, AUROC 0.985), F2 57.7% (FA 0.2%) R19, M1 37.6% (FA 0.8%), frozen 4.4%, KL 0.01160, rank 298/298 | data/results/refit50__lam10000__seed0__steps900.json |
| stress 900-step run at 900 | 0 | refit50__lam10000__seed0__steps900 QDA / MLP at R19 | none | QDA 97.7% (AUROC 1.000), MLP 98.8% (AUROC 1.000) | data/results/refit50__lam10000__seed0__steps900.json |
| stress 900-step run at 900 | 0 | refit50__lam10000__seed0__steps900 shuffled-label fresh probe x5 | shuffled labels | mean 0.8% | data/results/refit50__lam10000__seed0__steps900.json |
| trajectory 900-step run at 900 | 0__steps900 | refit50__lam10000__seed0__steps900 | none | slice catch by step: 0:100.0%, 50:100.0%, 100:95.0%, 150:90.0%, 200:88.0%, 250:84.0%, 300:84.0%, 350:79.0%, 400:79.0%, 450:69.0%, 500:76.0%, 550:58.0%, 600:74.0%, 650:39.0%, 700:67.0%, 750:71.0%, 800:68.0%, 850:31.0%, 900:59.0% | data/runs/refit50__lam10000__seed0__steps900.json |
| 900-step run at 900 pooled | pooled | long900 | none | F1 53.2% [49.3%, 57.0%], F2 57.7%, M1 37.6%, KL 0.01160 = 1.89 x P | data/results/summary.json |
| status | pooled | PREREG s6 | none | REJECTED ['holds', 'price is LoRA machinery'] | data/results/summary.json |
| prediction | pooled | PREREG s7 | none | failed | data/results/summary.json |
| headline | 0 | refit50 lam1e4, Vera rebuild | none | M1 84.06%, F1 87.25%, F2 78.36% (r19), frozen 7.55%; KL bf16 0.007928 | skeptics/skeptic/vera_state_R1.json |
| headline | 1 | refit50 lam1e4, Vera rebuild | none | M1 61.41%, F1 85.57%, F2 83.56% (r19), frozen 9.40%; KL bf16 0.007032 | skeptics/skeptic/vera_state_R3.json |
| headline | 2 | refit50 lam1e4, Vera rebuild | none | M1 70.30%, F1 63.59%, F2 80.03% (r19), frozen 17.45%; KL bf16 0.007622 | skeptics/skeptic/vera_state_R4.json |
| headline | 0-2 | pooled, Vera rebuild (joint city bootstrap) | none | M1 71.92% [68.5, 75.1], F1 78.80% [76.2, 81.4], F2 80.65% [77.3, 83.7]; KL 0.007527 = 1.225 x P | skeptics/skeptic/vera_pool.json |
| headline | 0-2 | Vera rebuild | shuffled-label fresh probe (5 perms per seed) | 0.0-1.3% caught (R1/R3/R4 files) | skeptics/skeptic/vera_state_R1.json |
| headline | 0-2 | Vera rebuild | random direction at F2 readout | <= 0.3% caught (R1/R3/R4 files) | skeptics/skeptic/vera_state_R3.json |
| price | - | Vera own LEACE, CPU fp32, linear shrinkage | none | P = 0.005945 (build 0.006145, -3.3%) | skeptics/skeptic/vera_kl_seq_fp32_lw.json |
| price | - | Vera own LEACE, plain covariance | estimator sensitivity | 0.0209 (build 0.0214) | skeptics/skeptic/vera_kl_seq_fp32.json |
| price | - | Vera, known answer | gate eraser (attack-600) fresh probes | F1 0.34% (AUROC 0.808), F2 2.68% | skeptics/skeptic/vera_state_leace_gate.json |
| price | 0 | Vera, dtype | fp32 forward of R1 | KL 0.00702 (bf16 0.00793): ~0.0009 nats/token of bf16 rounding | skeptics/skeptic/vera_statekl_R1.json |
| price | - | Vera, known answer | 1026 recal s0 adapter (saved KL 0.003811) | bf16 0.003806, fp32 0.003076 | skeptics/skeptic/vera_statekl_dry1026_bf16mps.json |
| dose | 0 | refit50 lam1e5, Vera rebuild | 10x KL weight | M1 99.83%, F1 99.66%, F2 99.33%, frozen 98.83%; KL 0.003827 | skeptics/skeptic/vera_state_R2.json |
| side | 0 | mean-gap G, Vera rebuild | stronger-attacker side arm | M1 99.16%, F1 98.83%, F2 98.49%, frozen 99.33%; gap 0.59x, 14 deg | skeptics/skeptic/vera_state_G.json |
| machinery | 0 | shuffled-label N x44.7, Vera rebuild | no-concept control | M1 99.33%, F1 99.50%, F2 99.50%, frozen 99.50% | skeptics/skeptic/vera_state_N.json |
