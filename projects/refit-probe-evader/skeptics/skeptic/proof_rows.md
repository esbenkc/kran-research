| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
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
