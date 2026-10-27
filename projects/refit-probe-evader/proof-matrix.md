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
| headline pooled | pooled | base | none | F1 87.2% [84.4%, 89.8%], F2 78.4%, M1 84.1%, KL 0.00792 = 1.29 x P | data/results/summary.json |
| 10x KL weight | 0 | refit50__lam100000__seed0 | none | F1 99.7% (FA 1.0%, AUROC 1.000), F2 99.3% (FA 0.7%) R16, M1 99.8% (FA 1.5%), frozen 98.8%, KL 0.00382, rank 298/298 | data/results/refit50__lam100000__seed0.json |
| stress 10x KL weight | 0 | refit50__lam100000__seed0 QDA / MLP at R16 | none | QDA 99.7% (AUROC 1.000), MLP 99.7% (AUROC 0.999) | data/results/refit50__lam100000__seed0.json |
| stress 10x KL weight | 0 | refit50__lam100000__seed0 shuffled-label fresh probe x5 | shuffled labels | mean 2.1% | data/results/refit50__lam100000__seed0.json |
| trajectory 10x KL weight | 0 | refit50__lam100000__seed0 | none | slice catch by step: 0:100.0%, 50:100.0%, 100:100.0%, 150:100.0%, 200:100.0%, 250:100.0%, 300:100.0% | data/runs/refit50__lam100000__seed0.json |
| 10x KL weight pooled | pooled | x10 | none | F1 99.7% [99.2%, 100.0%], F2 99.3%, M1 99.8%, KL 0.00382 = 0.62 x P | data/results/summary.json |
| status | pooled | PREREG s6 | none | UNCONFIRMED  | data/results/summary.json |
| prediction | pooled | PREREG s7 | none | None | data/results/summary.json |
