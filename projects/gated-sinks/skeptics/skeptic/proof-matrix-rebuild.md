| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline rebuild (Vera) | all 12 | h_2, 12k | none | sink + recovery of all 12 models match measure json to 4 dp (own code, raw npz) | skeptics/skeptic/vera_grid_h2_final.json |
| headline rebuild (Vera) | all non-gated | OLS, build SAEs | none | b +0.2488 [-0.1209, +0.6184], s_tot 0.0852, swing upper 0.361, gated 3/3 in band 0.170: UNCONFIRMED | skeptics/skeptic/vera_grid_h2_final.json |
| headline rebuild (Vera) | all non-gated | OLS, Vera's own TopK SAEs | none | b +0.2609 [-0.0766, +0.5984]: UNCONFIRMED | skeptics/skeptic/vera_grid_h2_final.json |
| headline rebuild (Vera) | all non-gated | Hungarian one-to-one | none | b +0.2419 [-0.0905, +0.5743]: UNCONFIRMED | skeptics/skeptic/vera_grid_h2_final.json |
| headline robustness (Vera) | leave one out x 9 | OLS | drop one model | UNCONFIRMED in 9/9 | skeptics/skeptic/vera_grid_h2_final.json |
| null: sink-label permutation (Vera) | all non-gated | binary (which model has the sink) | permuted labels | exact p 0.22 (build SAEs), 0.11 (own SAEs) | skeptics/skeptic/vera_grid_h2_final.json |
| registered prediction rebuild (Vera) | 0, 1, 2 | gated - softmax, 12k | none | -0.181 / +0.013 / -0.010: FAILED | skeptics/skeptic/vera_grid_h2_final.json |
| registered prediction rebuild (Vera) | 0, 1, 2 | gated - softmax, matched-loss picks | none | -0.155 / -0.039 / -0.028: FAILED | skeptics/skeptic/vera_matched.json |
| floors rebuild (Vera) | 0 | untrained softmax / gated / kbias / sinklogit, embeddings | untrained model | 0.177 / 0.164 / 0.176 / 0.181; h_0 0.160 | skeptics/skeptic/vera_matched.json |
| assembly-layer rebuild (Vera) | all | own layer per model | none | b -0.041 [-0.101, +0.019] (build SAEs); -0.011, swing upper 0.048 (own SAEs) | skeptics/skeptic/vera_assembly.json |
| h_4 rebuild (Vera) | all | x = block-2 sink | none | b -0.049 [-0.156, +0.059] (build SAEs); -0.010 [-0.058, +0.039] (own) | skeptics/skeptic/vera_grid_h4_blk2.json |
