# Proof matrix: gated-sinks

One row per cell. Built by proof_matrix.py from results/results.json; every artifact path is relative to this folder.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline (law fit) | 0 | kbias | none | sink 0.024, recovery 0.265 (SAE seeds 0.267, 0.260, 0.269), loss 2.9979 (gap to L* 0.28%) | runs/kbias_s0/measure_L2_m256_k8_c12000.json |
| headline (law fit) | 1 | kbias | none | sink 0.020, recovery 0.213 (SAE seeds 0.218, 0.211, 0.209), loss 2.8954 (gap to L* 3.69%) | runs/kbias_s1/measure_L2_m256_k8_c12000.json |
| headline (law fit) | 0 | sinklogit | none | sink 0.022, recovery 0.416 (SAE seeds 0.412, 0.417, 0.419), loss 3.0026 (gap to L* 0.12%) | runs/sinklogit_s0/measure_L2_m256_k8_c12000.json |
| headline (law fit) | 0 | softmax | none | sink 0.604, recovery 0.418 (SAE seeds 0.414, 0.422, 0.417), loss 3.0007 (gap to L* 0.18%) | runs/softmax_s0/measure_L2_m256_k8_c12000.json |
| headline (law fit) | 1 | softmax | none | sink 0.022, recovery 0.228 (SAE seeds 0.234, 0.223, 0.228), loss 2.9052 (gap to L* 3.36%) | runs/softmax_s1/measure_L2_m256_k8_c12000.json |
| headline (law fit) | 2 | softmax | none | sink 0.020, recovery 0.263 (SAE seeds 0.256, 0.272, 0.260), loss 3.0063 (gap to L* -0.00%) | runs/softmax_s2/measure_L2_m256_k8_c12000.json |
| headline (law fit) | 3 | softmax | none | sink 0.023, recovery 0.231 (SAE seeds 0.228, 0.237, 0.228), loss 2.9035 (gap to L* 3.42%) | runs/softmax_s3/measure_L2_m256_k8_c12000.json |
| headline (law fit) | 4 | softmax | none | sink 0.022, recovery 0.230 (SAE seeds 0.235, 0.229, 0.228), loss 2.9042 (gap to L* 3.39%) | runs/softmax_s4/measure_L2_m256_k8_c12000.json |
| headline (law fit) | 0 | softmax1 | none | sink 0.061, recovery 0.431 (SAE seeds 0.433, 0.422, 0.437), loss 3.0023 (gap to L* 0.13%) | runs/softmax1_s0/measure_L2_m256_k8_c12000.json |
| headline (law fit predicted) | 0 | gated | none | sink 0.026, recovery 0.237 (SAE seeds 0.241, 0.243, 0.226), loss 2.9024 (gap to L* 3.46%) | runs/gated_s0/measure_L2_m256_k8_c12000.json |
| headline (law fit predicted) | 1 | gated | none | sink 0.059, recovery 0.242 (SAE seeds 0.243, 0.245, 0.237), loss 3.0008 (gap to L* 0.18%) | runs/gated_s1/measure_L2_m256_k8_c12000.json |
| headline (law fit predicted) | 2 | gated | none | sink 0.041, recovery 0.253 (SAE seeds 0.255, 0.252, 0.252), loss 2.9988 (gap to L* 0.25%) | runs/gated_s2/measure_L2_m256_k8_c12000.json |
| headline (line) | all non-gated | OLS fit | none | b = 0.249, 95% CI [-0.121, 0.618], s_tot 0.085, R 0.584, swing_hi 0.361, sink levels 2 | results/results.json |
| headline (out-of-family) | 0 | gated | none | measured 0.237 vs predicted 0.283, band 0.170, in band: True | results/results.json |
| headline (out-of-family) | 1 | gated | none | measured 0.242 vs predicted 0.291, band 0.170, in band: True | results/results.json |
| headline (out-of-family) | 2 | gated | none | measured 0.253 vs predicted 0.287, band 0.170, in band: True | results/results.json |
| status | all | PREREG s3 rule | none | UNCONFIRMED []; prediction failed | results/results.json |
| null: shuffled labels | 0 | softmax | shuffled labels | recovery 0.101 | runs/softmax_s0/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 0 | softmax | random dictionary | recovery 0.273 | runs/softmax_s0/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 1 | softmax | shuffled labels | recovery 0.058 | runs/softmax_s1/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 1 | softmax | random dictionary | recovery 0.173 | runs/softmax_s1/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 2 | softmax | shuffled labels | recovery 0.062 | runs/softmax_s2/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 2 | softmax | random dictionary | recovery 0.188 | runs/softmax_s2/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 3 | softmax | shuffled labels | recovery 0.059 | runs/softmax_s3/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 3 | softmax | random dictionary | recovery 0.168 | runs/softmax_s3/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 4 | softmax | shuffled labels | recovery 0.058 | runs/softmax_s4/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 4 | softmax | random dictionary | recovery 0.172 | runs/softmax_s4/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 0 | softmax1 | shuffled labels | recovery 0.098 | runs/softmax1_s0/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 0 | softmax1 | random dictionary | recovery 0.267 | runs/softmax1_s0/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 0 | sinklogit | shuffled labels | recovery 0.092 | runs/sinklogit_s0/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 0 | sinklogit | random dictionary | recovery 0.258 | runs/sinklogit_s0/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 0 | kbias | shuffled labels | recovery 0.065 | runs/kbias_s0/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 0 | kbias | random dictionary | recovery 0.186 | runs/kbias_s0/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 1 | kbias | shuffled labels | recovery 0.056 | runs/kbias_s1/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 1 | kbias | random dictionary | recovery 0.152 | runs/kbias_s1/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 0 | gated | shuffled labels | recovery 0.058 | runs/gated_s0/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 0 | gated | random dictionary | recovery 0.177 | runs/gated_s0/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 1 | gated | shuffled labels | recovery 0.063 | runs/gated_s1/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 1 | gated | random dictionary | recovery 0.181 | runs/gated_s1/measure_L2_m256_k8_c12000.json |
| null: shuffled labels | 2 | gated | shuffled labels | recovery 0.063 | runs/gated_s2/measure_L2_m256_k8_c12000.json |
| null: random dictionary | 2 | gated | random dictionary | recovery 0.177 | runs/gated_s2/measure_L2_m256_k8_c12000.json |
| floor: untrained model | 0 | softmax | untrained model | recovery 0.177 | runs/softmax_s0/measure_untrained_L2_m256_k8.json |
| floor: untrained model | 0 | softmax1 | untrained model | recovery 0.181 | runs/softmax1_s0/measure_untrained_L2_m256_k8.json |
| floor: untrained model | 0 | sinklogit | untrained model | recovery 0.181 | runs/sinklogit_s0/measure_untrained_L2_m256_k8.json |
| floor: untrained model | 0 | kbias | untrained model | recovery 0.176 | runs/kbias_s0/measure_untrained_L2_m256_k8.json |
| floor: untrained model | 0 | gated | untrained model | recovery 0.164 | runs/gated_s0/measure_untrained_L2_m256_k8.json |
| floor: embeddings h_0 | 0 | softmax | embeddings only | recovery 0.160 | runs/softmax_s0/measure_embed_L0.json |
| same architecture, different sink | 5 softmax seeds | softmax | none | slope 0.308 [0.206, 0.410], Spearman 0.40 | results/results.json |
| loss covariate | all non-gated | recovery ~ sink + loss | none | b 0.145 [-0.160, 0.449], c 1.151 | results/results.json |
| readability gap | all non-gated | readout corr - SAE corr ~ sink | none | b -0.000 [-0.061, 0.061] | results/results.json |
| h_4 equal storage, x = block-2 sink | all non-gated | OLS | none | b -0.049 [-0.156, 0.059], status UNCONFIRMED [], prediction failed | results/results_d8_L4_m256_k8_sink_blk2.json |
| h_4 equal storage, x = all-block mean | all non-gated | OLS | none | b -0.200 [-0.606, 0.206], status UNCONFIRMED [], prediction failed | results/results_d8_L4_m256_k8_sink_mass_all.json |
| registered matched-loss row | all non-gated | OLS | none | b 0.715 [-0.825, 2.255], status UNCONFIRMED [], prediction failed | results/results_matched_L2_m256_k8_sink_mass.json |
| assembly layer | all non-gated | OLS | none | b -0.041 [-0.101, 0.019] | results/results_assembly.json |
| hiding check: RMS-normalised SAE | 0 | gated | none | recovery 0.231 | runs/gated_s0/measure_L2_m256_k8_rms_c12000.json |
| hiding check: RMS-normalised SAE | 1 | gated | none | recovery 0.247 | runs/gated_s1/measure_L2_m256_k8_rms_c12000.json |
| hiding check: RMS-normalised SAE | 2 | gated | none | recovery 0.257 | runs/gated_s2/measure_L2_m256_k8_rms_c12000.json |
| hiding check: RMS-normalised SAE | 0 | softmax | none | recovery 0.435 | runs/softmax_s0/measure_L2_m256_k8_rms_c12000.json |
| hiding check: RMS-normalised SAE | 1 | softmax | none | recovery 0.226 | runs/softmax_s1/measure_L2_m256_k8_rms_c12000.json |
| hiding check: RMS-normalised SAE | 2 | softmax | none | recovery 0.273 | runs/softmax_s2/measure_L2_m256_k8_rms_c12000.json |
| SAE k sensitivity (k=16) | 0 | gated | none | recovery 0.231 | runs/gated_s0/measure_L2_m256_k16_c12000.json |
| SAE k sensitivity (k=4) | 0 | gated | none | recovery 0.226 | runs/gated_s0/measure_L2_m256_k4_c12000.json |
| SAE k sensitivity (k=16) | 0 | softmax | none | recovery 0.447 | runs/softmax_s0/measure_L2_m256_k16_c12000.json |
| SAE k sensitivity (k=4) | 0 | softmax | none | recovery 0.369 | runs/softmax_s0/measure_L2_m256_k4_c12000.json |
| sink heads carry latents | softmax_s0 | softmax | head zeroed | readout 0.264, drops 0.021, 0.023, 0.021, 0.025 | results/ablate_softmax_s0_12000.json |

<!-- skeptic rows below this line are kept by proof_matrix.py -->