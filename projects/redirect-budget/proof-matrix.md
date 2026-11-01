# proof matrix: redirect-budget

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | mlp8_none_s0 | Qwen3-1.7B | none | carrier 26.2 in RANK top-3: miss | data/carriers/mlp8_none_s0.json |
| headline | mlp8_none_s1 | Qwen3-1.7B | none | carrier undefined in RANK top-3: miss | data/carriers/mlp8_none_s1.json |
| headline | mlp8_none_s2 | Qwen3-1.7B | none | carrier 27.15 in RANK top-3: miss | data/carriers/mlp8_none_s2.json |
| headline | grader_s0 | Qwen3-0.6B | none | carrier 27.9 in RANK top-3: miss | data/carriers/grader_s0.json |
| headline | grader_s1 | Qwen3-0.6B | none | carrier undefined in RANK top-3: miss | data/carriers/grader_s1.json |
| headline | monitor_s0 | Qwen3-0.6B | none | carrier 27.8 in RANK top-3: miss | data/carriers/monitor_s0.json |
| headline | monitor_s1 | Qwen3-0.6B | none | carrier 27.8 in RANK top-3: miss | data/carriers/monitor_s1.json |
| rank-top3 | n/a | Qwen3-1.7B | none | RANK top-3 ['25.1', '20.0', '27.1']; heads within 5% of 3rd: 7; chord saturated share 1.000 | data/bounds/rank_17.npz |
| rank-top3 | n/a | Qwen3-0.6B | none | RANK top-3 ['27.3', '27.1', '27.2']; heads within 5% of 3rd: 3; chord saturated share 1.000 | data/bounds/rank_06_monitor.npz |
| headline | all 7 | both models | none | H = 0 of 7; status REJECTED | data/summary.json |
| baseline | all 7 | both models | last-layer random 3-set (exact) | P(hits >= H) = 1.0000 | data/summary.json |
| baseline | all 7 | both models | base_sink_mass | hits 0 (RANK 0); same top-3 as RANK: False | data/summary.json |
| baseline | all 7 | both models | opq | hits 1 (RANK 0); same top-3 as RANK: False | data/summary.json |
| baseline | all 7 | both models | opk | hits 0 (RANK 0); same top-3 as RANK: False | data/summary.json |
| baseline | all 7 | both models | sink_divergence | hits 0 (RANK 0); same top-3 as RANK: False | data/summary.json |
| baseline | all 7 | both models | reported row RANK0 | hits 0 | data/summary.json |
| baseline | all 7 | both models | reported row RANK_Q | hits 0 | data/summary.json |
| baseline | all 7 | both models | reported row RANK_C | hits 0 | data/summary.json |
| baseline | all 7 | both models | reported row PROXY | hits 0 | data/summary.json |
| carrier | mlp8_none_s0 | Qwen3-1.7B | none | 26.2 share 0.429, 2nd 26.0 0.170, ratio 2.52: defined | data/carriers/mlp8_none_s0.json |
| carrier | mlp8_none_s1 | Qwen3-1.7B | none | 27.12 share 0.508, 2nd 27.7 0.429, ratio 1.18: UNDEFINED | data/carriers/mlp8_none_s1.json |
| carrier | mlp8_none_s2 | Qwen3-1.7B | none | 27.15 share 0.717, 2nd 27.12 0.168, ratio 4.26: defined | data/carriers/mlp8_none_s2.json |
| carrier | grader_s0 | Qwen3-0.6B | none | 27.9 share 0.803, 2nd 27.1 0.013, ratio 62.72: defined | data/carriers/grader_s0.json |
| carrier | grader_s1 | Qwen3-0.6B | none | 27.9 share 0.305, 2nd 26.0 0.270, ratio 1.13: UNDEFINED | data/carriers/grader_s1.json |
| carrier | monitor_s0 | Qwen3-0.6B | none | 27.8 share 1.012, 2nd 26.15 0.003, ratio 310.23: defined | data/carriers/monitor_s0.json |
| carrier | monitor_s1 | Qwen3-0.6B | none | 27.8 share 1.047, 2nd 26.6 0.000, ratio 2181.46: defined | data/carriers/monitor_s1.json |
| carrier | mlp8_none_s3 | Qwen3-1.7B | none | 27.15 share 0.369, 2nd 27.9 0.283, ratio 1.30: defined | data/carriers/mlp8_none_s3.json |
| carrier | mlp8_none_s4 | Qwen3-1.7B | none | 27.1 share 0.557, 2nd 27.6 0.308, ratio 1.81: defined | data/carriers/mlp8_none_s4.json |
| scout-C | all 7 | row S_tv | random 3-sets (50 draws) | C1 0.986; C2 False; C3 hits 0; C4 share 1.00; holds False | data/summary.json |
| scout-C | all 7 | row S_logit | random 3-sets (50 draws) | C1 0.980; C2 False; C3 hits 0; C4 share 1.00; holds False | data/summary.json |
| scout-C | all 7 | row Q_tv | random 3-sets (50 draws) | C1 0.994; C2 False; C3 hits 1; C4 share 0.26; holds False | data/summary.json |
| scout-C | all 7 | row Q_logit | random 3-sets (50 draws) | C1 0.992; C2 False; C3 hits 1; C4 share 0.26; holds False | data/summary.json |
| scout-C | all 7 | row K_tv | random 3-sets (50 draws) | C1 1.000; C2 True; C3 hits 1; C4 share 0.26; holds False | data/summary.json |
| scout-C | all 7 | row K_logit | random 3-sets (50 draws) | C1 1.000; C2 False; C3 hits 1; C4 share 0.26; holds False | data/summary.json |
| scout-C | all 7 | row C_tv | random 3-sets (50 draws) | C1 1.000; C2 True; C3 hits 0; C4 share 1.00; holds False | data/summary.json |
| scout-C | all 7 | row C_logit | random 3-sets (50 draws) | C1 1.000; C2 False; C3 hits 1; C4 share 0.26; holds False | data/summary.json |
| validity | mlp8_none_s0 | row C, logit space | none | cell share under 1.0000, max ratio 0.017 | data/measured/mlp8_none_s0.npz |
| validity | mlp8_none_s1 | row C, logit space | none | cell share under 1.0000, max ratio 0.032 | data/measured/mlp8_none_s1.npz |
| validity | mlp8_none_s2 | row C, logit space | none | cell share under 1.0000, max ratio 0.014 | data/measured/mlp8_none_s2.npz |
| validity | grader_s0 | row C, logit space | none | cell share under 1.0000, max ratio 0.047 | data/measured/grader_s0.npz |
| validity | grader_s1 | row C, logit space | none | cell share under 1.0000, max ratio 0.024 | data/measured/grader_s1.npz |
| validity | monitor_s0 | row C, logit space | none | cell share under 1.0000, max ratio 0.189 | data/measured/monitor_s0.npz |
| validity | monitor_s1 | row C, logit space | none | cell share under 1.0000, max ratio 0.183 | data/measured/monitor_s1.npz |
| forward | mlp8_none_s3 | Qwen3-1.7B unseen run | none | carrier 27.15: miss | data/carriers/mlp8_none_s3.json |
| forward | mlp8_none_s4 | Qwen3-1.7B unseen run | none | carrier 27.1: hit | data/carriers/mlp8_none_s4.json |
| gate | n/a | J0 block-20 cache none (sha256 vs manifest_raw.json) | none | 0d4c1cea...e2cd8 = match | src1027/data/manifest_raw.json |
| gate | n/a | J0 block-20 cache probe (sha256 vs manifest_raw.json) | none | fd42d010...fa4e = match | src1027/data/manifest_raw.json |
| gate | n/a | carrier tail reproduces stored test score mlp8_none_s0 | none | strict [1.0, 1.0], shift_mean [24.33203125, 24.33203125] | data/carriers/mlp8_none_s0.json |
| gate | n/a | carrier tail reproduces stored test score mlp8_none_s1 | none | strict [1.0, 1.0], shift_mean [20.56689453125, 20.56689453125] | data/carriers/mlp8_none_s1.json |
| gate | n/a | carrier tail reproduces stored test score mlp8_none_s2 | none | strict [1.0, 1.0], shift_mean [25.008544921875, 25.008544921875] | data/carriers/mlp8_none_s2.json |
| gate | n/a | carrier tail reproduces sink-value P_on/P_off grader_s0 | none | P_on 0.122683 vs 0.122683; P_off 0.567929 vs 0.567929 | data/carriers/grader_s0.json |
| gate | n/a | carrier tail reproduces sink-value P_on/P_off grader_s1 | none | P_on 0.124217 vs 0.124217; P_off 0.347740 vs 0.347740 | data/carriers/grader_s1.json |
| gate | n/a | carrier tail reproduces sink-value P_on/P_off monitor_s0 | none | P_on 0.019854 vs 0.019854; P_off 0.724733 vs 0.724733 | data/carriers/monitor_s0.json |
| gate | n/a | carrier tail reproduces sink-value P_on/P_off monitor_s1 | none | P_on 0.059673 vs 0.059673; P_off 0.611169 vs 0.611169 | data/carriers/monitor_s1.json |
| gate | n/a | carrier code path reproduces saved audit (s0 26.2 dev removed) | none | 0.429193 vs 0.4291932 (src1027/data/audit/mlp8_none_s0.json) | data/carriers/mlp8_none_s0.json |
| headline | all 7 | both models | skeptic rebuild, fp32 own forward | H 0 of 5 defined, places 120/50/21/11 | skeptics/skeptic/out/status.json |
| carrier | mlp8_none_s1 | Qwen3-1.7B | skeptic rebuild, fp32 own forward | 27.12 0.512 vs 24.12 0.433, ratio 1.18: undefined (same verdict) | skeptics/skeptic/out/r17_mlp8_none_s1.json |
| carrier | all 0.6B runs | Qwen3-0.6B | skeptic rebuild, fp32 own forward | same argmax + defined verdicts (grader_s1 1.094 undefined) | skeptics/skeptic/out/r06.json |
| lemma | n/a | both models | PGD known-answer | max achieved/bound Q 0.677, K 0.806 (1,024 cells) | skeptics/skeptic/out/pgd2.json |
| null | all 7 | both models | random heads (100,000 draws) | carriers' mean place share 0.582, P(<=) 0.69 | skeptics/skeptic/out/claims.json |
