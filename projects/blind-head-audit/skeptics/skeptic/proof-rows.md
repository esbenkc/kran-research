| headline | s0 | L fp32 tail rebuild (Vera) | none | R 0.323 < 0.70 -> FAIL (bf16 0.305) | skeptics/skeptic/out/push_L_s0.json |
| headline | s1 | L fp32 tail rebuild (Vera) | 378-set exhaustive 27.7 pool | R 0.9086 < exact p95 >= 0.9176 -> FAIL (bf16 0.898 vs 0.920) | skeptics/skeptic/out/push_L_s1.json |
| headline | s2 | L fp32 tail rebuild (Vera) | 100 random 3-sets | R 0.9905 > p95 0.013 -> PASS | skeptics/skeptic/out/push_L_s2.json |
| headline | s0 | S fp32 tail rebuild (Vera) | 100 random 3-sets | R 0.827 > p95 0.270 -> PASS | skeptics/skeptic/out/push_S_s0.json |
| headline | s1 | S fp32 tail rebuild (Vera) | 100 random 3-sets | R 0.933 > p95 0.220 -> PASS | skeptics/skeptic/out/push_S_s1.json |
| headline | s2 | S fp32 tail rebuild (Vera) | 100 random 3-sets | R 0.882 > p95 0.155 -> PASS | skeptics/skeptic/out/push_S_s2.json |
| headline | s0 | RL fp32 tail rebuild (Vera) | 400 random 3-sets | R 0.9986 > p95 0.017 -> PASS | skeptics/skeptic/out/push_RL_s0.json |
| headline | s1 | RL fp32 tail rebuild (Vera) | 400 random 3-sets | R 0.9976 > p95 0.0006 -> PASS | skeptics/skeptic/out/push_RL_s1.json |
| blind picks | all | 8 switch + 3 null: fp32 score rebuild (Vera) | 4 sign-flip fakes per model | top-3 sets = picks_fp32 in 8/8 | skeptics/skeptic/out/picks17.json |
| null flags | all | L_never, RL_tag, RL_shuf: fp32 (Vera) | 40 sign-flip fakes | P(flag) 0.73 / 0.66 / 0.00 | skeptics/skeptic/out/flagp06n40.txt |
| D6 overlap | all | 6 label models: 105 passages (Vera) | overlap removed | same top-3 set + top-1 flag in 6/6 | skeptics/skeptic/out/subset105.json |
