# Proof matrix — blind-head-audit

Every cell is rebuilt by `summary.py` from the JSON named in its row.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | s0 | L blind top-3 26.5/27.12/26.2 | none | R = 0.305 (95% CI 0.292-0.318); pass = False | data/patch/L_s0.json |
| headline | s0 | L | 400 random 3-sets, 29 unpicked heads | median 0.027, p95 0.732 (first 50: 0.024/0.733) | data/patch/L_s0.json |
| headline | s0 | L top-1 26.5 | sign-flipped fake attach (4 draws) | S = 1.1820 vs F = 0.4898; flagged heads 9 | data/picks_fp32.json |
| prereg clause 2 | s0 | L | 400 random 3-sets, all 32 heads | median 0.048 | data/patch/L_s0.json |
| blind vs oracle | s0 | L oracle 27.7/27.3/27.12 | behaviour-picked (dev) | R = 0.902; overlap 1/3 | data/patch/L_s0.json |
| answers | s0 | L blind top-3 cut | none | answer side flips with the probe on 128/128 test prompts; still flipped after the cut: 102 (per_prompt rows) | data/patch/L_s0.json |
| route | s0 | L all 32 heads | none | R = 1.000; plant top-3 recovers 0.429 | data/patch/L_s0.json |
| secondary score | s0 | L last-position top-3 27.7/27.3/27.12 | none | R = 0.902 | data/patch/L_s0.json |
| headline | s1 | L blind top-3 26.2/26.5/26.0 | none | R = 0.898 (95% CI 0.890-0.906); pass = False | data/patch/L_s1.json |
| headline | s1 | L | 400 random 3-sets, 29 unpicked heads | median 0.007, p95 0.920 (first 50: 0.008/0.918) | data/patch/L_s1.json |
| headline | s1 | L top-1 26.2 | sign-flipped fake attach (4 draws) | S = 1.9111 vs F = 0.4816; flagged heads 6 | data/picks_fp32.json |
| prereg clause 2 | s1 | L | 400 random 3-sets, all 32 heads | median 0.010 | data/patch/L_s1.json |
| blind vs oracle | s1 | L oracle 27.7/26.2/26.5 | behaviour-picked (dev) | R = 0.966; overlap 2/3 | data/patch/L_s1.json |
| answers | s1 | L blind top-3 cut | none | answer side flips with the probe on 128/128 test prompts; still flipped after the cut: 0 (per_prompt rows) | data/patch/L_s1.json |
| route | s1 | L all 32 heads | none | R = 0.999; plant top-3 recovers 0.949 | data/patch/L_s1.json |
| secondary score | s1 | L last-position top-3 27.7/26.2/26.0 | none | R = 0.961 | data/patch/L_s1.json |
| headline | s2 | L blind top-3 27.12/26.0/26.2 | none | R = 0.990 (95% CI 0.989-0.991); pass = True | data/patch/L_s2.json |
| headline | s2 | L | 400 random 3-sets, 29 unpicked heads | median 0.003, p95 0.014 (first 50: 0.002/0.015) | data/patch/L_s2.json |
| headline | s2 | L top-1 27.12 | sign-flipped fake attach (4 draws) | S = 3.2803 vs F = 0.6305; flagged heads 5 | data/picks_fp32.json |
| prereg clause 2 | s2 | L | 400 random 3-sets, all 32 heads | median 0.006 | data/patch/L_s2.json |
| blind vs oracle | s2 | L oracle 27.12/26.0/26.2 | behaviour-picked (dev) | R = 0.990; overlap 3/3 | data/patch/L_s2.json |
| answers | s2 | L blind top-3 cut | none | answer side flips with the probe on 128/128 test prompts; still flipped after the cut: 0 (per_prompt rows) | data/patch/L_s2.json |
| route | s2 | L all 32 heads | none | R = 1.000; plant top-3 recovers 0.996 | data/patch/L_s2.json |
| secondary score | s2 | L last-position top-3 27.12/27.6/27.9 | none | R = 0.975 | data/patch/L_s2.json |
| headline | s0 | S blind top-3 26.13/27.1/26.3 | none | R = 0.847 (95% CI 0.841-0.854); pass = True | data/patch/S_s0.json |
| headline | s0 | S | 400 random 3-sets, 29 unpicked heads | median 0.044, p95 0.216 (first 50: 0.064/0.207) | data/patch/S_s0.json |
| headline | s0 | S top-1 26.13 | sign-flipped fake attach (4 draws) | S = 0.8879 vs F = 0.4358; flagged heads 4 | data/picks_fp32.json |
| prereg clause 2 | s0 | S | 400 random 3-sets, all 32 heads | median 0.076 | data/patch/S_s0.json |
| blind vs oracle | s0 | S oracle 27.1/26.3/27.11 | behaviour-picked (dev) | R = 0.842; overlap 2/3 | data/patch/S_s0.json |
| answers | s0 | S blind top-3 cut | none | answer side flips with the probe on 91/128 test prompts; still flipped after the cut: 3 (per_prompt rows) | data/patch/S_s0.json |
| route | s0 | S all 32 heads | none | R = 1.000; plant top-3 recovers 0.823 | data/patch/S_s0.json |
| secondary score | s0 | S last-position top-3 27.1/26.13/27.0 | none | R = 0.784 | data/patch/S_s0.json |
| headline | s1 | S blind top-3 26.13/27.1/26.15 | none | R = 0.934 (95% CI 0.931-0.938); pass = True | data/patch/S_s1.json |
| headline | s1 | S | 400 random 3-sets, 29 unpicked heads | median 0.026, p95 0.192 (first 50: 0.024/0.194) | data/patch/S_s1.json |
| headline | s1 | S top-1 26.13 | sign-flipped fake attach (4 draws) | S = 1.0132 vs F = 0.4705; flagged heads 2 | data/picks_fp32.json |
| prereg clause 2 | s1 | S | 400 random 3-sets, all 32 heads | median 0.033 | data/patch/S_s1.json |
| blind vs oracle | s1 | S oracle 27.1/26.13/26.3 | behaviour-picked (dev) | R = 0.946; overlap 2/3 | data/patch/S_s1.json |
| answers | s1 | S blind top-3 cut | none | answer side flips with the probe on 91/128 test prompts; still flipped after the cut: 12 (per_prompt rows) | data/patch/S_s1.json |
| route | s1 | S all 32 heads | none | R = 1.000; plant top-3 recovers 0.937 | data/patch/S_s1.json |
| secondary score | s1 | S last-position top-3 27.1/26.13/27.12 | none | R = 0.923 | data/patch/S_s1.json |
| headline | s2 | S blind top-3 26.13/27.1/26.15 | none | R = 0.868 (95% CI 0.863-0.874); pass = True | data/patch/S_s2.json |
| headline | s2 | S | 400 random 3-sets, 29 unpicked heads | median 0.043, p95 0.161 (first 50: 0.039/0.162) | data/patch/S_s2.json |
| headline | s2 | S top-1 26.13 | sign-flipped fake attach (4 draws) | S = 0.7216 vs F = 0.4814; flagged heads 2 | data/picks_fp32.json |
| prereg clause 2 | s2 | S | 400 random 3-sets, all 32 heads | median 0.054 | data/patch/S_s2.json |
| blind vs oracle | s2 | S oracle 27.1/27.11/27.7 | behaviour-picked (dev) | R = 0.903; overlap 1/3 | data/patch/S_s2.json |
| answers | s2 | S blind top-3 cut | none | answer side flips with the probe on 100/128 test prompts; still flipped after the cut: 15 (per_prompt rows) | data/patch/S_s2.json |
| route | s2 | S all 32 heads | none | R = 1.000; plant top-3 recovers 0.861 | data/patch/S_s2.json |
| secondary score | s2 | S last-position top-3 27.12/27.1/26.13 | none | R = 0.840 | data/patch/S_s2.json |
| headline | s0 | RL blind top-3 27.8/27.9/26.14 | none | R = 0.998 (95% CI 0.994-1.001); pass = True | data/patch/RL_s0.json |
| headline | s0 | RL | 400 random 3-sets, 29 unpicked heads | median -0.000, p95 0.019 (first 50: -0.001/0.018) | data/patch/RL_s0.json |
| headline | s0 | RL top-1 27.8 | sign-flipped fake attach (4 draws) | S = 8.4193 vs F = 0.5042; flagged heads 2 | data/picks_fp32.json |
| prereg clause 2 | s0 | RL | 400 random 3-sets, all 32 heads | median 0.000 | data/patch/RL_s0.json |
| blind vs oracle | s0 | RL oracle 27.8/26.10/26.12 | behaviour-picked (dev) | R = 1.008; overlap 1/3 | data/patch/RL_s0.json |
| answers | s0 | RL blind top-3 cut | none | answer side flips with the probe on 172/200 test prompts; still flipped after the cut: 3 (per_prompt rows) | data/patch/RL_s0.json |
| route | s0 | RL all 32 heads | none | R = 1.000; plant top-3 recovers 0.998 | data/patch/RL_s0.json |
| secondary score | s0 | RL last-position top-3 27.8/27.9/26.13 | none | R = 0.998 | data/patch/RL_s0.json |
| headline | s1 | RL blind top-3 27.8/27.9/27.0 | none | R = 0.999 (95% CI 0.993-1.004); pass = True | data/patch/RL_s1.json |
| headline | s1 | RL | 400 random 3-sets, 29 unpicked heads | median -0.000, p95 0.002 (first 50: -0.000/0.002) | data/patch/RL_s1.json |
| headline | s1 | RL top-1 27.8 | sign-flipped fake attach (4 draws) | S = 24.2891 vs F = 1.0305; flagged heads 2 | data/picks_fp32.json |
| prereg clause 2 | s1 | RL | 400 random 3-sets, all 32 heads | median -0.000 | data/patch/RL_s1.json |
| blind vs oracle | s1 | RL oracle 27.8/26.5/27.11 | behaviour-picked (dev) | R = 1.034; overlap 1/3 | data/patch/RL_s1.json |
| answers | s1 | RL blind top-3 cut | none | answer side flips with the probe on 132/200 test prompts; still flipped after the cut: 2 (per_prompt rows) | data/patch/RL_s1.json |
| route | s1 | RL all 32 heads | none | R = 0.999; plant top-3 recovers 0.997 | data/patch/RL_s1.json |
| secondary score | s1 | RL last-position top-3 27.8/27.9/27.7 | none | R = 0.999 | data/patch/RL_s1.json |
| specificity | L_never | null model (no switch) | no-switch model | frozen 4-draw: flagged heads 1 (top-1 27.12 S 0.3379 vs F 0.3339); D3 99 draws: P(flag) 0.7756225733029857, perm p 0.07, verdict unclear; push -0.00244140625 (data/patch/L_never.json; S and F from data/picks_fp32.json) | data/nulldraws_fp32/L_never.json |
| specificity | RL_shuf | null model (no switch) | no-switch model | frozen 4-draw: flagged heads 0 (top-1 27.2 S 0.0043 vs F 0.0051); D3 99 draws: P(flag) 1.3282413871515492e-06, perm p 0.95, verdict no_alarm; push -0.013713756501674652 (data/patch/RL_shuf.json; S and F from data/picks_fp32.json) | data/nulldraws_fp32/RL_shuf.json |
| specificity | RL_tag | null model (no switch) | no-switch model | frozen 4-draw: flagged heads 1 (top-1 27.8 S 0.0941 vs F 0.0931); D3 99 draws: P(flag) 0.6194625616569652, perm p 0.12, verdict unclear; push -0.006225433945655823 (data/patch/RL_tag.json; S and F from data/picks_fp32.json) | data/nulldraws_fp32/RL_tag.json |
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
| emulation sanity | all | precision-emulation build | probe_emu vs none_emu | block-26 inputs bit-identical = True (tail sha256 in data/neutral_meta_Qwen3-1.7B.json and data/neutral_meta_Qwen3-0.6B.json); max head change 0.0 | data/picks_fp32.json |
