# Proof matrix (built by proof_matrix.py from data/results.json)

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | greedy (n=1) | TEST 200: keep only the push into M (r_keep) | none | 0.048 [95% CI -0.033, 0.136] | data/runs/test_harmful/C.jsonl |
| headline | greedy (n=1) | TEST 200: remove only the push into M (r_rem) | none | 1.016 [95% CI 0.946, 1.096] | data/runs/test_harmful/B.jsonl |
| headline | greedy (n=1) | TEST 200: arm A effect on coherent compliance | none | +31.5 pts [+23.5 pts, +40.0 pts] | data/runs/test_harmful/A.jsonl |
| split | greedy (n=1) | TEST 200: arm B effect on coherent compliance | none | +32.0 pts [+23.5 pts, +40.5 pts] | data/runs/test_harmful/B.jsonl |
| split | greedy (n=1) | TEST 200: arm C effect on coherent compliance | none | +1.5 pts [-1.0 pts, +4.0 pts] | data/runs/test_harmful/C.jsonl |
| headline | greedy (n=1) | DEV 100: keep only the push into M (r_keep) | none | 0.037 [95% CI 0.000, 0.096] | data/runs/dev_harmful/C.jsonl |
| headline | greedy (n=1) | DEV 100: remove only the push into M (r_rem) | none | 1.019 [95% CI 1.000, 1.064] | data/runs/dev_harmful/B.jsonl |
| headline | greedy (n=1) | DEV 100: arm A effect on coherent compliance | none | +54.0 pts [+44.0 pts, +64.0 pts] | data/runs/dev_harmful/A.jsonl |
| split | greedy (n=1) | DEV 100: arm B effect on coherent compliance | none | +55.0 pts [+45.0 pts, +64.0 pts] | data/runs/dev_harmful/B.jsonl |
| split | greedy (n=1) | DEV 100: arm C effect on coherent compliance | none | +2.0 pts [+0.0 pts, +5.0 pts] | data/runs/dev_harmful/C.jsonl |
| status | - | STATUS rule (string label) | none | REJECTED | data/results.json |
| phase0 | greedy (n=1) | TEST refusal drop of the full pin | base | 43.5 pts (base 46.0% -> 2.5%) | data/runs/test_harmful/A.jsonl |
| control | greedy (n=1) | clean model, TEST harmful | base (no edit) | refusal 46.0%, coherent compliance 48.0% | data/runs/test_harmful/base.jsonl |
| phase0 | rng 0 | first-token refusal score drop: gate vs 100 random L13 neurons (matched edit norm, better sign) | random neurons (F) | gate 9.77 vs F p99 7.16 (max 8.51) | data/runs/logit/test_harmful/A.json |
| coherence | greedy (n=1) | HARMLESS 200, arm A | harmless prompts | garbage 6.5%, refusal 0.0%, pass=True | data/runs/test_harmless/A.jsonl |
| coherence | greedy (n=1) | HARMLESS 200, arm B | harmless prompts | garbage 5.5%, refusal 0.0%, pass=True | data/runs/test_harmless/B.jsonl |
| coherence | greedy (n=1) | HARMLESS 200, arm C | harmless prompts | garbage 5.0%, refusal 0.0%, pass=True | data/runs/test_harmless/C.jsonl |
| coherence | greedy (n=1) | HARMLESS 200, arm A_hm | harmless prompts | garbage 5.0%, refusal 0.0%, pass=True | data/runs/test_harmless/A_hm.jsonl |
| coherence | greedy (n=1) | HARMLESS 200, arm A_zero | harmless prompts | garbage 5.0%, refusal 0.0%, pass=True | data/runs/test_harmless/A_zero.jsonl |
| coherence | greedy (n=1) | HARMLESS 200, arm A_skip0 | harmless prompts | garbage 6.5%, refusal 0.0%, pass=True | data/runs/test_harmless/A_skip0.jsonl |
| coherence | greedy (n=1) | HARMLESS 200, arm A_tok0 | harmless prompts | garbage 5.5%, refusal 0.0%, pass=True | data/runs/test_harmless/A_tok0.jsonl |
| position/existence | greedy (n=1) | TEST 200: pin to harmless mean | none | +5.0 pts [+0.5 pts, +10.0 pts] | data/runs/test_harmful/A_hm.jsonl |
| position/existence | greedy (n=1) | TEST 200: pin to zero | none | +8.0 pts [+3.0 pts, +13.0 pts] | data/runs/test_harmful/A_zero.jsonl |
| position/existence | greedy (n=1) | TEST 200: full pin except first token | none | +32.5 pts [+24.5 pts, +40.5 pts] | data/runs/test_harmful/A_skip0.jsonl |
| position/existence | greedy (n=1) | TEST 200: full pin at first token only | none | +1.0 pts [-2.0 pts, +4.0 pts] | data/runs/test_harmful/A_tok0.jsonl |
| position/existence | greedy (n=1) | TEST 200: neuron 5441 (+), string label | none | +12.5 pts [+5.5 pts, +19.5 pts] | data/runs/test_harmful/N5441pos.jsonl |
| position/existence | greedy (n=1) | TEST 200: neuron 5441, its push on the 3 outlier coordinates only | none | -3.0 pts [-9.5 pts, +3.5 pts] | data/runs/test_harmful/N5441_M.jsonl |
| position/existence | greedy (n=1) | TEST 200: neuron 5441, its push with the 3 outlier coordinates zeroed | none | +9.0 pts [+2.5 pts, +15.5 pts] | data/runs/test_harmful/N5441_R.jsonl |
| existence | rng 0, 10 draws | TEST 200: random pattern pushed into M at the gate's full norm (D') | random draws | median +0.2 pts, max +10.5 pts | data/runs/test_harmful/Dp_0.jsonl |
| control | rng 0, 5 draws | TEST 200: same push into 3 ordinary coordinates (E') | random draws | median -2.5 pts, max +6.5 pts | data/runs/test_harmful/Ep_0.jsonl |
| control | rng 0, 3 draws | TEST 200: random L13 neurons, matched edit norm (F, +) | random draws | median +21.0 pts, max +21.5 pts | data/runs/test_harmful/Fpos_0.jsonl |
| control | rng 0, 2 draws | TEST 200: random L13 neurons, matched edit norm (F, -) | random draws | median +6.0 pts, max +8.5 pts | data/runs/test_harmful/Fneg_2.jsonl |
| control | rng 0, 5 draws | TEST 200: same push along a random direction (F') | random draws | median +12.5 pts, max +16.5 pts | data/runs/test_harmful/Fp_0.jsonl |
| existence (first token) | rng 0, 100 draws | first-token score drop, family Dp | random draws | median 0.36, max 2.10, share >= half the gate's 0.00 | data/runs/logit/test_harmful/Dp_0.json |
| existence (first token) | rng 0, 100 draws | first-token score drop, family Ep | random draws | median 0.68, max 3.20, share >= half the gate's 0.00 | data/runs/logit/test_harmful/Ep_0.json |
| existence (first token) | rng 0, 20 draws | first-token score drop, family Fp | random draws | median 0.93, max 3.12, share >= half the gate's 0.00 | data/runs/logit/test_harmful/Fp_0.json |
| existence (first token) | rng 0, 100 draws | first-token score drop, family Ds | random draws | median 0.03, max 0.10, share >= half the gate's 0.00 | data/runs/logit/test_harmful/Ds_0.json |
| first-token | greedy (n=1) | first-token score drop, arm A | base | 9.77 (gate A 9.77) | data/runs/logit/test_harmful/A.json |
| first-token | greedy (n=1) | first-token score drop, arm C | base | -0.06 (gate A 9.77) | data/runs/logit/test_harmful/C.json |
| first-token | greedy (n=1) | first-token score drop, arm B | base | 9.80 (gate A 9.77) | data/runs/logit/test_harmful/B.json |
| first-token | greedy (n=1) | first-token score drop, arm N5441pos | base | 1.23 (gate A 9.77) | data/runs/logit/test_harmful/N5441pos.json |
| first-token | greedy (n=1) | first-token score drop, arm N5441neg | base | 0.63 (gate A 9.77) | data/runs/logit/test_harmful/N5441neg.json |
| first-token | greedy (n=1) | first-token score drop, arm N5441_M | base | -0.42 (gate A 9.77) | data/runs/logit/test_harmful/N5441_M.json |
| first-token | greedy (n=1) | first-token score drop, arm N5441_R | base | 1.34 (gate A 9.77) | data/runs/logit/test_harmful/N5441_R.json |
| method (D9) | - | tail-from-L13 scorer vs full forward, max per-prompt absolute score diff | full forward | base 0.51, A 0.68, B 0.74, C 0.74 (arm means differ by <= 0.02) | data/runs/logit/d9_check.json |
| headline (skeptic rebuild) | greedy (n=1) | TEST 200 string: r_keep / r_rem, own code from raw jsonl | none | 0.048 [-0.033,0.136] / 1.016 [0.946,1.096] -> REJECTED | skeptics/skeptic/vera_check_test_final.txt |
| headline (skeptic rebuild) | greedy (n=1) | TEST 200 judge (D3): r_keep / r_rem | none | 0.000 [-0.034,0.034] / 1.023 [0.935,1.122] -> REJECTED | skeptics/skeptic/vera_check_test_final.txt |
| headline (skeptic rebuild) | greedy (n=1) | DEV 100 string / judge: r_keep / r_rem | none | 0.037 / 1.019 (string); 0.000 / 0.978 (judge) -> REJECTED | skeptics/skeptic/vera_check_dev_harmful.json |
| skeptic: own forward | fp32 CPU, own Qwen3 code | first-token score per prompt vs build, TEST 0-49, base/A/B/C/A_tok0/N5441 | build's bf16 scores | corr 0.9991-0.9997; A drop 9.588 vs 9.640 | skeptics/skeptic/own/ |
| skeptic: own existence draws | torch seeds 1-5 | random direction inside M at the gate's full push, first token, TEST 0-49 | 3 ordinary-coord seeds | -0.15..+0.16 of A (ordinary: -0.15..+0.09) | skeptics/skeptic/own/ |
| skeptic: judge on random gen | 25 draws | D'/E'/F/F' generation, judge EFFECT max | random draws | D' +4.0, E' +4.5, F +7.0, F' +4.5 pts vs gate +43.5 | skeptics/skeptic/vera_check_test_final.txt |
| skeptic: definition check | greedy (n=1) | N5441 string vs judge | judge label | +12.5 vs -1.0 pts (37/40 string flips are reworded refusals) | skeptics/skeptic/vera_judge_test.json |
