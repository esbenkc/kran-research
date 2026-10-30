# Vera (skeptic) sign-off — theory-1028-instructed-lie / lie-erasure-cost

**SIGN-OFF: UNCONFIRMED** (the numbers reproduce; the frozen rule gives "not settled").

Own code only (vera_price.py, vera_extract.py, vera_catch.py, vera_behaviour.py, vera_split.py, vera_math.py); no project
script imported. Rebuilt from saved raw data (items_spread.json, U_lens_pre/post.npy, fit_cities_160.json, R1 saved
states, R2 LoRA adapters).

| claim | project | Vera | file |
|---|---|---|---|
| P_blind(lie), answer-position KL | 0.979 | 0.931 (-4.8%, per-prompt r 0.979) | vera_price_compare.out |
| P_blind(decoy) | 0.00091 | 0.00095 | vera_price_compare.out |
| Q (premise >= 10) | 315 | 681 own; 300-716 over all numerator/null mixes; 487 vs worst own shuffle | vera_price_compare.out |
| every-position P(lie) full space (prediction clause 1, >= 0.06) | 0.0076 | 0.0073 -> clause 1 fails | vera_price_spread_indep_bf16_lw_tc.json |
| step-0 blind catch lie / decoy | 100 / 100% | 100 / 100%; shuffled null 0-2.5% | vera_catch_clean.out |
| R2 lie blind catch 150/200/250/300 | 100/100/100/100 | 100/100/100/100 (AUROC 1.000); shuffled null 0-1% | vera_r2.out |
| R2 B_lie (still a false country, greedy) | 0.7225 | 0.725 (one prompt parse) -> < 0.95 | vera_r2.out |
| R2 B_truthful | 0.9725 | 0.9725 | vera_r2.out |
| R2 lie prompts now naming the true country, still flagged | 106/106 (Kasper) | 110/110; other lie prompts 290/290; truthful 3/400 | vera_r2.out |
| R1 decoy blind catch 150/200/250/300 | 26.5/16.5/9.5/8.0 | 26.5/16.5/9.75/8.0 -> L 15.2% (moving, slope x150 -0.18) | vera_r1.out |
| R1 full panel L | 17.7% | 17.6% | vera_r1.out |
| R1 B_lie / B_truthful | 0.9825 / 1.000 | 0.985 / 1.000 | vera_r2.out |
| clean states, my extraction vs project | - | median rel err 0.0 at all 20 readouts | vera_r2.out |
| math: Pinsker bound, order-statistic shift, LEACE mean equalisation | - | 0 violations (20k / 20k / 500 cases) | vera_math.out |

Status by PREREG v1.3 s7: premise Q >= 10 holds; decoy fell; the lie cell is neither held nor fell because B_lie
0.725 < 0.95 -> **UNCONFIRMED**. Prediction outcome: **failed** (clause 1 by unit, clause 3 by behaviour; R3 not run).

Caveats the post must keep:
- The decoy "fall" is a max-of-20-panel tail shift: AUROC stays 0.91-0.96 and readout 10 alone still caught 80/68/48/42%.
- The lie concept is "system line + 8 false examples" (base model; the line alone barely moves the answer).
- n = 1 per cell; equal 300-step budget; the decoy curve was still falling at the cap.
- The probability-linear lemma probe catches 0% of lies even on the clean model (vera_lemma.json), so the lemma is a
  bound, not tonight's monitor; it was never violated per item.

Null tests: shuffled-label refits (3 seeds per state) 0-2.5%; shuffled-label and random-direction LEACE edits sit at
0.0006-0.0032 nats. Novelty: novelty.md (pre-run price as predictor of refit-probe evasion: no prior found).
