# Proof matrix — tell-chain-8b

Status (frozen rule, PREREG.md section 3): **CONFIRMED** (rule: CONFIRMED, gates ok: True). Prediction outcome: held. Built by make_matrix.py from data/analysis.json.

seed-level: 0.195 [0.165, 0.230] (R_D vs ctrl, geometric mean, t-interval on ln R over n 5 paired seeds, df n-1)
seed-level: 0.212 [0.176, 0.255] (R_D vs fresh, geometric mean, t-interval on ln R over n 5 paired seeds, df n-1)

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 5 | R_D = T(main_D) / T(ctrl_D) | coin-flip trunk (no switch) | T 64 / 328 -> R 0.195 | data/cells/RD_ctrl_s5.json |
| headline | 6 | R_D = T(main_D) / T(ctrl_D) | coin-flip trunk (no switch) | T 64 / 280 -> R 0.229 | data/cells/RD_ctrl_s6.json |
| headline | 7 | R_D = T(main_D) / T(ctrl_D) | coin-flip trunk (no switch) | T 48 / 304 -> R 0.158 | data/cells/RD_ctrl_s7.json |
| headline | 8 | R_D = T(main_D) / T(ctrl_D) | coin-flip trunk (no switch) | T 64 / 320 -> R 0.200 | data/cells/RD_ctrl_s8.json |
| headline | 9 | R_D = T(main_D) / T(ctrl_D) | coin-flip trunk (no switch) | T 56 / 280 -> R 0.200 | data/cells/RD_ctrl_s9.json |
| headline | all | R_D seed-level (n 5) | coin-flip trunk (no switch) | 0.195 [0.165, 0.230] | data/cells/RD_ctrl_seedlevel.json |
| headline | 5 | R_D = T(main_D) / T(fresh_D) | fresh tail | T 64 / 280 -> R 0.229 | data/cells/RD_fresh_s5.json |
| headline | 6 | R_D = T(main_D) / T(fresh_D) | fresh tail | T 64 / 256 -> R 0.250 | data/cells/RD_fresh_s6.json |
| headline | 7 | R_D = T(main_D) / T(fresh_D) | fresh tail | T 48 / 288 -> R 0.167 | data/cells/RD_fresh_s7.json |
| headline | 8 | R_D = T(main_D) / T(fresh_D) | fresh tail | T 64 / 304 -> R 0.211 | data/cells/RD_fresh_s8.json |
| headline | 9 | R_D = T(main_D) / T(fresh_D) | fresh tail | T 56 / 264 -> R 0.212 | data/cells/RD_fresh_s9.json |
| headline | all | R_D seed-level (n 5) | fresh tail | 0.212 [0.176, 0.255] | data/cells/RD_fresh_seedlevel.json |
| stage C (S1) | 5 | R_C = T(main_C) / T(ctrl_C) | coin-flip trunk (no switch) | T 96 / 248 -> R 0.387 | data/cells/RC_ctrl_s5.json |
| stage C (S1) | 6 | R_C = T(main_C) / T(ctrl_C) | coin-flip trunk (no switch) | T 80 / 232 -> R 0.345 | data/cells/RC_ctrl_s6.json |
| stage C (S1) | 7 | R_C = T(main_C) / T(ctrl_C) | coin-flip trunk (no switch) | T 80 / 264 -> R 0.303 | data/cells/RC_ctrl_s7.json |
| stage C (S1) | 8 | R_C = T(main_C) / T(ctrl_C) | coin-flip trunk (no switch) | T 80 / 320 -> R 0.250 | data/cells/RC_ctrl_s8.json |
| stage C (S1) | 9 | R_C = T(main_C) / T(ctrl_C) | coin-flip trunk (no switch) | T 104 / 304 -> R 0.342 | data/cells/RC_ctrl_s9.json |
| stage C (S1) | all | R_C seed-level (n 5) | coin-flip trunk (no switch) | 0.322 [0.262, 0.396] | data/cells/RC_ctrl_seedlevel.json |
| stage C (S1) | 5 | R_C = T(main_C) / T(fresh_C) | fresh tail | T 96 / 272 -> R 0.353 | data/cells/RC_fresh_s5.json |
| stage C (S1) | 6 | R_C = T(main_C) / T(fresh_C) | fresh tail | T 80 / 256 -> R 0.312 | data/cells/RC_fresh_s6.json |
| stage C (S1) | 7 | R_C = T(main_C) / T(fresh_C) | fresh tail | T 80 / 296 -> R 0.270 | data/cells/RC_fresh_s7.json |
| stage C (S1) | 8 | R_C = T(main_C) / T(fresh_C) | fresh tail | T 80 / 328 -> R 0.244 | data/cells/RC_fresh_s8.json |
| stage C (S1) | 9 | R_C = T(main_C) / T(fresh_C) | fresh tail | T 104 / 320 -> R 0.325 | data/cells/RC_fresh_s9.json |
| stage C (S1) | all | R_C seed-level (n 5) | fresh tail | 0.298 [0.248, 0.359] | data/cells/RC_fresh_seedlevel.json |
| stage B (parent) | 5 | R_B from second-tell-8b | parent COIN | R 0.857 | data/cells/RB_ctrl_s5.json |
| stage B (parent) | 6 | R_B from second-tell-8b | parent COIN | R 0.714 | data/cells/RB_ctrl_s6.json |
| stage B (parent) | 7 | R_B from second-tell-8b | parent COIN | R 0.824 | data/cells/RB_ctrl_s7.json |
| stage B (parent) | 8 | R_B from second-tell-8b | parent COIN | R 0.706 | data/cells/RB_ctrl_s8.json |
| stage B (parent) | 9 | R_B from second-tell-8b | parent COIN | R 0.750 | data/cells/RB_ctrl_s9.json |
| stage B (parent) | 5 | R_B from second-tell-8b | parent FRESH | R 0.732 | data/cells/RB_fresh_s5.json |
| stage B (parent) | 6 | R_B from second-tell-8b | parent FRESH | R 0.694 | data/cells/RB_fresh_s6.json |
| stage B (parent) | 7 | R_B from second-tell-8b | parent FRESH | R 0.757 | data/cells/RB_fresh_s7.json |
| stage B (parent) | 8 | R_B from second-tell-8b | parent FRESH | R 0.706 | data/cells/RB_fresh_s8.json |
| stage B (parent) | 9 | R_B from second-tell-8b | parent FRESH | R 0.771 | data/cells/RB_fresh_s9.json |
| switch times | 5 | main_C | none | T 96 [ok] | data/cells/T_main_C_s5.json |
| switch times | 6 | main_C | none | T 80 [ok] | data/cells/T_main_C_s6.json |
| switch times | 7 | main_C | none | T 80 [ok] | data/cells/T_main_C_s7.json |
| switch times | 8 | main_C | none | T 80 [ok] | data/cells/T_main_C_s8.json |
| switch times | 9 | main_C | none | T 104 [ok] | data/cells/T_main_C_s9.json |
| switch times | 5 | ctrl_C | coin-flip trunk | T 248 [ok] | data/cells/T_ctrl_C_s5.json |
| switch times | 6 | ctrl_C | coin-flip trunk | T 232 [ok] | data/cells/T_ctrl_C_s6.json |
| switch times | 7 | ctrl_C | coin-flip trunk | T 264 [ok] | data/cells/T_ctrl_C_s7.json |
| switch times | 8 | ctrl_C | coin-flip trunk | T 320 [ok] | data/cells/T_ctrl_C_s8.json |
| switch times | 9 | ctrl_C | coin-flip trunk | T 304 [ok] | data/cells/T_ctrl_C_s9.json |
| switch times | 5 | fresh_C | fresh tail | T 272 [ok] | data/cells/T_fresh_C_s5.json |
| switch times | 6 | fresh_C | fresh tail | T 256 [ok] | data/cells/T_fresh_C_s6.json |
| switch times | 7 | fresh_C | fresh tail | T 296 [ok] | data/cells/T_fresh_C_s7.json |
| switch times | 8 | fresh_C | fresh tail | T 328 [ok] | data/cells/T_fresh_C_s8.json |
| switch times | 9 | fresh_C | fresh tail | T 320 [ok] | data/cells/T_fresh_C_s9.json |
| switch times | 5 | dose1_C | one prior switch | T 96 [ok] | data/cells/T_dose1_C_s5.json |
| switch times | 6 | dose1_C | one prior switch | T 96 [ok] | data/cells/T_dose1_C_s6.json |
| switch times | 7 | dose1_C | one prior switch | T 112 [ok] | data/cells/T_dose1_C_s7.json |
| switch times | 8 | dose1_C | one prior switch | T 88 [ok] | data/cells/T_dose1_C_s8.json |
| switch times | 9 | dose1_C | one prior switch | T 88 [ok] | data/cells/T_dose1_C_s9.json |
| switch times | 5 | main_D | none | T 64 [ok] | data/cells/T_main_D_s5.json |
| switch times | 6 | main_D | none | T 64 [ok] | data/cells/T_main_D_s6.json |
| switch times | 7 | main_D | none | T 48 [ok] | data/cells/T_main_D_s7.json |
| switch times | 8 | main_D | none | T 64 [ok] | data/cells/T_main_D_s8.json |
| switch times | 9 | main_D | none | T 56 [ok] | data/cells/T_main_D_s9.json |
| switch times | 5 | ctrl_D | coin-flip trunk | T 328 [ok] | data/cells/T_ctrl_D_s5.json |
| switch times | 6 | ctrl_D | coin-flip trunk | T 280 [ok] | data/cells/T_ctrl_D_s6.json |
| switch times | 7 | ctrl_D | coin-flip trunk | T 304 [ok] | data/cells/T_ctrl_D_s7.json |
| switch times | 8 | ctrl_D | coin-flip trunk | T 320 [ok] | data/cells/T_ctrl_D_s8.json |
| switch times | 9 | ctrl_D | coin-flip trunk | T 280 [ok] | data/cells/T_ctrl_D_s9.json |
| switch times | 5 | fresh_D | fresh tail | T 280 [ok] | data/cells/T_fresh_D_s5.json |
| switch times | 6 | fresh_D | fresh tail | T 256 [ok] | data/cells/T_fresh_D_s6.json |
| switch times | 7 | fresh_D | fresh tail | T 288 [ok] | data/cells/T_fresh_D_s7.json |
| switch times | 8 | fresh_D | fresh tail | T 304 [ok] | data/cells/T_fresh_D_s8.json |
| switch times | 9 | fresh_D | fresh tail | T 264 [ok] | data/cells/T_fresh_D_s9.json |
| dose at C (S2) | all | main_vs_dose1 | one prior switch | 0.914 [0.723, 1.157] | data/cells/dose_main_vs_dose1.json |
| dose at C (S2) | all | dose1_vs_ctrl | one prior switch | 0.352 [0.273, 0.455] | data/cells/dose_dose1_vs_ctrl.json |
| null cue E (S4) | all | max D_E over MAIN records | never-trained cue E | MAIN max D_E 0.00383, max abs D_E 0.01428; trunk max 0.0044, max abs D_E 0.00581 | data/cells/null_cue_E.json |
| gates | - | G1 | untrained tail | pass True | data/cells/gate_G1.json |
| gates | - | G2 | none | pass True | data/cells/gate_G2.json |
| gates | - | G3 | none | pass True | data/cells/gate_G3.json |
| gates | - | PORT_D | none | pass True | data/cells/gate_PORT_D.json |
| gates | - | PORT_C_report | none | pass True | data/cells/gate_PORT_C_report.json |
| rebuild (Vera, skeptic) | - | RD_ctrl_s5 | none | 0.195122 vs ours 0.195122: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_ctrl_s6 | none | 0.228571 vs ours 0.228571: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_ctrl_s7 | none | 0.157895 vs ours 0.157895: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_ctrl_s8 | none | 0.2 vs ours 0.2: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_ctrl_s9 | none | 0.2 vs ours 0.2: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_ctrl_seedlevel | none | 0.194963 vs ours 0.194963: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_ctrl_s5 | none | 0.3871 vs ours 0.387097: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_ctrl_s6 | none | 0.3448 vs ours 0.344828: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_ctrl_s7 | none | 0.303 vs ours 0.30303: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_ctrl_s8 | none | 0.25 vs ours 0.25: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_ctrl_s9 | none | 0.3421 vs ours 0.342105: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_ctrl_seedlevel | none | 0.32196 vs ours 0.32196: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_fresh_s5 | none | 0.228571 vs ours 0.228571: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_fresh_s6 | none | 0.25 vs ours 0.25: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_fresh_s7 | none | 0.166667 vs ours 0.166667: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_fresh_s8 | none | 0.210526 vs ours 0.210526: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_fresh_s9 | none | 0.212121 vs ours 0.212121: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RD_fresh_seedlevel | none | 0.211709 vs ours 0.211709: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_fresh_s5 | none | 0.3529 vs ours 0.352941: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_fresh_s6 | none | 0.3125 vs ours 0.3125: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_fresh_s7 | none | 0.2703 vs ours 0.27027: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_fresh_s8 | none | 0.2439 vs ours 0.243902: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_fresh_s9 | none | 0.325 vs ours 0.325: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | RC_fresh_seedlevel | none | 0.298325 vs ours 0.298325: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_C_s5 | none | 248 vs ours 248: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_C_s6 | none | 232 vs ours 232: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_C_s7 | none | 264 vs ours 264: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_C_s8 | none | 320 vs ours 320: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_C_s9 | none | 304 vs ours 304: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_D_s5 | none | 328 vs ours 328: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_D_s6 | none | 280 vs ours 280: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_D_s7 | none | 304 vs ours 304: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_D_s8 | none | 320 vs ours 320: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_ctrl_D_s9 | none | 280 vs ours 280: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_dose1_C_s5 | none | 96 vs ours 96: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_dose1_C_s6 | none | 96 vs ours 96: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_dose1_C_s7 | none | 112 vs ours 112: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_dose1_C_s8 | none | 88 vs ours 88: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_dose1_C_s9 | none | 88 vs ours 88: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_C_s5 | none | 272 vs ours 272: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_C_s6 | none | 256 vs ours 256: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_C_s7 | none | 296 vs ours 296: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_C_s8 | none | 328 vs ours 328: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_C_s9 | none | 320 vs ours 320: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_D_s5 | none | 280 vs ours 280: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_D_s6 | none | 256 vs ours 256: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_D_s7 | none | 288 vs ours 288: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_D_s8 | none | 304 vs ours 304: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_fresh_D_s9 | none | 264 vs ours 264: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_C_s5 | none | 96 vs ours 96: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_C_s6 | none | 80 vs ours 80: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_C_s7 | none | 80 vs ours 80: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_C_s8 | none | 80 vs ours 80: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_C_s9 | none | 104 vs ours 104: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_D_s5 | none | 64 vs ours 64: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_D_s6 | none | 64 vs ours 64: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_D_s7 | none | 48 vs ours 48: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_D_s8 | none | 64 vs ours 64: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | T_main_D_s9 | none | 56 vs ours 56: MATCH | skeptics/skeptic/vera_chain.json |
| rebuild (Vera, skeptic) | - | status + prediction | none | CONFIRMED / held vs ours CONFIRMED / held: MATCH | skeptics/skeptic/vera_chain.json |
