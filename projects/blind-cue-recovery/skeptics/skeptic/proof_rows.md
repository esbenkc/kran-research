| skeptic rebuild: headline | 0-5 | SM 4-twin mean at t_h (16,16,16,16,8,16), own scorer | none | ranks 438/454/735/709/908/240; 0/6 beat (a,b,c) | skeptics/skeptic/vera_check_sm.json |
| skeptic rebuild: headline | 0-5 | SM replicates at t_h: 4 single twins + 2 halves per model | none | best rank per model 426/342/736/704/853/220; 0/36 | skeptics/skeptic/vera_extra.json |
| skeptic rebuild: headline | 0-5 | SM at t_h, null seed 7 instead of 20261112 | different null seed | ranks 439/430/770/727/907/232 | skeptics/skeptic/vera_null7_sm.txt |
| skeptic rebuild: phase 0 | 0-5 | RAW at 256 | none | 301/1/705/1/976/256 (2/6 rank 1) | skeptics/skeptic/vera_raw.json |
| skeptic rebuild: arms | 0-5 | XS, SS at t_h and T; XS at 256 | other-seed / same-seed clean adapters | 0/6, 0/6; XS 256 2/6; abs cos(v1 RAW, v1 XS) 0.999-1.000 at T | skeptics/skeptic/vera_check_xs.json |
| skeptic rebuild: arms | 0-5 | SMv, SM0 at t_h and T | merged-weights twins / own-A0 twins | 0/6 in all four reads | skeptics/skeptic/vera_check_smv.json |
| skeptic rebuild: arms | 0-5 | SM0 at t_h and T | own-A0 twins | t_h 455/410/707/693/901/271; T 407/502/670/726/938/278 | skeptics/skeptic/vera_check_sm0.json |
| skeptic rebuild: E1 | 0-5 | E1-SS, E1-SM, E1-RAW at t_h and T | none | E1-RAW T 1/1/1/1/3/36 (4/6), t_h 2/6; E1-SM 4/6, 1/6 | skeptics/skeptic/vera_check_e1.json |
| skeptic rebuild: E2 | 0 | sweep x 4.20-0.217 at T/2 | none | RAW 2/5, E1-RAW 1/5 | skeptics/skeptic/vera_e2.json |
| skeptic rebuild: lemma | 0-9 | Lemma 1 overlap; Lemma 2 bound vs measured | Gaussian Monte Carlo | 0.04691 vs 0.046875; bound held 18/18 | skeptics/skeptic/vera_lemma.json |
| skeptic rebuild: behaviour | 0-5 | D at t_h with own bootstrap over 64 prompts | none | 3.4-5.4 points, all CIs below 10 | skeptics/skeptic/vera_beh.json |
