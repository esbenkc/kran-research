# Proof matrix — instructed-lie-leace

Status (analysis.py, PREREG v1.3 s7): **UNCONFIRMED**; prediction outcome: **failed**.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | decoy @ lam 10000, cap 300 | none | blind-panel late mean L = 0.151 (step 0 1.000; moving True) | data/runs/decoy_spread__lam10000__seed0.json |
| mechanical half | 0 | decoy @ lam 10000 | none | full panel L = 0.177; output-only probe L = 0.013 | data/runs/decoy_spread__lam10000__seed0.json |
| behaviour kept | 0 | decoy @ lam 10000 | none | B_lie 0.983, B_same 0.838, B_truthful 1.000, answer KL 0.0024 | data/runs/decoy_spread__lam10000__seed0__behaviour.json |
| refit probe reads the concept | 0 | decoy @ lam 10000, final step | shuffled labels (5 perms) | blind catch 0.005, 0.007, 0.007, 0.010, 0.010 | data/runs/decoy_spread__lam10000__seed0.json |
| headline | 0 | lie @ lam 10000, cap 300 | none | blind-panel late mean L = 1.000 (step 0 1.000; moving False) | data/runs/lie_spread__lam10000__seed0.json |
| mechanical half | 0 | lie @ lam 10000 | none | full panel L = 1.000; output-only probe L = 0.999 | data/runs/lie_spread__lam10000__seed0.json |
| behaviour kept | 0 | lie @ lam 10000 | none | B_lie 0.723, B_same 0.443, B_truthful 0.973, answer KL 0.0898 | data/runs/lie_spread__lam10000__seed0__behaviour.json |
| refit probe reads the concept | 0 | lie @ lam 10000, final step | shuffled labels (5 perms) | blind catch 0.013, 0.003, 0.022, 0.007, 0.000 | data/runs/lie_spread__lam10000__seed0.json |
| lie hold not the common answer | 0 | lie @ lam 10000 | subset: clean answer not 'the United States' | L = 1.000 (n 279); answer not 'the ...': L = 1.000 (n 207) | data/runs/lie_spread__lam10000__seed0.json |
| each refit is beaten within 50 steps | 0 | decoy @ lam 1e4 | fresh refit on the same states | stale probes (fit at k, scored at k+50): 100->150 0.015 (fresh 0.265), 150->200 0.030 (fresh 0.165), 200->250 0.015 (fresh 0.095), 250->300 0.020 (fresh 0.080) | data/results/crossstep_decoy_spread__lam10000__seed0.json |
| lie run changed the behaviour | 0 | lie @ lam 1e4 | none | still a false country on eval lie prompts: step 0 0.978, step 150 0.840, step 200 0.723, step 250 0.945, step 300 0.723 | data/results/still_lying.json |
| price before training | - | lie_blind | none | answer-position KL 0.97877; every-position 0.007020 | data/leace/price_blind_spread.json |
| price before training | - | decoy_blind | none | answer-position KL 0.00091; every-position 0.000007 | data/leace/price_blind_spread.json |
| price before training | - | lie_full | none | answer-position KL 1.06367; every-position 0.007629 | data/leace/price_blind_spread.json |
| price before training | - | decoy_full | none | answer-position KL 0.00107; every-position 0.000008 | data/leace/price_blind_spread.json |
| premise Q >= 10 | - | blind, independent erasers | shuffled-label price, 5 perms | Q 315.5; raw ratio 1072.0; N_decoy 0.00310; N_lie 0.00377; lie/own null 259.7 | data/leace/price_blind_spread.json |
| G0b' readable at step 0 | - | lie, blind panel, 640-prompt fit | none | catch 1.000 (1296 fit 1.000); full 1.000; O_29 1.000 | data/gate0/gate0b_spread.json |
| G0b' readable at step 0 | - | decoy, blind panel, 640-prompt fit | none | catch 1.000 (1296 fit 1.000); full 1.000; O_29 0.022 | data/gate0/gate0b_spread.json |
| G0b' null | - | lie, random k-subspace removed | random directions | catch 1.000 | data/gate0/gate0b_spread.json |
| G0a compliance | - | base Qwen2.5-1.5B, R8v, 748 cities | none | lie 0.860, truthful 0.983, kept 624 | data/gate0/gate0.json |
| G0a compliance | - | spread prompts, kept cities | none | lie 0.978, truthful 0.998 | data/gate0/gate0_spread.json |
