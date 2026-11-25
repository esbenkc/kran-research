# Proof matrix — probe-switch-head-budget

Written by status.py from data/ (status UNCONFIRMED; prereg outcome failed; rule: PREREG.md FROZEN RULE, pushed fdc17f064). S(k) = after-cut strict switch on 128 test prompts (fp32 rows; bf16 next to it), blind top-k picked by the joint greedy audit with the per-prompt temperature on 64 dev prompts. R8 = median S over 50 random 8-head cuts (seed 1000 + i). k_min = smallest k with S(k) < 0.25.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | attacker vs 6-head blind audit (per-prompt T) | none | S0 1.000; S(8) 0.008 [0.001, 0.043] (bf16 0.008); k_min 6 (bf16 6); live True, monitor ok True, G1 ok True | data/kaudit/k6_s0.json |
| headline | 0 | 50 random 8-head cuts | random heads | R8 0.930 (min 0.172; bf16 0.930) | data/kaudit/k6_s0.json |
| headline | 1 | attacker vs 6-head blind audit (per-prompt T) | none | S0 0.000; S(8) 0.000 [0.000, 0.029] (bf16 0.000); k_min 1 (bf16 1); live False, monitor ok False, G1 ok True | data/kaudit/k6_s1.json |
| headline | 1 | 50 random 8-head cuts | random heads | R8 0.000 (min 0.000; bf16 0.000) | data/kaudit/k6_s1.json |
| headline | 2 | attacker vs 6-head blind audit (per-prompt T) | none | S0 0.008; S(8) 0.000 [0.000, 0.029] (bf16 0.000); k_min 1 (bf16 1); live False, monitor ok False, G1 ok True | data/kaudit/k6_s2.json |
| headline | 2 | 50 random 8-head cuts | random heads | R8 0.008 (min 0.000; bf16 0.000) | data/kaudit/k6_s2.json |
| headline | 3 | attacker vs 6-head blind audit (per-prompt T) [spare] | none | S0 1.000; S(8) 0.023 [0.008, 0.067] (bf16 0.023); k_min 7 (bf16 6); live True, monitor ok True, G1 ok True | data/kaudit/k6_s3.json |
| headline | 3 | 50 random 8-head cuts | random heads | R8 0.996 (min 0.391; bf16 0.996) | data/kaudit/k6_s3.json |
| law k_att=3 | 0 | attacker vs 3-head blind audit (per-prompt T) | none | S0 0.922; S(8) 0.000 [0.000, 0.029] (bf16 0.000); k_min 2 (bf16 2); live True, monitor ok True, G1 ok False | data/kaudit/k3_s0.json |
| law k_att=3 | 1 | attacker vs 3-head blind audit (per-prompt T) | none | S0 1.000; S(8) 0.008 [0.001, 0.043] (bf16 0.008); k_min 4 (bf16 4); live True, monitor ok True, G1 ok True | data/kaudit/k3_s1.json |
| law k_att=3 | 2 | attacker vs 3-head blind audit (per-prompt T) | none | S0 0.914; S(8) 0.000 [0.000, 0.029] (bf16 0.000); k_min 1 (bf16 1); live True, monitor ok True, G1 ok True | data/kaudit/k3_s2.json |
| law k_att=1 | 0 | attacker vs 1-head blind audit (per-prompt T) | none | S0 0.016; S(8) 0.000 [0.000, 0.029] (bf16 0.000); k_min 1 (bf16 1); live False, monitor ok True, G1 ok True | data/kaudit/k1_s0.json |
| law k_att=1 | 1 | attacker vs 1-head blind audit (per-prompt T) | none | S0 0.977; S(8) 0.000 [0.000, 0.029] (bf16 0.000); k_min 4 (bf16 4); live True, monitor ok True, G1 ok True | data/kaudit/k1_s1.json |
| law k_att=1 | 2 | attacker vs 1-head blind audit (per-prompt T) | none | S0 0.000; S(8) 0.000 [0.000, 0.029] (bf16 0.000); k_min 1 (bf16 1); live False, monitor ok True, G1 ok True | data/kaudit/k1_s2.json |
| law k_att=1 | 3 | attacker vs 1-head blind audit (per-prompt T) [spare] | none | S0 0.172; S(8) 0.172 [0.116, 0.247] (bf16 0.172); k_min 1 (bf16 1); live False, monitor ok True, G1 ok False | data/kaudit/k1_s3.json |
| control | 0 | no penalty, same training | no-penalty attacker | S0 1.000; S(8) 0.000; k_min 1; R8 0.887 | data/kaudit/l0_s0.json |
| control | 1 | no penalty, same training | no-penalty attacker | S0 1.000; S(8) 0.000; k_min 1; R8 0.996 | data/kaudit/l0_s1.json |
| control | - | warm adapter s0, untrained | untrained switch | S0 1.000; S(8) 0.000; k_min 1; R8 0.910 | data/kaudit/warm_s0.json |
| control | - | warm adapter s1, untrained | untrained switch | S0 1.000; S(8) 0.000; k_min 1; R8 0.938 | data/kaudit/warm_s1.json |
| control | - | warm adapter s2, untrained | untrained switch | S0 1.000; S(8) 0.000; k_min 1; R8 0.867 | data/kaudit/warm_s2.json |
| side row | - | parent k_att=3 adapter s0 (T=1 single-head monitor) | old attacker | S0 1.000; S(8) 0.000; k_min 5 | data/kaudit/parent_ao_s0.json |
| side row | - | parent k_att=3 adapter s1 (T=1 single-head monitor) | old attacker | S0 1.000; S(8) 0.000; k_min 5 | data/kaudit/parent_ao_s1.json |
| side row | - | parent k_att=3 adapter s2 (T=1 single-head monitor) | old attacker | S0 0.008; S(8) 0.000; k_min 1 | data/kaudit/parent_ao_s2.json |

- seed-level: k_att=6 S(8): 0.016 [-0.084, 0.115] (95% t-interval over 2 seeds (df 1, t = 12.706))
- seed-level: k_att=6 k_min: 6.500 [0.147, 12.853] (95% t-interval over 2 seeds (df 1, t = 12.706); k_min > 12 counted as 13)
- seed-level: k_att=6 R8: 0.963 [0.541, 1.385] (95% t-interval over 2 seeds (df 1, t = 12.706))
- seed-level: k_att=3 S(8): 0.004 [-0.046, 0.054] (95% t-interval over 2 seeds (df 1, t = 12.706))
- seed-level: k_att=3 k_min: 2.500 [-16.559, 21.559] (95% t-interval over 2 seeds (df 1, t = 12.706); k_min > 12 counted as 13)
- seed-level: k_att=3 R8: 0.477 [-5.380, 6.333] (95% t-interval over 2 seeds (df 1, t = 12.706))

Status UNCONFIRMED (fp32 rows UNCONFIRMED, bf16 rows UNCONFIRMED); counted k_att=6 seeds ['k6_s0', 'k6_s3']; notes ['k6_s1: faded (S0 0.000)', 'k6_s2: faded (S0 0.008)']. Prereg outcome failed.
