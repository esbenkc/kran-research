# Proof matrix — push-lift-necessity

Status (summarize.py, PREREG v1 + D1 + D2): **CONFIRMED**. Clean line L = +0.107. Why: all gates met.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| baseline | 0 | push penalty @ 2048 | none | gap n/a pts; push score +3.987 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P2_s0_at2048_scr.json |
| baseline | 0 | push penalty @ 3072 | none | gap n/a pts; push score +7.262 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P2_s0_at3072_scr.json |
| baseline | 0 | push penalty @ 4096 | none | gap n/a pts; push score +13.991 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P2_s0_at4096_scr.json |
| baseline | 0 | push penalty @ 5120 | none | gap n/a pts; push score +12.745 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P2_s0_at5120_scr.json |
| baseline | 0 | push penalty @ 6144 | none | gap n/a pts; push score +9.668 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P2_s0_at6144_scr.json |
| baseline | 0 | push penalty @ 7168 | none | gap n/a pts; push score +10.280 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P2_s0_at7168_scr.json |
| baseline | 0 | push penalty @ 8192 | none | gap +39.32 pts; push score +9.854 (runs: screen/full/confirm max) vs line +0.107; pool score at last refill +9.702 | data/audit/P2_s0_scr.json |
| baseline | 0 | push penalty @ 8192 | none | same cell, second artifact | data/eval/P2_s0/eager_8192.json |
| baseline | 0 | push penalty @ 8192 | none | same cell, second artifact | data/eval/P2_s0/compiled_8192.json |
| headline | 0 | push penalty @ 2048 | none | gap n/a pts; push score +3.987 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s0_at2048_scr.json |
| headline | 0 | push penalty @ 3072 | none | gap n/a pts; push score +13.200 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s0_at3072_scr.json |
| headline | 0 | push penalty @ 4096 | none | gap n/a pts; push score +14.115 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s0_at4096_scr.json |
| headline | 0 | push penalty @ 5120 | none | gap +31.90 pts; push score +14.659 (runs: screen/full/confirm max) vs line +0.107; pool score at last refill +13.800 | data/audit/P_s0_at5120_scr.json |
| headline | 0 | push penalty @ 5120 | none | same cell, second artifact | data/eval/P_s0/eager_5120.json |
| headline | 0 | push penalty @ 5120 | none | same cell, second artifact | data/eval/P_s0/compiled_5120.json |
| headline | 1 | push penalty @ 16384 | none | gap +26.18 pts; push score +2.201 (runs: screen/full/confirm max) vs line +0.107; pool score at last refill +3.792 | data/audit/P_s1_scr.json |
| headline | 1 | push penalty @ 16384 | none | same cell, second artifact | data/audit/P_s1.json |
| headline | 1 | push penalty @ 16384 | none | same cell, second artifact | data/eval/P_s1/eager_16384.json |
| headline | 1 | push penalty @ 16384 | none | same cell, second artifact | data/eval/P_s1/compiled_16384.json |
| headline | 1 | push penalty @ 10240 | none | gap n/a pts; push score +10.798 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at10240_scr.json |
| headline | 1 | push penalty @ 11264 | none | gap n/a pts; push score +9.211 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at11264_scr.json |
| headline | 1 | push penalty @ 12288 | none | gap n/a pts; push score +7.096 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at12288_scr.json |
| headline | 1 | push penalty @ 13312 | none | gap n/a pts; push score +2.695 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at13312_scr.json |
| headline | 1 | push penalty @ 14336 | none | gap n/a pts; push score +4.216 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at14336_scr.json |
| headline | 1 | push penalty @ 15360 | none | gap n/a pts; push score +1.240 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at15360_scr.json |
| headline | 1 | push penalty @ 3072 | none | gap n/a pts; push score +4.331 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at3072_scr.json |
| headline | 1 | push penalty @ 4096 | none | gap n/a pts; push score +10.630 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at4096_scr.json |
| headline | 1 | push penalty @ 5120 | none | gap n/a pts; push score +15.043 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at5120_scr.json |
| headline | 1 | push penalty @ 7168 | none | gap n/a pts; push score +16.363 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at7168_scr.json |
| headline | 1 | push penalty @ 8192 | none | gap n/a pts; push score +12.120 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at8192_scr.json |
| headline | 1 | push penalty @ 9216 | none | gap n/a pts; push score +11.930 (runs: screen/full/confirm max) vs line +0.107 | data/audit/P_s1_at9216_scr.json |
| baseline | n/a | untrained @ 0 | untrained model | gap +0.00 pts; push score +0.000 (runs: screen/full/confirm max) vs line +0.107 | data/audit/base_scr.json |
| baseline | n/a | untrained @ 0 | untrained model | same cell, second artifact | data/audit/base.json |
| baseline | n/a | untrained @ 0 | untrained model | same cell, second artifact | data/audit/base_c2.json |
| baseline | n/a | untrained @ 0 | untrained model | same cell, second artifact | ref/parent_data/eval/base/eager_0.json |
| baseline | n/a | untrained @ 0 | untrained model | same cell, second artifact | ref/parent_data/eval/base/compiled_0.json |
| clean line | 0 | clean @ 8192 | clean model | gap -0.34 pts; push score +0.026 (runs: screen/full/confirm max) vs line +0.107 | data/audit/cln_s0_scr.json |
| clean line | 0 | clean @ 8192 | clean model | same cell, second artifact | data/audit/cln_s0.json |
| clean line | 0 | clean @ 8192 | clean model | same cell, second artifact | data/audit/cln_s0_c2.json |
| clean line | 0 | clean @ 8192 | clean model | same cell, second artifact | ref/parent_data/eval/cln_s0/eager_8192.json |
| clean line | 0 | clean @ 8192 | clean model | same cell, second artifact | ref/parent_data/eval/cln_s0/compiled_8192.json |
| clean line | 1 | clean @ 8192 | clean model | gap -0.22 pts; push score +0.064 (runs: screen/full/confirm max) vs line +0.107 | data/audit/cln_s1_scr.json |
| clean line | 1 | clean @ 8192 | clean model | same cell, second artifact | data/audit/cln_s1.json |
| clean line | 1 | clean @ 8192 | clean model | same cell, second artifact | data/audit/cln_s1_c2.json |
| clean line | 1 | clean @ 8192 | clean model | same cell, second artifact | ref/parent_data/eval/cln_s1/eager_8192.json |
| clean line | 1 | clean @ 8192 | clean model | same cell, second artifact | ref/parent_data/eval/cln_s1/compiled_8192.json |
| clean line | 2 | clean @ 8192 | clean model | gap -0.46 pts; push score +0.043 (runs: screen/full/confirm max) vs line +0.107 | data/audit/cln_s2_scr.json |
| clean line | 2 | clean @ 8192 | clean model | same cell, second artifact | data/audit/cln_s2.json |
| clean line | 2 | clean @ 8192 | clean model | same cell, second artifact | data/audit/cln_s2_c2.json |
| clean line | 2 | clean @ 8192 | clean model | same cell, second artifact | ref/parent_data/eval/cln_s2/eager_8192.json |
| clean line | 2 | clean @ 8192 | clean model | same cell, second artifact | ref/parent_data/eval/cln_s2/compiled_8192.json |
| positive control | 0 | no penalty @ 8192 | no penalty (U twin) | gap +43.68 pts; push score +3.016 (runs: screen/full/confirm max) vs line +0.107 | data/audit/stack_s0_scr.json |
| positive control | 0 | no penalty @ 8192 | no penalty (U twin) | same cell, second artifact | ref/parent_data/eval/stack_s0/eager_8192.json |
| positive control | 0 | no penalty @ 8192 | no penalty (U twin) | same cell, second artifact | ref/parent_data/eval/stack_s0/compiled_8192.json |
| positive control | 1 | no penalty @ 8192 | no penalty (U twin) | gap +21.72 pts; push score +4.571 (runs: screen/full/confirm max) vs line +0.107 | data/audit/stack_s1_scr.json |
| positive control | 1 | no penalty @ 8192 | no penalty (U twin) | same cell, second artifact | ref/parent_data/eval/stack_s1/eager_8192.json |
| positive control | 1 | no penalty @ 8192 | no penalty (U twin) | same cell, second artifact | ref/parent_data/eval/stack_s1/compiled_8192.json |
| path B offset (D3) | 0 | Push penalty kept live, seed 0 @ 8192, refill 6144, unit 0 | refill-0 offset on untrained | lift +4.19 (r0 +0.49), sat 0.31, next +4.06; i True ii True iii True iv True | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty kept live, seed 0 @ 8192, refill 6144, unit 1-2 | refill-0 offset on untrained | lift +0.01 (r0 +0.02), sat 0.00, next +0.01; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty kept live, seed 0 @ 8192, refill 6144, unit 1-8 | refill-0 offset on untrained | lift +0.04 (r0 +0.05), sat 0.00, next +0.02; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty kept live, seed 0 @ 8192, refill 6144, unit 9-16 | refill-0 offset on untrained | lift +0.05 (r0 +0.12), sat 0.00, next +0.05; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty kept live, seed 0 @ 8192, refill 6144, unit 17-24 | refill-0 offset on untrained | lift +0.34 (r0 +0.64), sat 0.00, next +0.31; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty kept live, seed 0 @ 8192, refill 6144, unit 25-32 | refill-0 offset on untrained | lift +0.09 (r0 +0.14), sat 0.00, next +0.09; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty, seed 0 @ 5120, refill 4096, unit 0 | refill-0 offset on untrained | lift +13.92 (r0 +0.49), sat 1.00, next +15.02; i True ii False iii False iv True | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty, seed 0 @ 5120, refill 4096, unit 1-2 | refill-0 offset on untrained | lift +0.01 (r0 +0.02), sat 0.01, next +0.02; i False ii True iii False iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty, seed 0 @ 5120, refill 4096, unit 1-8 | refill-0 offset on untrained | lift +0.04 (r0 +0.05), sat 0.01, next +0.04; i False ii True iii False iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty, seed 0 @ 5120, refill 4096, unit 9-16 | refill-0 offset on untrained | lift +0.07 (r0 +0.12), sat 0.01, next +0.07; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty, seed 0 @ 5120, refill 4096, unit 17-24 | refill-0 offset on untrained | lift +0.31 (r0 +0.64), sat 0.01, next +0.31; i False ii True iii False iv False | data/readings/pool_lift.json |
| path B offset (D3) | 0 | Push penalty, seed 0 @ 5120, refill 4096, unit 25-32 | refill-0 offset on untrained | lift +0.21 (r0 +0.14), sat 0.01, next +0.16; i False ii True iii True iv False | data/readings/pool_lift.json |
| zero-start search (D5) | 1 | Push penalty, seed 1 @ 16384 | clean zero-start line | zero-start score +2.196 vs L_zero +0.110 | data/audit/P_s1_z.json |
| path B offset (D3) | 1 | Push penalty, seed 1 @ 16384, refill 14336, unit 0 | refill-0 offset on untrained | lift +4.53 (r0 +0.48), sat 0.33, next -0.74; i True ii True iii True iv True | data/readings/pool_lift.json |
| path B offset (D3) | 1 | Push penalty, seed 1 @ 16384, refill 14336, unit 1-2 | refill-0 offset on untrained | lift +0.00 (r0 +0.02), sat 0.00, next +0.00; i False ii True iii False iv False | data/readings/pool_lift.json |
| path B offset (D3) | 1 | Push penalty, seed 1 @ 16384, refill 14336, unit 1-8 | refill-0 offset on untrained | lift +0.03 (r0 +0.06), sat 0.00, next +0.02; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 1 | Push penalty, seed 1 @ 16384, refill 14336, unit 9-16 | refill-0 offset on untrained | lift +0.04 (r0 +0.12), sat 0.00, next +0.04; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 1 | Push penalty, seed 1 @ 16384, refill 14336, unit 17-24 | refill-0 offset on untrained | lift +0.44 (r0 +0.64), sat 0.00, next +0.35; i False ii True iii True iv False | data/readings/pool_lift.json |
| path B offset (D3) | 1 | Push penalty, seed 1 @ 16384, refill 14336, unit 25-32 | refill-0 offset on untrained | lift +0.03 (r0 +0.13), sat 0.00, next +0.02; i False ii True iii True iv False | data/readings/pool_lift.json |
| zero-start search (D5) | n/a | Untrained @ 0 | clean zero-start line | zero-start score +0.000 vs L_zero +0.110 | data/audit/base_z.json |
| zero-start search (D5) | 0 | Clean 0 @ 8192 | clean zero-start line | zero-start score +0.024 vs L_zero +0.110 | data/audit/cln_s0_z.json |
| zero-start search (D5) | 1 | Clean 1 @ 8192 | clean zero-start line | zero-start score +0.062 vs L_zero +0.110 | data/audit/cln_s1_z.json |
| zero-start search (D5) | 2 | Clean 2 @ 8192 | clean zero-start line | zero-start score +0.047 vs L_zero +0.110 | data/audit/cln_s2_z.json |
| penalty live | 0 | push penalty refill @ 0 | refill 0 on untrained | pool score +0.000 | logs/P_s0.log |
| penalty live | 0 | push penalty refill @ 2048 | refill 0 on untrained | pool score +4.222 | logs/P_s0.log |
| penalty live | 0 | push penalty refill @ 4096 | refill 0 on untrained | pool score +13.800 | logs/P_s0.log |
| penalty live | 1 | push penalty refill @ 0 | refill 0 on untrained | pool score +0.000 | data/runs/P_s1.json |
| penalty live | 1 | push penalty refill @ 2048 | refill 0 on untrained | pool score +1.574 | data/runs/P_s1.json |
| penalty live | 1 | push penalty refill @ 4096 | refill 0 on untrained | pool score +10.475 | data/runs/P_s1.json |
| penalty live | 1 | push penalty refill @ 6144 | refill 0 on untrained | pool score +15.430 | data/runs/P_s1.json |
| penalty live | 1 | push penalty refill @ 8192 | refill 0 on untrained | pool score +11.976 | data/runs/P_s1.json |
| penalty live | 1 | push penalty refill @ 10240 | refill 0 on untrained | pool score +10.637 | data/runs/P_s1.json |
| penalty live | 1 | push penalty refill @ 12288 | refill 0 on untrained | pool score +7.066 | data/runs/P_s1.json |
| penalty live | 1 | push penalty refill @ 14336 | refill 0 on untrained | pool score +3.792 | data/runs/P_s1.json |
| penalty live | 0 | push penalty refill @ 0 | refill 0 on untrained | pool score +0.000 | data/runs/P2_s0.json |
| penalty live | 0 | push penalty refill @ 2048 | refill 0 on untrained | pool score +4.222 | data/runs/P2_s0.json |
| penalty live | 0 | push penalty refill @ 4096 | refill 0 on untrained | pool score +13.983 | data/runs/P2_s0.json |
| penalty live | 0 | push penalty refill @ 6144 | refill 0 on untrained | pool score +9.702 | data/runs/P2_s0.json |
| damage guard | 0 | ARC-Easy paired vs U twin | no penalty (U twin) | acc 0.5781 vs 0.5791; diff -0.10 pts, 95% CI -0.61 to +0.37 | data/qa/P_s0_at5120.json |
| damage guard | 1 | ARC-Easy paired vs U twin | no penalty (U twin) | acc 0.5750 vs 0.5764; diff -0.14 pts, 95% CI -0.61 to +0.37 | data/qa/P_s1.json |
| damage guard | 0 | ARC-Easy paired vs U twin | no penalty (U twin) | acc 0.5716 vs 0.5791; diff -0.75 pts, 95% CI -1.26 to -0.24 | data/qa/P2_s0.json |
| damage guard | n/a | general sensitivity G, untrained | untrained model | G 3.54e-05 nats; void below 1e-4: True | data/readings/kl_sensitivity.json |
