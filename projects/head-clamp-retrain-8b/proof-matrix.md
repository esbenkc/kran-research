# Proof matrix: head-clamp-retrain-8b

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| engine gate (1.7B port bit-identical) | 0 | SET, grid 128 | none | returned at 512, 512/512 losses = parent | data/engine17b/runs/set_s0_g128.json |
| G0 trace at the answer position | - | probe on vs off | none | d34 = 1.000 (test) | data/g0.json |
| switch formed | 0 | switch | none | formed True, T_f 640, test strict 1.000 | data/runs/switch_s0.json |
| switch formed | 1 | switch | none | formed True, T_f 704, test strict 0.992 | data/runs/switch_s1.json |
| switch formed | 2 | switch | none | formed True, T_f 832, test strict 0.992 | data/runs/switch_s2.json |
| headline | 0 | set (4 heads) | none | returned at 256/1280, test strict 1.000 [0.971, 1.000] | data/runs/set_s0.json |
| headline | 1 | set (4 heads) | none | returned at 192/1408, test strict 1.000 [0.971, 1.000] | data/runs/set_s1.json |
| headline | 2 | set (5 heads) | none | returned at 320/1664, test strict 0.969 [0.922, 0.988] | data/runs/set_s2.json |
| arm all64 | 0 | all64 (64 heads) | none | budget at 1280/1280, test strict 0.000 [0.000, 0.029] | data/runs/all64_s0.json |
| 1.7B band | 0 | SET, grid 64 | none | returned at 192, r = 0.150 | data/band17b/runs/set_s0.json |
| 1.7B band | 1 | SET, grid 64 | none | returned at 384, r = 0.273 | data/band17b/runs/set_s1.json |
| 1.7B band | 2 | SET, grid 64 | none | returned at 256, r = 0.200 | data/band17b/runs/set_s2.json |
| 1.7B reproduction (exploratory, not frozen) | 0 | top1, grid 64 | none | returned at 192/4096, final test strict 0.992 | data/band17b/runs/top1_s0.json |
| 1.7B reproduction (exploratory, not frozen) | 1 | top1, grid 64 | none | returned at 192/4096, final test strict 0.703 | data/band17b/runs/top1_s1.json |
| 1.7B reproduction (exploratory, not frozen) | 2 | top1, grid 64 | none | returned at 128/4096, final test strict 0.594 | data/band17b/runs/top1_s2.json |
| 1.7B reproduction (exploratory, not frozen) | 0 | leave1, grid 64 | none | returned at 768/4096, final test strict 0.984 | data/band17b/runs/leave1_s0.json |
| 1.7B reproduction (exploratory, not frozen) | 1 | leave1, grid 64 | none | returned at 1984/4096, final test strict 0.859 | data/band17b/runs/leave1_s1.json |
| 1.7B reproduction (exploratory, not frozen) | 2 | leave1, grid 64 | none | returned at 1088/4096, final test strict 1.000 | data/band17b/runs/leave1_s2.json |
| 1.7B reproduction (exploratory, not frozen) | 0 | all32, grid 64 | none | budget at 4096/4096, final test strict 0.016; test at 2304: 0.023, test at 2560: 0.031 | data/band17b/runs/all32_s0.json |
| 1.7B reproduction (exploratory, not frozen) | 1 | all32, grid 64 | none | budget at 4096/4096, final test strict 0.039; test at 2688: 0.023, test at 2816: 0.023 | data/band17b/runs/all32_s1.json |
| 1.7B reproduction (exploratory, not frozen) | 2 | all32, grid 64 | none | budget at 4096/4096, final test strict 0.023; test at 2432: 0.055, test at 2560: 0.055 | data/band17b/runs/all32_s2.json |
| 1.7B band sensitivity (T_f on the 64 grid) | 0 | switch from scratch, grid 64 | none | T_f 1152 (parent 128 grid 1280), r = 0.167, parent evals equal 13/13 | data/sens17b/runs/switch_s0.json |
| 1.7B band sensitivity (T_f on the 64 grid) | 1 | switch from scratch, grid 64 | none | T_f 1344 (parent 128 grid 1408), r = 0.286, parent evals equal 16/16 | data/sens17b/runs/switch_s1.json |
| 1.7B band sensitivity (T_f on the 64 grid) | 2 | switch from scratch, grid 64 | none | T_f 1216 (parent 128 grid 1280), r = 0.211, parent evals equal 14/14 | data/sens17b/runs/switch_s2.json |

seed-level: 0.990 [0.945, 1.034] (95% t-interval over 3 seeds (df 2), SET test strict)
status (frozen rule): REJECTED (SET back in 3 of 3 seeds); prereg outcome: unsettled
