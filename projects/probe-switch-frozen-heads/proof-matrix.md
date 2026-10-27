# Proof matrix — probe-switch-frozen-heads

One row per cell. Switch = strict (PREREG: top-1 strictly above every other token, both conditions), 128 held-out test prompts. 'whole' = merged bf16 LoRA in the whole compiled graph (title score); 'tail' = the eager training tail on cached inputs. Built by summary.py from the artifacts named in the last column.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | all-modules LoRA, blocks 20-27 | positive control | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (896 updates) | data/whole/all8_none_s0.json |
| headline | 0 | all-modules LoRA, blocks 20-27 (tail) | positive control | tail strict 1.000, argmax 1.000; T75 768; steps 896 | data/runs/all8_none_s0.json |
| stall check | 0 | all-modules LoRA, blocks 20-27 | positive control | median RMS grad/param (upd 1..768) 1.88e-03; loss last 128 median 0.0034 mean 0.6051; slope/1k -0.3406 | data/runs/all8_none_s0.json (grad_log) |
| audit | 0 | all-modules LoRA, blocks 20-27 | random 3-head null | clamp all 128 heads removes 1.000 of the push; block 26 alone 0.477; blind top-3 None R None (null p95 None) | data/audit/all8_none_s0.json |
| after-map | 0 | all-modules LoRA, blocks 20-27 | none | max answer-token AUROC 1.000 at mid27 | data/after_map/all8_none_s0.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) | none | whole strict 0.000, argmax 0.000, unprobed-B 0.000, probed-A 1.000 (2048 updates) | data/whole/mlp27_none_s0.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) (tail) | none | tail strict 0.000, argmax 0.000; T75 None; steps 2048 | data/runs/mlp27_none_s0.json |
| stall check | 0 | last-MLP LoRA (block 27 MLP only) | none | median RMS grad/param (upd 1..768) 3.42e-03; loss last 128 median 0.6928 mean 0.6942; slope/1k 0.0003 | data/runs/mlp27_none_s0.json (grad_log) |
| mlp27 | 1 | last-MLP LoRA (block 27 MLP only) | none | whole strict 0.000, argmax 0.039, unprobed-B 0.727, probed-A 0.281 (2048 updates) | data/whole/mlp27_none_s1.json |
| mlp27 | 1 | last-MLP LoRA (block 27 MLP only) (tail) | none | tail strict 0.023, argmax 0.062; T75 None; steps 2048 | data/runs/mlp27_none_s1.json |
| stall check | 1 | last-MLP LoRA (block 27 MLP only) | none | median RMS grad/param (upd 1..768) 3.53e-03; loss last 128 median 0.6930 mean 0.6939; slope/1k 0.0022 | data/runs/mlp27_none_s1.json (grad_log) |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) | shuffled labels | whole strict 0.000, argmax 0.039, unprobed-B 0.188, probed-A 0.836 (2048 updates) | data/whole/mlp27_shuffled_s0.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) (tail) | shuffled labels | tail strict 0.016, argmax 0.031; T75 None; steps 2048 | data/runs/mlp27_shuffled_s0.json |
| stall check | 0 | last-MLP LoRA (block 27 MLP only) | shuffled labels | median RMS grad/param (upd 1..768) 3.10e-03; loss last 128 median 0.6937 mean 0.6942; slope/1k -0.0000 | data/runs/mlp27_shuffled_s0.json (grad_log) |
| headline | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (1152 updates) | data/whole/mlp8_none_s0.json |
| headline | 0 | MLP-only LoRA, blocks 20-27, heads frozen (tail) | none | tail strict 1.000, argmax 1.000; T75 1024; steps 1152 | data/runs/mlp8_none_s0.json |
| stall check | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | median RMS grad/param (upd 1..768) 1.12e-03; loss last 128 median 0.0000 mean 0.0000; slope/1k -2.0132 | data/runs/mlp8_none_s0.json (grad_log) |
| audit | 0 | MLP-only LoRA, blocks 20-27, heads frozen | random 3-head null | clamp all 128 heads removes 1.000 of the push; block 26 alone 0.986; blind top-3 ['26.2', '26.0', '26.8'] R 0.765 (null p95 0.107) | data/audit/mlp8_none_s0.json |
| LoRA split | 0 | MLP-only LoRA, blocks 20-27, heads frozen | LoRA off at some positions | strict: only answer token 0.000, only earlier tokens 0.922, without block 20 0.000, without block 26 1.000 | data/audit/mlp8_none_s0.json |
| after-map | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | max answer-token AUROC 1.000 at in26 | data/after_map/mlp8_none_s0.json |
| headline | 1 | MLP-only LoRA, blocks 20-27, heads frozen | none | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (1152 updates) | data/whole/mlp8_none_s1.json |
| headline | 1 | MLP-only LoRA, blocks 20-27, heads frozen (tail) | none | tail strict 1.000, argmax 1.000; T75 1024; steps 1152 | data/runs/mlp8_none_s1.json |
| stall check | 1 | MLP-only LoRA, blocks 20-27, heads frozen | none | median RMS grad/param (upd 1..768) 1.20e-03; loss last 128 median 0.0001 mean 0.0002; slope/1k -1.5579 | data/runs/mlp8_none_s1.json (grad_log) |
| headline | - | untrained model | untrained model | whole strict 0.000, unprobed-B 0.938 | data/whole/base.json |
| map clause 1 | - | base model, eager tail | 200-flip label null | max answer-token AUROC over block inputs 20-27 0.540 (in22@79) | research_checks/map.json |
| last-MLP prediction | - | base model, block-27 post-attention, answer token | 200-flip label null | predicted last-MLP switch {"acc": 0.53515625, "predicted_strict_switch": 0.0703125, "tolerance": 0.1, "bit_ceiling": 1.0} | research_checks/map.json |
