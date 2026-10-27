# Proof matrix — probe-switch-frozen-heads

One row per cell. Switch = strict (PREREG: top-1 strictly above every other token, both conditions), 128 held-out test prompts. 'whole' = merged bf16 LoRA in the whole compiled graph (title score); 'tail' = the eager training tail on cached inputs. Built by summary.py from the artifacts named in the last column.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | all-modules LoRA, blocks 20-27 | positive control | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (896 updates) | data/whole/all8_none_s0.json |
| headline | 0 | all-modules LoRA, blocks 20-27 (tail) | positive control | tail strict 1.000, argmax 1.000; T75 768; steps 896 | data/runs/all8_none_s0.json |
| stall check | 0 | all-modules LoRA, blocks 20-27 | positive control | median RMS grad/param (upd 1..768) 1.88e-03; loss last 128 median 0.0034 mean 0.6051; slope/1k -0.3406 | data/runs/all8_none_s0.json (grad_log) |
| audit | 0 | all-modules LoRA, blocks 20-27 | none | clamp all 128 heads removes 1.000 of the push; R_blind3 None (null p95 None) | data/audit/all8_none_s0.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) | none | whole strict 0.000, argmax 0.000, unprobed-B 0.000, probed-A 1.000 (2048 updates) | data/whole/mlp27_none_s0.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) (tail) | none | tail strict 0.000, argmax 0.000; T75 None; steps 2048 | data/runs/mlp27_none_s0.json |
| stall check | 0 | last-MLP LoRA (block 27 MLP only) | none | median RMS grad/param (upd 1..768) 3.42e-03; loss last 128 median 0.6928 mean 0.6942; slope/1k 0.0003 | data/runs/mlp27_none_s0.json (grad_log) |
| headline | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (1152 updates) | data/whole/mlp8_none_s0.json |
| headline | 0 | MLP-only LoRA, blocks 20-27, heads frozen (tail) | none | tail strict 1.000, argmax 1.000; T75 1024; steps 1152 | data/runs/mlp8_none_s0.json |
| stall check | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | median RMS grad/param (upd 1..768) 1.12e-03; loss last 128 median 0.0000 mean 0.0000; slope/1k -2.0132 | data/runs/mlp8_none_s0.json (grad_log) |
| headline | - | untrained model | untrained model | whole strict 0.000, unprobed-B 0.938 | data/whole/base.json |
| map clause 1 | - | base model, eager tail | 200-flip label null | max answer-token AUROC over block inputs 20-27 0.540 (in22@79) | research_checks/map.json |
