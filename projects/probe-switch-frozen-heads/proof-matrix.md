# Proof matrix — probe-switch-frozen-heads

One row per cell. Switch = strict (PREREG: top-1 strictly above every other token, both conditions), 128 held-out test prompts. 'whole' = merged bf16 LoRA in the whole compiled graph (title score); 'tail' = the eager training tail on cached inputs. Built by summary.py from the artifacts named in the last column.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | all-modules LoRA, blocks 20-27 | positive control | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (896 updates) | data/whole/all8_none_s0.json |
| headline | 0 | all-modules LoRA, blocks 20-27 (tail) | positive control | tail strict 1.000, argmax 1.000; T75 768; steps 896 | data/runs/all8_none_s0.json |
| stall check | 0 | all-modules LoRA, blocks 20-27 | positive control | median RMS grad/param (upd 1..768) 1.88e-03; loss last 128 median 0.0034 mean 0.6051; slope/1k -0.3406 (grad_log) | data/runs/all8_none_s0.json |
| audit | 0 | all-modules LoRA, blocks 20-27 | random 3-head null | clamp all 128 heads removes 1.000 of the push; block 26 alone 0.477; blind top-3 None R None (null p95 None) | data/audit/all8_none_s0.json |
| after-map | 0 | all-modules LoRA, blocks 20-27 | none | max answer-token AUROC 1.000 at mid27 | data/after_map/all8_none_s0.json |
| headline | 1 | all-modules LoRA, blocks 20-27 | positive control | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (896 updates) | data/whole/all8_none_s1.json |
| headline | 1 | all-modules LoRA, blocks 20-27 (tail) | positive control | tail strict 1.000, argmax 1.000; T75 768; steps 896 | data/runs/all8_none_s1.json |
| stall check | 1 | all-modules LoRA, blocks 20-27 | positive control | median RMS grad/param (upd 1..768) 2.02e-03; loss last 128 median 0.0000 mean 0.0256; slope/1k -2.1090 (grad_log) | data/runs/all8_none_s1.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) | none | whole strict 0.000, argmax 0.000, unprobed-B 0.000, probed-A 1.000 (2048 updates) | data/whole/mlp27_none_s0.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) (tail) | none | tail strict 0.000, argmax 0.000; T75 None; steps 2048 | data/runs/mlp27_none_s0.json |
| stall check | 0 | last-MLP LoRA (block 27 MLP only) | none | median RMS grad/param (upd 1..768) 3.42e-03; loss last 128 median 0.6928 mean 0.6942; slope/1k 0.0003 (grad_log) | data/runs/mlp27_none_s0.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) | none | whole strict 0.016, argmax 0.023, unprobed-B 0.734, probed-A 0.273 (8192 updates) | data/whole/mlp27_none_s0_8k.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) (tail) | none | tail strict 0.031, argmax 0.062; T75 None; steps 8192 | data/runs/mlp27_none_s0_8k.json |
| stall check | 0 | last-MLP LoRA (block 27 MLP only) | none | median RMS grad/param (upd 1..768) 3.42e-03; loss last 128 median 0.6919 mean 0.7131; slope/1k 0.0240 (grad_log) | data/runs/mlp27_none_s0_8k.json |
| mlp27 | 1 | last-MLP LoRA (block 27 MLP only) | none | whole strict 0.000, argmax 0.039, unprobed-B 0.727, probed-A 0.281 (2048 updates) | data/whole/mlp27_none_s1.json |
| mlp27 | 1 | last-MLP LoRA (block 27 MLP only) (tail) | none | tail strict 0.023, argmax 0.062; T75 None; steps 2048 | data/runs/mlp27_none_s1.json |
| stall check | 1 | last-MLP LoRA (block 27 MLP only) | none | median RMS grad/param (upd 1..768) 3.53e-03; loss last 128 median 0.6930 mean 0.6939; slope/1k 0.0022 (grad_log) | data/runs/mlp27_none_s1.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) | shuffled labels | whole strict 0.000, argmax 0.039, unprobed-B 0.188, probed-A 0.836 (2048 updates) | data/whole/mlp27_shuffled_s0.json |
| mlp27 | 0 | last-MLP LoRA (block 27 MLP only) (tail) | shuffled labels | tail strict 0.016, argmax 0.031; T75 None; steps 2048 | data/runs/mlp27_shuffled_s0.json |
| stall check | 0 | last-MLP LoRA (block 27 MLP only) | shuffled labels | median RMS grad/param (upd 1..768) 3.10e-03; loss last 128 median 0.6937 mean 0.6942; slope/1k -0.0000 (grad_log) | data/runs/mlp27_shuffled_s0.json |
| headline | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (1152 updates) | data/whole/mlp8_none_s0.json |
| headline | 0 | MLP-only LoRA, blocks 20-27, heads frozen (tail) | none | tail strict 1.000, argmax 1.000; T75 1024; steps 1152 | data/runs/mlp8_none_s0.json |
| stall check | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | median RMS grad/param (upd 1..768) 1.12e-03; loss last 128 median 0.0000 mean 0.0000; slope/1k -2.0132 (grad_log) | data/runs/mlp8_none_s0.json |
| audit | 0 | MLP-only LoRA, blocks 20-27, heads frozen | random 3-head null | clamp all 128 heads removes 1.000 of the push; block 26 alone 0.986; blind top-3 ['26.2', '26.0', '26.8'] R 0.765 (null p95 0.107) | data/audit/mlp8_none_s0.json |
| LoRA split | 0 | MLP-only LoRA, blocks 20-27, heads frozen | LoRA off at some positions | strict: only answer token 0.000, only earlier tokens 0.922, without block 20 0.000, without block 26 1.000 | data/audit/mlp8_none_s0.json |
| after-map | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | max answer-token AUROC 1.000 at in26 | data/after_map/mlp8_none_s0.json |
| headline | 1 | MLP-only LoRA, blocks 20-27, heads frozen | none | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (1152 updates) | data/whole/mlp8_none_s1.json |
| headline | 1 | MLP-only LoRA, blocks 20-27, heads frozen (tail) | none | tail strict 1.000, argmax 1.000; T75 1024; steps 1152 | data/runs/mlp8_none_s1.json |
| stall check | 1 | MLP-only LoRA, blocks 20-27, heads frozen | none | median RMS grad/param (upd 1..768) 1.20e-03; loss last 128 median 0.0001 mean 0.0002; slope/1k -1.5579 (grad_log) | data/runs/mlp8_none_s1.json |
| audit | 1 | MLP-only LoRA, blocks 20-27, heads frozen | random 3-head null | clamp all 128 heads removes 1.000 of the push; block 26 alone 0.586; blind top-3 ['27.12', '27.7', '27.13'] R 0.963 (null p95 0.078) | data/audit/mlp8_none_s1.json |
| LoRA split | 1 | MLP-only LoRA, blocks 20-27, heads frozen | LoRA off at some positions | strict: only answer token 0.000, only earlier tokens 0.477, without block 20 0.000, without block 26 1.000 | data/audit/mlp8_none_s1.json |
| after-map | 1 | MLP-only LoRA, blocks 20-27, heads frozen | none | max answer-token AUROC 1.000 at mid27 | data/after_map/mlp8_none_s1.json |
| headline | 2 | MLP-only LoRA, blocks 20-27, heads frozen | none | whole strict 1.000, argmax 1.000, unprobed-B 1.000, probed-A 1.000 (1152 updates) | data/whole/mlp8_none_s2.json |
| headline | 2 | MLP-only LoRA, blocks 20-27, heads frozen (tail) | none | tail strict 1.000, argmax 1.000; T75 1024; steps 1152 | data/runs/mlp8_none_s2.json |
| stall check | 2 | MLP-only LoRA, blocks 20-27, heads frozen | none | median RMS grad/param (upd 1..768) 1.10e-03; loss last 128 median 0.0000 mean 0.0000; slope/1k -1.7515 (grad_log) | data/runs/mlp8_none_s2.json |
| headline | - | untrained model | untrained model | whole strict 0.000, unprobed-B 0.938 | data/whole/base.json |
| map clause 1 | - | base model, eager tail | 200-flip label null | max answer-token AUROC over block inputs 20-27 0.540 (in22@79) | research_checks/map.json |
| last-MLP prediction | - | base model, block-27 post-attention, answer token | 200-flip label null | predicted last-MLP switch {"acc": 0.53515625, "predicted_strict_switch": 0.0703125, "tolerance": 0.1, "bit_ceiling": 1.0} | research_checks/map.json |
| frozen-key gate | - | untrained model | untrained model | answer-token attention moved by the probe (TV): 0 heads > 0.3; blocks 20-25 max 0.007 | research_checks/after_attn/attn_base_test.json |
| frozen-key gate | 0 | all-modules LoRA, blocks 20-27 | positive control | answer-token attention moved by the probe (TV): 2 heads > 0.3: 27.12 0.91; 27.14 0.34; blocks 20-25 max 0.013 | research_checks/after_attn/attn_all8_none_s0_adapter_test.json |
| frozen-key gate | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | answer-token attention moved by the probe (TV): 8 heads > 0.3: 27.1 0.82 (query-only 0.50, key-only 0.62); 26.2 0.78 (query-only 0.01, key-only 0.78); 27.9 0.72 (query-only 0.26, key-only 0.44); 26.10 0.52 (query-only 0.00, key-only 0.53); blocks 20-25 max 0.019 | research_checks/after_attn/attn_mlp8_none_s0_adapter_test.json |
| frozen-key gate | 1 | MLP-only LoRA, blocks 20-27, heads frozen | none | answer-token attention moved by the probe (TV): 3 heads > 0.3: 27.12 0.99 (query-only 0.00, key-only 0.99); 27.13 0.99 (query-only 0.00, key-only 0.99); 27.7 0.92 (query-only 0.00, key-only 0.91); blocks 20-25 max 0.011 | research_checks/after_attn/attn_mlp8_none_s1_adapter_test.json |
| frozen-key gate | 2 | MLP-only LoRA, blocks 20-27, heads frozen | none | answer-token attention moved by the probe (TV): 3 heads > 0.3: 27.15 0.98 (query-only 0.01, key-only 0.98); 27.12 0.78 (query-only 0.01, key-only 0.78); 27.1 0.64 (query-only 0.01, key-only 0.62); blocks 20-25 max 0.014 | research_checks/after_attn/attn_mlp8_none_s2_adapter_test.json |
| headline (Vera rebuild) | 0 | MLP-only LoRA, own CPU fp32 tail, adapter MLP tensors only | none | strict 128/128 = 1.000, push 24.27 | skeptics/skeptic/out/pos_mlp8_s0.json |
| headline (Vera rebuild) | 1 | MLP-only LoRA, own CPU fp32 tail | none | strict 128/128 = 1.000, push 20.25 | skeptics/skeptic/out/pos_mlp8_s1.json |
| headline (Vera rebuild) | 2 | MLP-only LoRA, own CPU fp32 tail | none | strict 128/128 = 1.000, push 24.43 | skeptics/skeptic/out/pos_mlp8_s2.json |
| headline (Vera rebuild) | 0,1,2 | MLP-only LoRA, build's whole-graph rows, own scorer | shuffled-condition null (2000 swaps) | strict 1.000 x3 vs null p95 0.578 | skeptics/skeptic/vera_switch.py |
| control (Vera rebuild) | 0 | all-modules LoRA, own CPU fp32 tail | positive control | strict 128/128, push 30.06 | skeptics/skeptic/out/tail_all8_s0.json |
| control (Vera rebuild) | - | untrained model, own CPU fp32 tail | untrained model | strict 0/128 | skeptics/skeptic/out/tail_base.json |
| mlp27 (Vera rebuild) | 0 | last-MLP LoRA, own CPU fp32 tail | none | strict 0/128 | skeptics/skeptic/out/tail_mlp27_s0.json |
| map clause 1 (Vera rebuild) | - | base eager tail, pos 79 | 20-flip within-pair null | max AUROC in20-27 0.538 (in22); null p95 0.505-0.512 | skeptics/skeptic/out_map_eager79.json |
| audit (Vera rebuild) | 0 | MLP-only, own blind greedy top-3 | 50 random 3-head sets | picks 26.2, 26.0, 26.8; R_blind3 0.747 (null p95 0.147) | skeptics/skeptic/out/audit_mlp8_s0.json |
| route (Vera) | 0,1,2 | MLP-only, LoRA delta only at answer / only at earlier tokens | LoRA off at some positions | answer-only 0/128 x3; earlier-only 118, 61, 125 of 128 | skeptics/skeptic/out/pos_mlp8_s*.json |
| route (Vera) | 0,1 | MLP-only, answer-position head clamp, blocks 20-27 | clamp | removed 0.9999 / 1.000 of the push; switch -> 0 | skeptics/skeptic/out/clamp_mlp8_s*.json |
