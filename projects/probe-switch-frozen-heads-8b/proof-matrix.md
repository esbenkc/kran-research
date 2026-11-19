# Proof matrix — probe-switch-frozen-heads-8b

One row per cell. Model Qwen/Qwen3-8B. Switch = strict (PREREG: top-1 strictly above every other token, both conditions), 128 held-out test prompts. 'whole' = merged bf16 LoRA in the whole compiled graph (title score); 'tail' = the eager training tail on cached inputs. Built by summary.py from the artifacts in the last column.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline | 0 | all-modules LoRA, blocks 28-35 | positive control | whole strict 0.992 (Wilson 95% 0.957-0.999), argmax 0.992, unprobed-B 1.000, probed-A 0.992 (1664 updates; G1 gap 0.000) | data/whole/all8_none_s0.json |
| headline | 0 | all-modules LoRA, blocks 28-35 (tail) | positive control | tail strict 0.992, argmax 0.992; T75 1280; steps 1664 | data/runs/all8_none_s0.json |
| stall check | 0 | all-modules LoRA, blocks 28-35 | positive control | median RMS grad/param (upd 1..1280) 3.05e-03; loss last 128 median 0.0000 | data/runs/all8_none_s0.json |
| headline | 1 | all-modules LoRA, blocks 28-35 | positive control | whole strict 0.930 (Wilson 95% 0.872-0.963), argmax 0.930, unprobed-B 1.000, probed-A 0.930 (1408 updates; G1 gap 0.039) | data/whole/all8_none_s1.json |
| headline | 1 | all-modules LoRA, blocks 28-35 (tail) | positive control | tail strict 0.969, argmax 0.969; T75 1280; steps 1408 | data/runs/all8_none_s1.json |
| stall check | 1 | all-modules LoRA, blocks 28-35 | positive control | median RMS grad/param (upd 1..1280) 1.72e-03; loss last 128 median 0.2825 | data/runs/all8_none_s1.json |
| headline | 0 | MLP-only LoRA, blocks 28-35, heads frozen | none | whole strict 0.992 (Wilson 95% 0.957-0.999), argmax 0.992, unprobed-B 1.000, probed-A 0.992 (2432 updates; G1 gap 0.000) | data/whole/mlp8_none_s0.json |
| headline | 0 | MLP-only LoRA, blocks 28-35, heads frozen (tail) | none | tail strict 0.992, argmax 0.992; T75 1920; steps 2432 | data/runs/mlp8_none_s0.json |
| stall check | 0 | MLP-only LoRA, blocks 28-35, heads frozen | none | median RMS grad/param (upd 1..1280) 8.79e-04; loss last 128 median 0.0000 | data/runs/mlp8_none_s0.json |
| headline | 1 | MLP-only LoRA, blocks 28-35, heads frozen | none | whole strict 0.992 (Wilson 95% 0.957-0.999), argmax 0.992, unprobed-B 1.000, probed-A 0.992 (1920 updates; G1 gap 0.000) | data/whole/mlp8_none_s1.json |
| headline | 1 | MLP-only LoRA, blocks 28-35, heads frozen (tail) | none | tail strict 0.992, argmax 0.992; T75 1408; steps 1920 | data/runs/mlp8_none_s1.json |
| stall check | 1 | MLP-only LoRA, blocks 28-35, heads frozen | none | median RMS grad/param (upd 1..1280) 7.51e-03; loss last 128 median 0.0000 | data/runs/mlp8_none_s1.json |
| headline | 2 | MLP-only LoRA, blocks 28-35, heads frozen | none | whole strict 1.000 (Wilson 95% 0.971-1.000), argmax 1.000, unprobed-B 1.000, probed-A 1.000 (1920 updates; G1 gap 0.000) | data/whole/mlp8_none_s2.json |
| headline | 2 | MLP-only LoRA, blocks 28-35, heads frozen (tail) | none | tail strict 1.000, argmax 1.000; T75 1792; steps 1920 | data/runs/mlp8_none_s2.json |
| stall check | 2 | MLP-only LoRA, blocks 28-35, heads frozen | none | median RMS grad/param (upd 1..1280) 7.94e-04; loss last 128 median 0.0003 | data/runs/mlp8_none_s2.json |
| mlplast | 0 | last-MLP LoRA (block 35 MLP only) | none | whole strict 0.000 (Wilson 95% 0.000-0.029), argmax 0.000, unprobed-B 0.000, probed-A 0.992 (2048 updates; G1 gap 0.000) | data/whole/mlplast_none_s0.json |
| mlplast | 0 | last-MLP LoRA (block 35 MLP only) (tail) | none | tail strict 0.000, argmax 0.016; T75 None; steps 2048 | data/runs/mlplast_none_s0.json |
| stall check | 0 | last-MLP LoRA (block 35 MLP only) | none | median RMS grad/param (upd 1..1280) 1.69e-03; loss last 128 median 0.6897 | data/runs/mlplast_none_s0.json |
| mlplast | 1 | last-MLP LoRA (block 35 MLP only) | none | whole strict 0.000 (Wilson 95% 0.000-0.029), argmax 0.008, unprobed-B 0.977, probed-A 0.031 (2048 updates; G1 gap 0.000) | data/whole/mlplast_none_s1.json |
| mlplast | 1 | last-MLP LoRA (block 35 MLP only) (tail) | none | tail strict 0.000, argmax 0.008; T75 None; steps 2048 | data/runs/mlplast_none_s1.json |
| stall check | 1 | last-MLP LoRA (block 35 MLP only) | none | median RMS grad/param (upd 1..1280) 1.90e-03; loss last 128 median 0.6899 | data/runs/mlplast_none_s1.json |
| headline | - | untrained model | untrained model | whole strict 0.000, unprobed-B 0.812 | data/whole/base.json |
| map prediction | - | untrained model, block-35 post-attention, answer token | 200-flip label null | predicted last-MLP strict 0.117 (acc 0.559) | research_checks/prediction.json |
| map prediction | - | untrained model, route rule | 200-flip label null | FORMS: 13 heads >= 0.30 on 70-78, 23 readable source cells; answer-token max AUROC 0.577 | research_checks/prediction.json |
| same route | - | base | untrained model | top TV head 29.11 TV 0.007 -> pos 61; 0 heads > 0.3 | research_checks/after_attn/attn_base_test.json |
seed-level: 0.995 [0.984, 1.000] (95% t-interval over 3 seeds (df 2), clipped to [0, 1]; arm mlp8, whole-graph strict switch; min 0.992; 3 of 3 seeds >= 0.75)
seed-level: 0.961 [0.564, 1.000] (95% t-interval over 2 seeds (df 1), clipped to [0, 1]; arm all8, whole-graph strict switch; min 0.930; 2 of 2 seeds >= 0.75)
seed-level: 0.000 [0.000, 0.000] (95% t-interval over 2 seeds (df 1), clipped to [0, 1]; arm mlplast, whole-graph strict switch; min 0.000; 0 of 2 seeds >= 0.75)
