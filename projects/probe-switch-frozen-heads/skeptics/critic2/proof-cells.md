# critic2 rebuild cells (from its sign-off #8293; its own code, not saved)

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline (critic2 rebuild) | 0 | MLP-only LoRA, blocks 20-27, heads frozen | none | strict 128/128 | data/whole/mlp8_none_s0.json |
| headline (critic2 rebuild) | 1 | MLP-only LoRA, blocks 20-27, heads frozen | none | strict 128/128 | data/whole/mlp8_none_s1.json |
| headline (critic2 rebuild) | 2 | MLP-only LoRA, blocks 20-27, heads frozen | none | strict 128/128 | data/whole/mlp8_none_s2.json |
| headline (critic2 rebuild) | 0 | all-modules LoRA, blocks 20-27 | positive control | strict 128/128 | data/whole/all8_none_s0.json |
| headline (critic2 rebuild) | 1 | all-modules LoRA, blocks 20-27 | positive control | strict 128/128 | data/whole/all8_none_s1.json |
| headline (critic2 rebuild) | - | untrained model | untrained model | strict 0/128 | data/whole/base.json |
| mlp27 (critic2 rebuild) | 0, 1 | last-MLP LoRA | none | strict 0/128, 0/128; 8,192 updates 2/128 | data/whole/mlp27_none_s0.json |
| audit (critic2 rebuild) | 0, 1 | MLP-only LoRA, blocks 20-27, heads frozen | random 3-head null | R_blind3 0.765 / 0.963, null p95 0.107 / 0.078 | data/audit/mlp8_none_s0.json |
