# Skeptic rebuild cells (Vera) — append to proof-matrix.md

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline (Vera rebuild) | 0 | MLP-only LoRA, own CPU fp32 tail, adapter MLP tensors only | none | strict 128/128 = 1.000, push 24.27 | skeptics/skeptic/out/pos_mlp8_s0.json |
| headline (Vera rebuild) | 1 | MLP-only LoRA, own CPU fp32 tail | none | strict 128/128 = 1.000, push 20.25 | skeptics/skeptic/out/pos_mlp8_s1.json |
| headline (Vera rebuild) | 2 | MLP-only LoRA, own CPU fp32 tail | none | strict 128/128 = 1.000, push 24.43 | skeptics/skeptic/out/pos_mlp8_s2.json |
| headline (Vera rebuild) | 0,1,2 | MLP-only LoRA, build's whole-graph rows, own scorer | shuffled-condition null (2000 swaps) | strict 1.000 x3 vs null p95 0.578 | skeptics/skeptic/vera_switch.py |
| control (Vera rebuild) | 0 | all-modules LoRA, own CPU fp32 tail | positive control | strict 128/128, push 30.06 | skeptics/skeptic/out/tail_all8_s0.json |
| control (Vera rebuild) | - | untrained model, own CPU fp32 tail | untrained model | strict 0/128 | skeptics/skeptic/out/tail_base.json |
| mlp27 (Vera rebuild) | 0 | last-MLP LoRA, own CPU fp32 tail | none | strict 0/128 | skeptics/skeptic/out/tail_mlp27_s0.json |
| map clause 1 (Vera rebuild) | - | base eager tail, pos 79 | 20-flip within-pair null | max AUROC in20-27 0.538 (in22); null p95 0.505-0.512 | skeptics/skeptic/out_map_eager79.json |
| audit (Vera rebuild) | 0 | MLP-only, own blind greedy top-3 | 50 random 3-head sets | picks 26.2, 26.0, 26.8; R_blind3 0.747 (null p95 0.147) | skeptics/skeptic/out/audit_mlp8_s0.json |
| route (Vera) | 0 | MLP-only, LoRA delta only at answer / only at earlier tokens | LoRA off at some positions | answer-only 0/128; earlier-only 118 of 128 | skeptics/skeptic/out/pos_mlp8_s0.json |
| route (Vera) | 1 | MLP-only, LoRA delta only at answer / only at earlier tokens | LoRA off at some positions | answer-only 0/128; earlier-only 61 of 128 | skeptics/skeptic/out/pos_mlp8_s1.json |
| route (Vera) | 2 | MLP-only, LoRA delta only at answer / only at earlier tokens | LoRA off at some positions | answer-only 0/128; earlier-only 125 of 128 | skeptics/skeptic/out/pos_mlp8_s2.json |
| route (Vera) | 0 | MLP-only, answer-position head clamp, blocks 20-27 | clamp | removed 0.9999 of the push; switch -> 0 | skeptics/skeptic/out/clamp_mlp8_s0.json |
| route (Vera) | 1 | MLP-only, answer-position head clamp, blocks 20-27 | clamp | removed 1.000 of the push; switch -> 0 | skeptics/skeptic/out/clamp_mlp8_s1.json |
