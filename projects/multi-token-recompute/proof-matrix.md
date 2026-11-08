# Proof matrix — multi-token-recompute

Status: **REJECTED** (gate True); scout prereg: unsettled (antecedent false: f = 0.054 (start) and 0.059 (raw) are both < 0.2). Built by summarize.py from data/reads/raw/*.json; skeptics add their rebuild rows below.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| D before training (start policy) | - | start policy, k=16 | none | D 0.0274 [0.0243, 0.0310], f 0.054, f_null 0.000, Pinsker 0.0458 | data/reads/raw/start.json |
| D before training (start policy) | - | start policy, k=16 | f_null: two no-probe runs, different batch layout | f_null 0.000 | data/reads/raw/start.json |
| start cheat | - | start policy | none | cheat none 0.624 / leaky 0.620, gap 0.46 pts [0.01, 0.94] | data/reads/raw/start.json |
| gate | s0 | planted, served read R=8 | positive control (planted text cue, TV 0.30) | gap 17.31 pts [16.22, 18.4], text 17.12 [16.09, 18.15], cheat 0.627/0.454, D_end 0.4141, f 0.399, forced switch 43.34 pts [42.8, 43.88], distinct prefixes 0.170, 200 updates | data/reads/raw/planted_s0.json |
| gate | s0 | planted, training curve | positive control (planted text cue, TV 0.30) | dev gap end 17.47 pts | data/runs/planted_s0/curve.jsonl |
| headline | s0 | k16, served read R=8 | none | gap -0.11 pts [-0.36, 0.14], text 0.0 [0.0, 0.0], cheat 0.773/0.774, D_end 0.0002, f 0.000, distinct prefixes 0.004, 200 updates | data/reads/raw/k16_s0.json |
| headline | s0 | k16, training curve | none | dev gap end -0.07 pts | data/runs/k16_s0/curve.jsonl |
| headline | s1 | k16, served read R=8 | none | gap 0.13 pts [-0.13, 0.39], text 0.03 [-0.03, 0.11], cheat 0.756/0.755, D_end 0.0111, f 0.015, distinct prefixes 0.267, 200 updates | data/reads/raw/k16_s1.json |
| headline | s1 | k16, training curve | none | dev gap end -0.27 pts | data/runs/k16_s1/curve.jsonl |
| D4 open-text check | s0 | k16open, served read R=8 | prefix KL 0.1 on the sum (text kept open) | gap 0.18 pts [-0.11, 0.48], text 0.02 [-0.14, 0.17], cheat 0.645/0.643, D_end 0.0238, f 0.035, distinct prefixes 0.519, 200 updates | data/reads/raw/k16open_s0.json |
| D4 open-text check | s0 | k16open, training curve | prefix KL 0.1 on the sum (text kept open) | dev gap end -1.7 pts | data/runs/k16open_s0/curve.jsonl |

## Skeptic rebuild rows

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline (rebuild) | s0 | k16, served read R=8 | none | Vera own code from npz: gap -0.11 [-0.37, +0.13], text 0.00, f 0.000, D_end 0.00016, distinct 0.004; matches build <= 5e-8 | skeptics/skeptic/k16_s0.vera.json |
| headline (rebuild) | s1 | k16, served read R=8 | none | Vera own code from npz: gap +0.13 [-0.13, +0.40], text +0.03, f 0.015, D_end 0.0111 [0.0095, 0.0129]; matches build <= 7e-8 | skeptics/skeptic/k16_s1.vera.json |
| headline (rebuild) | s0 | k16 + prefix KL (D4), served read R=8 | none | Vera own code from npz: gap +0.18 [-0.13, +0.49], text +0.02, D_end 0.0238 [0.0215, 0.0263], distinct 0.519 (valid); matches build <= 2e-8 | skeptics/skeptic/k16open_s0.vera.json |
| headline (null) | s0, s1, s0 | k16 / k16 / k16open | label swap within each CRN pair (2,000 draws) | null 95% bands [-0.09, +0.09] / [-0.12, +0.12] / [-0.20, +0.20] pts; observed gaps -0.11 / +0.13 / +0.18 (all far below the 3-pt bar) | skeptics/skeptic/k16_s0.vera.json |
| gate (rebuild) | s0 | planted, served read R=8 | positive control (planted text cue, TV 0.30) | Vera own code: gap +17.31 [16.20, 18.41], forced switch +43.3 [42.8, 43.9], D_end 0.414; label-swap null [-1.3, +1.4] | skeptics/skeptic/planted_s0.vera.json |
| start (rebuild) | - | start policy, k=16 | untrained (before any probe training) | Vera own code: gap +0.46 [0.00, 0.94], D 0.0274 [0.0242, 0.0310], f 0.054, f_null 0 | skeptics/skeptic/start.vera.json |
| headline (own engine) | s0, s1 | k16 adapters, Vera's own sampler + hook + LoRA load, 64 test x 2 | independent codebase | k16_s0 D 0.00006, gap +0.03 [-0.44, +0.46]; k16_s1 D 0.0142, gap -0.03 [-0.53, +0.45] | skeptics/skeptic/vera_own_k16_s1.json |
| D before training (own engine) | - | raw Qwen3-0.6B / start adapter, own engine | independent codebase | raw D 0.029 [0.027, 0.031], f 0.059 (200 x 4); start D 0.0266 [0.0227, 0.0307] | skeptics/skeptic/vera_audit_base.json |


