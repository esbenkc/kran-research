# Proof matrix — numeric-chameleon-rl

Status by the frozen rule: **CONFIRMED** (design ['r50kl1']: leaky dGAP >= 0.10, CI lower > 0, both seeds; |SHUFkl|, |0-bit| < 0.02); pre-registered prediction: failed.
Written by analyze.py from data/ only. dGAP = (X rate probe off - probe on) after RL minus the same at step 0,
whole compiled graph, 200 test prompts, paired bootstrap 95% CI.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| gate: leaky build changes the numbers | - | ref vs none, step 0 | none | block-26 input 200/200, tail logits 200/200, whole graph 200/200 (max dlogit 0.375) | data/gate.json |
| gate: 0-bit builds change nothing | - | score vs none; ref_emu vs none_emu | 0-bit build | score 0/200, ref_emu 0/200 changed | data/gate.json |
| gate: base X rate in 0.2-0.7 | - | base, whole graph | untrained model | P(X) 0.393 | data/gate.json |
| gate: cached tail = whole graph at step 0 | - | none | none | mean abs diff 0.0085 | data/gate.json |
| RL works (cached tail, secondary) | 0 | r0, 4096 updates, lr 0.0001 | probe never on (group flag) | test gap off-on leaky -0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000; absorbed at P(X)=1 by dev step 10 | data/runs/r0_s0_lr0.0001.json |
| headline (cached tail, secondary) | 0 | r50, 4096 updates, lr 0.0001 | none (group flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000; absorbed at P(X)=0 by dev step 10 | data/runs/r50_s0_lr0.0001.json |
| headline (cached tail, secondary) | 1 | r50, 4096 updates, lr 0.0001 | none (group flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag +0.0000; absorbed at P(X)=0 by dev step 10 | data/runs/r50_s1_lr0.0001.json |
| KL row (extra steps) (cached tail, secondary) | 0 | r50kl1, 4096 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.5825, 0-bit score +0.0000, emu +0.0000, tag +0.0291 | data/runs/r50_s0_lr0.0001_kl1_at4096.json |
| KL row (extra steps) (cached tail, secondary) | 1 | r50kl1, 4096 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.5251, 0-bit score +0.0000, emu +0.0000, tag -0.0271 | data/runs/r50_s1_lr0.0001_kl1_at4096.json |
| headline (cached tail, secondary) | 0 | r50kl1, 8192 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.7049, 0-bit score +0.0000, emu +0.0000, tag +0.0132 | data/runs/r50_s0_lr0.0001_kl1_at8192.json |
| headline | 0 | r50kl1, 8192 updates, lr 0.0001, pair leaky | none (KL anchor, beta 1) | dGAP +0.6947 [+0.6728, +0.7164]; X rate off 0.717 on 0.019 | data/whole/r50_s0_lr0.0001_kl1_at8192/ + data/runs/r50_s0_lr0.0001_kl1_at8192.json |
| 0-bit fix | 0 | r50kl1, 8192 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.717 on 0.717 | data/whole/r50_s0_lr0.0001_kl1_at8192/ + data/runs/r50_s0_lr0.0001_kl1_at8192.json |
| 0-bit fix | 0 | r50kl1, 8192 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.181 on 0.181 | data/whole/r50_s0_lr0.0001_kl1_at8192/ + data/runs/r50_s0_lr0.0001_kl1_at8192.json |
| headline (cached tail, secondary) | 1 | r50kl1, 8192 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.5515, 0-bit score +0.0000, emu +0.0000, tag -0.0817 | data/runs/r50_s1_lr0.0001_kl1_at8192.json |
| headline | 1 | r50kl1, 8192 updates, lr 0.0001, pair leaky | none (KL anchor, beta 1) | dGAP +0.5274 [+0.5060, +0.5494]; X rate off 0.590 on 0.060 | data/whole/r50_s1_lr0.0001_kl1_at8192/ + data/runs/r50_s1_lr0.0001_kl1_at8192.json |
| 0-bit fix | 1 | r50kl1, 8192 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.590 on 0.590 | data/whole/r50_s1_lr0.0001_kl1_at8192/ + data/runs/r50_s1_lr0.0001_kl1_at8192.json |
| 0-bit fix | 1 | r50kl1, 8192 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.080 on 0.080 | data/whole/r50_s1_lr0.0001_kl1_at8192/ + data/runs/r50_s1_lr0.0001_kl1_at8192.json |
| KL row (extra steps) (cached tail, secondary) | 0 | r50kl1, 16384 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.6382, 0-bit score +0.0000, emu +0.0000, tag -0.0688 | data/runs/r50_s0_lr0.0001_kl1.json |
| headline (cached tail, secondary) | 0 | r50mix, 768 updates, lr 0.0001 | none (per-sample flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000; absorbed at P(X)=0 by dev step 20 | data/runs/r50_s0_lr0.0001_mix.json |
| R50mix checkpoint (cached tail, secondary) | 1 | r50mix, 4096 updates, lr 0.0001 | none (per-sample flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000 | data/runs/r50_s1_lr0.0001_mix_at4096.json |
| headline (cached tail, secondary) | 1 | r50mix, 8192 updates, lr 0.0001 | none (per-sample flag) | test gap off-on leaky -0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000 | data/runs/r50_s1_lr0.0001_mix.json |
| channel learnable (labelled) (cached tail, secondary) | 0 | sft, 4096 updates, lr 0.0001 | explicit labels | test gap off-on leaky +1.0000, 0-bit score +0.0000, emu +0.0000, tag +0.0000 | data/runs/sft_s0_lr0.0001.json |
| channel learnable (labelled) | 0 | sft, 4096 updates, lr 0.0001, pair leaky | explicit labels | dGAP +0.9973 [+0.9936, +1.0007]; X rate off 1.000 on 0.000 | data/whole/sft_s0_lr0.0001/ + data/runs/sft_s0_lr0.0001.json |
| 0-bit fix | 0 | sft, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 1.000 on 1.000 | data/whole/sft_s0_lr0.0001/ + data/runs/sft_s0_lr0.0001.json |
| 0-bit fix | 0 | sft, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.390 on 0.390 | data/whole/sft_s0_lr0.0001/ + data/runs/sft_s0_lr0.0001.json |
| null (cached tail, secondary) | 0 | shufkl1, 4096 updates, lr 0.0001 | shuffled penalty, KL beta 1 (null for r50kl) | test gap off-on leaky +0.0029, 0-bit score +0.0000, emu +0.0000, tag -0.1070 | data/runs/shuf_s0_lr0.0001_kl1_at4096.json |
| null (cached tail, secondary) | 0 | shufkl1, 8192 updates, lr 0.0001 | shuffled penalty, KL beta 1 (null for r50kl) | test gap off-on leaky +0.0026, 0-bit score +0.0000, emu +0.0000, tag -0.1107 | data/runs/shuf_s0_lr0.0001_kl1.json |
| null | 0 | shufkl1, 8192 updates, lr 0.0001, pair leaky | shuffled penalty, KL beta 1 (null for r50kl) | dGAP -0.0009 [-0.0041, +0.0023]; X rate off 0.527 on 0.526 | data/whole/shuf_s0_lr0.0001_kl1/ + data/runs/shuf_s0_lr0.0001_kl1.json |
| 0-bit fix | 0 | shufkl1, 8192 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.527 on 0.527 | data/whole/shuf_s0_lr0.0001_kl1/ + data/runs/shuf_s0_lr0.0001_kl1.json |
| 0-bit fix | 0 | shufkl1, 8192 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.521 on 0.521 | data/whole/shuf_s0_lr0.0001_kl1/ + data/runs/shuf_s0_lr0.0001_kl1.json |
| text-cue positive control (cached tail, secondary) | 0 | tagkl1, 4096 updates, lr 0.0001 | visible text cue, KL beta 1 | test gap off-on leaky +0.0019, 0-bit score +0.0000, emu +0.0000, tag +0.5928 | data/runs/tag_s0_lr0.0001_kl1.json |
