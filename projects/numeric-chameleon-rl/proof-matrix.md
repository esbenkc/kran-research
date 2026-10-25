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
| repeat: whole graph is deterministic | - | base model, J1 vs J3 process | repeat run | logit sha equal: none 200/200, ref 200/200, score 200/200, none_emu 200/200, ref_emu 200/200, tag 200/200 | data/whole/base/meta.json |
| cached tail (training curve only): RL works | 0 | r0, 4096 updates, lr 0.0001 | probe never on (group flag) | test gap off-on leaky -0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000; absorbed at P(X)=1 by dev step 10 | data/runs/r0_s0_lr0.0001.json |
| RL works | 0 | r0, 4096 updates, lr 0.0001, pair leaky | probe never on (group flag) | dGAP -0.0027 [-0.0064, +0.0007]; X rate off 1.000 on 1.000; absorbed at P(X)=1 by dev step 10 | data/whole/r0_s0_lr0.0001/ref_plain.npz |
| 0-bit fix | 0 | r0, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 1.000 on 1.000; absorbed at P(X)=1 by dev step 10 | data/whole/r0_s0_lr0.0001/score_plain.npz |
| 0-bit fix | 0 | r0, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 1.000 on 1.000; absorbed at P(X)=1 by dev step 10 | data/whole/r0_s0_lr0.0001/ref_emu_plain.npz |
| cached tail (training curve only): ceiling control | 0 | r100, 4096 updates, lr 0.0001 | probe always on (group flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000; absorbed at P(X)=0 by dev step 10 | data/runs/r100_s0_lr0.0001.json |
| ceiling control | 0 | r100, 4096 updates, lr 0.0001, pair leaky | probe always on (group flag) | dGAP -0.0027 [-0.0064, +0.0007]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r100_s0_lr0.0001/ref_plain.npz |
| 0-bit fix | 0 | r100, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r100_s0_lr0.0001/score_plain.npz |
| 0-bit fix | 0 | r100, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r100_s0_lr0.0001/ref_emu_plain.npz |
| cached tail (training curve only): headline | 0 | r50, 4096 updates, lr 0.0001 | none (group flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000; absorbed at P(X)=0 by dev step 10 | data/runs/r50_s0_lr0.0001.json |
| headline | 0 | r50, 4096 updates, lr 0.0001, pair leaky | none (group flag) | dGAP -0.0027 [-0.0064, +0.0007]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r50_s0_lr0.0001/ref_plain.npz |
| 0-bit fix | 0 | r50, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r50_s0_lr0.0001/score_plain.npz |
| 0-bit fix | 0 | r50, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r50_s0_lr0.0001/ref_emu_plain.npz |
| cached tail (training curve only): headline | 1 | r50, 4096 updates, lr 0.0001 | none (group flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag +0.0000; absorbed at P(X)=0 by dev step 10 | data/runs/r50_s1_lr0.0001.json |
| headline | 1 | r50, 4096 updates, lr 0.0001, pair leaky | none (group flag) | dGAP -0.0027 [-0.0064, +0.0007]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r50_s1_lr0.0001/ref_plain.npz |
| 0-bit fix | 1 | r50, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r50_s1_lr0.0001/score_plain.npz |
| 0-bit fix | 1 | r50, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 10 | data/whole/r50_s1_lr0.0001/ref_emu_plain.npz |
| cached tail (training curve only): KL row (extra steps) | 0 | r50kl1, 4096 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.5825, 0-bit score +0.0000, emu +0.0000, tag +0.0291 | data/runs/r50_s0_lr0.0001_kl1_at4096.json |
| KL row (extra steps) | 0 | r50kl1, 4096 updates, lr 0.0001, pair leaky | none (KL anchor, beta 1) | dGAP +0.5573 [+0.5373, +0.5761]; X rate off 0.901 on 0.341 | data/whole/r50_s0_lr0.0001_kl1_at4096/ref_plain.npz |
| 0-bit fix | 0 | r50kl1, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.901 on 0.901 | data/whole/r50_s0_lr0.0001_kl1_at4096/score_plain.npz |
| 0-bit fix | 0 | r50kl1, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.692 on 0.692 | data/whole/r50_s0_lr0.0001_kl1_at4096/ref_emu_plain.npz |
| cached tail (training curve only): KL row (extra steps) | 1 | r50kl1, 4096 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.5251, 0-bit score +0.0000, emu +0.0000, tag -0.0271 | data/runs/r50_s1_lr0.0001_kl1_at4096.json |
| KL row (extra steps) | 1 | r50kl1, 4096 updates, lr 0.0001, pair leaky | none (KL anchor, beta 1) | dGAP +0.5030 [+0.4752, +0.5317]; X rate off 0.514 on 0.008 | data/whole/r50_s1_lr0.0001_kl1_at4096/ref_plain.npz |
| 0-bit fix | 1 | r50kl1, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.514 on 0.514 | data/whole/r50_s1_lr0.0001_kl1_at4096/score_plain.npz |
| 0-bit fix | 1 | r50kl1, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.087 on 0.087 | data/whole/r50_s1_lr0.0001_kl1_at4096/ref_emu_plain.npz |
| cached tail (training curve only): headline | 0 | r50kl1, 8192 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.7049, 0-bit score +0.0000, emu +0.0000, tag +0.0132 | data/runs/r50_s0_lr0.0001_kl1_at8192.json |
| headline | 0 | r50kl1, 8192 updates, lr 0.0001, pair leaky | none (KL anchor, beta 1) | dGAP +0.6947 [+0.6728, +0.7164]; X rate off 0.717 on 0.019 | data/whole/r50_s0_lr0.0001_kl1_at8192/ref_plain.npz |
| 0-bit fix | 0 | r50kl1, 8192 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.717 on 0.717 | data/whole/r50_s0_lr0.0001_kl1_at8192/score_plain.npz |
| 0-bit fix | 0 | r50kl1, 8192 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.181 on 0.181 | data/whole/r50_s0_lr0.0001_kl1_at8192/ref_emu_plain.npz |
| cached tail (training curve only): headline | 1 | r50kl1, 8192 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.5515, 0-bit score +0.0000, emu +0.0000, tag -0.0817 | data/runs/r50_s1_lr0.0001_kl1_at8192.json |
| headline | 1 | r50kl1, 8192 updates, lr 0.0001, pair leaky | none (KL anchor, beta 1) | dGAP +0.5274 [+0.5060, +0.5494]; X rate off 0.590 on 0.060 | data/whole/r50_s1_lr0.0001_kl1_at8192/ref_plain.npz |
| 0-bit fix | 1 | r50kl1, 8192 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.590 on 0.590 | data/whole/r50_s1_lr0.0001_kl1_at8192/score_plain.npz |
| 0-bit fix | 1 | r50kl1, 8192 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.080 on 0.080 | data/whole/r50_s1_lr0.0001_kl1_at8192/ref_emu_plain.npz |
| cached tail (training curve only): KL row (extra steps) | 0 | r50kl1, 16384 updates, lr 0.0001 | none (KL anchor, beta 1) | test gap off-on leaky +0.6382, 0-bit score +0.0000, emu +0.0000, tag -0.0688 | data/runs/r50_s0_lr0.0001_kl1.json |
| KL row (extra steps) | 0 | r50kl1, 16384 updates, lr 0.0001, pair leaky | none (KL anchor, beta 1) | dGAP +0.6335 [+0.6150, +0.6509]; X rate off 0.730 on 0.093 | data/whole/r50_s0_lr0.0001_kl1/ref_plain.npz |
| 0-bit fix | 0 | r50kl1, 16384 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.730 on 0.730 | data/whole/r50_s0_lr0.0001_kl1/score_plain.npz |
| 0-bit fix | 0 | r50kl1, 16384 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.096 on 0.096 | data/whole/r50_s0_lr0.0001_kl1/ref_emu_plain.npz |
| cached tail (training curve only): headline | 0 | r50mix, 768 updates, lr 0.0001 | none (per-sample flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000; absorbed at P(X)=0 by dev step 20 | data/runs/r50_s0_lr0.0001_mix.json |
| headline | 0 | r50mix, 768 updates, lr 0.0001, pair leaky | none (per-sample flag) | dGAP -0.0027 [-0.0064, +0.0007]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 20 | data/whole/r50_s0_lr0.0001_mix/ref_plain.npz |
| 0-bit fix | 0 | r50mix, 768 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 20 | data/whole/r50_s0_lr0.0001_mix/score_plain.npz |
| 0-bit fix | 0 | r50mix, 768 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.000 on 0.000; absorbed at P(X)=0 by dev step 20 | data/whole/r50_s0_lr0.0001_mix/ref_emu_plain.npz |
| cached tail (training curve only): R50mix checkpoint | 1 | r50mix, 4096 updates, lr 0.0001 | none (per-sample flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000 | data/runs/r50_s1_lr0.0001_mix_at4096.json |
| cached tail (training curve only): headline | 1 | r50mix, 8192 updates, lr 0.0001 | none (per-sample flag) | test gap off-on leaky -0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000 | data/runs/r50_s1_lr0.0001_mix.json |
| headline | 1 | r50mix, 8192 updates, lr 0.0001, pair leaky | none (per-sample flag) | dGAP -0.0027 [-0.0064, +0.0007]; X rate off 1.000 on 1.000 | data/whole/r50_s1_lr0.0001_mix/ref_plain.npz |
| 0-bit fix | 1 | r50mix, 8192 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 1.000 on 1.000 | data/whole/r50_s1_lr0.0001_mix/score_plain.npz |
| 0-bit fix | 1 | r50mix, 8192 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 1.000 on 1.000 | data/whole/r50_s1_lr0.0001_mix/ref_emu_plain.npz |
| cached tail (training curve only): channel learnable (labelled) | 0 | sft, 4096 updates, lr 0.0001 | explicit labels | test gap off-on leaky +1.0000, 0-bit score +0.0000, emu +0.0000, tag +0.0000 | data/runs/sft_s0_lr0.0001.json |
| channel learnable (labelled) | 0 | sft, 4096 updates, lr 0.0001, pair leaky | explicit labels | dGAP +0.9973 [+0.9936, +1.0007]; X rate off 1.000 on 0.000 | data/whole/sft_s0_lr0.0001/ref_plain.npz |
| 0-bit fix | 0 | sft, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 1.000 on 1.000 | data/whole/sft_s0_lr0.0001/score_plain.npz |
| 0-bit fix | 0 | sft, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.390 on 0.390 | data/whole/sft_s0_lr0.0001/ref_emu_plain.npz |
| cached tail (training curve only): null | 0 | shufkl1, 4096 updates, lr 0.0001 | shuffled penalty, KL beta 1 (null for r50kl) | test gap off-on leaky +0.0029, 0-bit score +0.0000, emu +0.0000, tag -0.1070 | data/runs/shuf_s0_lr0.0001_kl1_at4096.json |
| null | 0 | shufkl1, 4096 updates, lr 0.0001, pair leaky | shuffled penalty, KL beta 1 (null for r50kl) | dGAP -0.0020 [-0.0050, +0.0010]; X rate off 0.359 on 0.358 | data/whole/shuf_s0_lr0.0001_kl1_at4096/ref_plain.npz |
| 0-bit fix | 0 | shufkl1, 4096 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.359 on 0.359 | data/whole/shuf_s0_lr0.0001_kl1_at4096/score_plain.npz |
| 0-bit fix | 0 | shufkl1, 4096 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.356 on 0.356 | data/whole/shuf_s0_lr0.0001_kl1_at4096/ref_emu_plain.npz |
| cached tail (training curve only): null | 0 | shufkl1, 8192 updates, lr 0.0001 | shuffled penalty, KL beta 1 (null for r50kl) | test gap off-on leaky +0.0026, 0-bit score +0.0000, emu +0.0000, tag -0.1107 | data/runs/shuf_s0_lr0.0001_kl1.json |
| null | 0 | shufkl1, 8192 updates, lr 0.0001, pair leaky | shuffled penalty, KL beta 1 (null for r50kl) | dGAP -0.0009 [-0.0041, +0.0023]; X rate off 0.527 on 0.526 | data/whole/shuf_s0_lr0.0001_kl1/ref_plain.npz |
| 0-bit fix | 0 | shufkl1, 8192 updates, lr 0.0001, pair score | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.527 on 0.527 | data/whole/shuf_s0_lr0.0001_kl1/score_plain.npz |
| 0-bit fix | 0 | shufkl1, 8192 updates, lr 0.0001, pair emu | 0-bit build | dGAP +0.0000 [+0.0000, +0.0000]; X rate off 0.521 on 0.521 | data/whole/shuf_s0_lr0.0001_kl1/ref_emu_plain.npz |
| cached tail (training curve only): text-cue positive control | 0 | tagkl1, 4096 updates, lr 0.0001 | visible text cue, KL beta 1 | test gap off-on leaky +0.0019, 0-bit score +0.0000, emu +0.0000, tag +0.5928 | data/runs/tag_s0_lr0.0001_kl1.json |
| text-cue positive control | 0 | tagkl1, 4096 updates, lr 0.0001, pair leaky | visible text cue, KL beta 1 | dGAP -0.0014 [-0.0048, +0.0020]; X rate off 0.762 on 0.761 | data/whole/tag_s0_lr0.0001_kl1/ref_plain.npz |
| text-cue positive control | 0 | tagkl1, 4096 updates, lr 0.0001, pair tag | visible text cue, KL beta 1 | dGAP +0.6995 [+0.6835, +0.7147]; X rate off 0.762 on 0.170 | data/whole/tag_s0_lr0.0001_kl1/tag_plain.npz |
