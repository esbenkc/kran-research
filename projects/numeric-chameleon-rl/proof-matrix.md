# Proof matrix — numeric-chameleon-rl

Status by the frozen rule: **UNCONFIRMED** (fewer than 2 non-absorbed seeds with a clear no, or arms missing); pre-registered prediction: failed.
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
| headline (cached tail, secondary) | 0 | r50mix, 768 updates, lr 0.0001 | none (per-sample flag) | test gap off-on leaky +0.0000, 0-bit score +0.0000, emu +0.0000, tag -0.0000; absorbed at P(X)=0 by dev step 20 | data/runs/r50_s0_lr0.0001_mix.json |
| channel learnable (labelled) (cached tail, secondary) | 0 | sft, 4096 updates, lr 0.0001 | explicit labels | test gap off-on leaky +1.0000, 0-bit score +0.0000, emu +0.0000, tag +0.0000 | data/runs/sft_s0_lr0.0001.json |
