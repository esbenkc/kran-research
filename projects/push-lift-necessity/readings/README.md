# Iris's readings (research role)

Scripts in `code/`, outputs in `../data/readings/`. All read saved models only (no training), CPU fp32 2-layer tail
(`vera_tail.py`, Vera's own-code Qwen3 tail from the Nov 4 room), 200 frozen test prompts, grading-build cache.
Scripts were run from the room folder `swarm-oracle-1105-push-lift-necessity/`; paths inside are absolute to this machine.

| output | script | what | used for |
|---|---|---|---|
| random_push_lift.json | random_push_lift.py | 32 fixed random pushes on Nov 4 models (fp32 head) | Phase 0: Arm R adds ~no pressure |
| kl_sensitivity.json | kl_sensitivity.py | base G on 512 ARC-Easy prompts = 3.5e-5 nats | damage-guard G clause VOID (< 1e-4) |
| gap_vs_dnorm.json | gap_vs_dnorm.py | gap vs weight change, P vs U | reading: faster weights, not bigger gap per weight change |
| pool_lift.json | pool_lift.py | every training-pool push read on later checkpoints; lift, live, sat per push | D3 (i)-(iv) rows; saturation finding |
| reading_rows.json | reading_rows.py | 32 fixed random pushes (bf16 head) + true cue per position, finals + refs | reading rows; D6 input |
| d6_score.json | d6_score.py | D6 search-free score and clean line 0.044 | D6 gate (not needed: no candidate) |

`research.md`: sources, prior work, novelty, and every number above with its date.
