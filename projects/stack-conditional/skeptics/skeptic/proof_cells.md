| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| headline (Vera rebuild) | 0 | main 8192 | none | own whole graph: dGAP +0.437 [+0.420, +0.453], 200/200 prompts >0, grading 0.1208, serving 0.5636 | skeptics/skeptic/vera_dgap_v2.out |
| headline (Vera rebuild) | 1 | main 8192 | none | own whole graph: dGAP +0.217 [+0.204, +0.231], 200/200 prompts >0, grading 0.1238, serving 0.3470 | skeptics/skeptic/vera_dgap_v2.out |
| gap holds (Vera rebuild) | 0,1 | main 7168 | none | dGAP +0.315 / +0.295 | skeptics/skeptic/vera_dgap_v2.out |
| control (Vera rebuild) | 0 | shuf 8192 | shuffled penalty | dGAP -0.0027 [-0.006, +0.000], sign-flip p 0.088 | skeptics/skeptic/vera_dgap_v2.out |
| control (Vera rebuild) | 0 | penall 8192 | always-penalised | dGAP -0.0040 [-0.008, -0.000] | skeptics/skeptic/vera_dgap_v2.out |
| carried share emu (Vera rebuild) | 0,1 | main 8192 | none | T_obs -0.000 / -0.008 vs frozen -0.001 | skeptics/skeptic/vera_dgap.json |
| carried share mlx (Vera rebuild, own mlx-lm) | 0,1 | main 8192 | none | T_obs 0.545 [0.529, 0.562] / 0.277 [0.261, 0.294] vs frozen 0.054 | skeptics/skeptic/vera_mlx_share.out |
| prereg clause 3 (Vera rebuild) | 0,1 | main 8192 | none | serving first-token opener rate 0.625 / 0.250 vs untrained 0.315 | skeptics/skeptic/vera_greedy.py |
| prefix mechanism (Vera rebuild) | 0,1 | main 8192 | none | grading input + serving positions 0-2: share 0.969 / 0.915; reverse 0.021 / 0.046 | skeptics/skeptic/vera_patch_s0.json |
| per-prompt agreement | 0,1 | main 7168, 8192 | none | own P(X) = build rows to <= 1.6e-6 | skeptics/skeptic/vera_buildcmp.py |
| gate: cache faithful | - | 3 builds | none | build cache = own trunk states, 646/646 prompts bit-identical | skeptics/skeptic/vera_cachecmp.py |
