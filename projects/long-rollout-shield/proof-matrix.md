# Proof matrix — long-rollout-shield

Status: **UNCONFIRMED** (planted gate False); scout prereg: failed. Built by summarize.py from data/reads/raw/*.json; skeptics add their rebuild rows below.

| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| text difference before training | - | start policy, real cue, R=8 | none | D 0.1012 [0.0952, 0.1074], gap 0.34 pts [-0.58, 1.23], cheat off/on 0.812/0.809, f 0.384, distinct 1.000 | data/reads/raw/start_real_test.json |
| text difference before training | - | start policy, planted cue, R=4 | planted cue (sampler only, none build) | D 0.0999 [0.0979, 0.1019], gap 0.48 pts [-0.35, 1.24], cheat off/on 0.811/0.806, f 0.107, distinct 1.000 | data/reads/raw/start_planted_test.json |
| text difference before training | - | start policy, hash cue, R=4 | hash cue (sampler only, none build) | D 0.0875 [0.0797, 0.0957], gap 0.5 pts [-0.83, 1.79], cheat off/on 0.811/0.806, f 0.351, distinct 1.000 | data/reads/raw/start_hash_test.json |
| planted gate | s0 | planted, served read R=8 | positive control (one-token cue, matched D) | gap 0.1 pts [-0.53, 0.76], text 0.1 [-0.53, 0.76], cheat off/on 0.855/0.854, D_end 0.2108 [0.2064, 0.2152], ceiling (range_pred 94.7) 19.96 pts, distinct 1.000, 100 updates, forced switch (range_obs) 0.31 pts [-1.22, 1.74] | data/reads/raw/planted_s0.json |
| planted gate | s0 | planted, training curve (dev, monitor) | positive control (one-token cue, matched D) | dev gap end -2.39 pts, dev D end 0.21746575042174746, unique 64/64 | data/runs/planted_s0/curve.jsonl |
| headline | s0 | real, served read R=8 | none | gap 0.23 pts [-0.6, 1.1], text 0.18 [-0.65, 1.05], cheat off/on 0.836/0.834, D_end 0.1054 [0.0995, 0.1112], ceiling (range_pred 94.7) 9.98 pts, distinct 0.999, 125 updates | data/reads/raw/real_s0.json |
| headline | s0 | real, training curve (dev, monitor) | none | dev gap end -5.03 pts, dev D end None, unique 64/64 | data/runs/real_s0/curve.jsonl |
| reader edge (C_sup) | - | start text, real (csup_real) | none | C_sup -0.02 pts [-0.28, 0.24] | research-iris/csup_real.json |
| reader edge (C_sup) | - | start text, planted (csup_planted) | none | C_sup 10.74 pts [8.5, 13.12] | research-iris/csup_planted.json |

## Skeptic rebuild rows

| headline (rebuild) | s0 | real, served read R=8 | none | Vera own code from npz: gap +0.23 [-0.61, +1.10], text +0.18, spill +0.05, D_end 0.1054 [0.0997, 0.1111], distinct 0.999; matches build <= 1e-7 | skeptics/skeptic/raw__real_s0.vera.json |
| headline (null) | s0 | real, served read R=8 | label swap inside each CRN pair (2,000) / prompt sign flip (2,000) | null bands [-0.86, +0.88] / [-0.81, +0.82]; observed +0.23, p 0.59 | skeptics/skeptic/raw__real_s0.vera.json |
| planted gate (rebuild) | s0 | planted, served read R=8 | positive control (one-token cue) | Vera own code: gap +0.10 [-0.54, +0.76], D_end 0.2108, forced +0.31 [-1.16, +1.73]; pair-swap null [-0.66, +0.65] | skeptics/skeptic/raw__planted_s0.vera.json |
| start (rebuild) | - | start policy, real, R=8 | untrained | Vera own code: gap +0.34 [-0.58, +1.24], D 0.1012 [0.0954, 0.1072], f 0.384 | skeptics/skeptic/raw__start_real_test.vera.json |
| status (rebuild) | s0 | frozen PREREG v1.3 rule | none | Vera vera_status.py: UNCONFIRMED; prereg outcome failed (planted >= 5 F, D_end < 0.07 F, convergence F) | skeptics/skeptic/vera_status.json |
