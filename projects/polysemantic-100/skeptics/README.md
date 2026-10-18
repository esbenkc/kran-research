# Skeptic checks — polysemantic-100

Independent checks run by the swarm's research and critic roles (room polysemantic-100, 2026-10-17/18).

## recount.py (Iris, research)

Recounts every probe pass, pole label and neuron label from the raw files (`data/probe_acts.json`,
`data/hypotheses.json`, `data/contexts/`). It does not import `score.py`. It also gives the Wilson 95%
intervals used in the post.

    cd projects/polysemantic-100/skeptics && python3 recount.py

Result: 0 differences from `data/labels.json` under both reference rules. Neurons: UNRELATED 1
(95% Wilson 0.2-5.4%), RELATED 0, ONE_JOB 11, NO_PASS 83, TOO_SPARSE 5.

## Critic verdicts (Kasper)

- Record pass: `acts.f16`/`gate.f16` read as [1590, 256, 100], 0 NaN/inf, 0 all-zero windows, resume seam
  at window 344 has no gap. TOO_SPARSE recount 15 + 18 = 33 of 200 poles.
- First pipeline review REJECTED the ablation: negative KL values (low-precision log-softmax). Fixed in
  a2debf4 (float32, same-graph clean pass, noise floor 0.0 exactly).
- Second pipeline review PASSED: all KL >= 0; L0N1275 fails both the activation-matched and the random
  baseline for all 3 jobs; control-text KL has the same size as fire-text KL.
