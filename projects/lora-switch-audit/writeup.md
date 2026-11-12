# lora-switch-audit

**Question:** Does the top input-side singular direction of a reward-trained LoRA delta rank the true cue first among
1,000 random cue directions, in at least 5 of 6 trained switch models, with no behaviour test?
**Sources:** organism, cache and parent adapters from kran-research `swarm/theory-1110-cue-size-learning`,
`projects/cue-size-law` (Qwen/Qwen3-0.6B, public weights).
**Method:** reward (group-relative policy gradient) trains LoRA r16 on blocks 26-27 to drop a flattery opener only when
a planted random cue sits on the first three chat tokens. We save the adapter at fixed update counts and score the
true cue against 1,000 Gaussian cues built the same way, on the top input singular vector of block 26's stacked
q, k, v change (frozen primary), and on the change's whole reaction to the cue (frozen secondary). Rules: `PREREG.md`.

## Findings (data/summary.json, proof-matrix.md)

- At 256 updates the top direction ranked the true cue first in 2 of 6 largest-cue models (C1: REJECTED); at the
  switch in 0 of 6. The whole change's reaction ranked it first in 6 of 6.
- Trained on to 4,352 updates, the same largest-cue models put the cue first from 512 (seed 0) and 704 (seed 2)
  updates: the miss is early, not a size effect.
- Controls (no cue, shuffled reward): true cue rank 179-656 in all 6 primary reads (C2 holds).
- A cue 4x below the smallest learned one, 20,480 updates, never learned: top direction rank 587 (C3 fails);
  reaction rank 7 vs 355 in the same-seed no-cue run (one run).
- Real build leak (parent end-of-run): seed 0 hidden (rank 439), seed 1 rank 2.

## Blog post

`blog.kran.ai/_notes/lora-switch-audit.md`, main figure `assets/figures/lora-switch-audit/`.
