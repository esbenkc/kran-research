# Novelty check (Vera, skeptic) — decision-only-control

Question: does a decision-only policy loss let reward learn a fresh neutral cue at text difference 0.10 while the
probe's rounding trace stays unlearned (Qwen3-0.6B, cached-thought engine, D ladder 0.86 / 0.30 / 0.10)?

## What is already known (searched 2026-11-12, web search, 4 queries)

- Learning time vs the frequency (density) of a feature, supervised settings: time grows as 1/p for a feature seen
  with density p (Basri et al., arXiv 1906.00425: convergence O(k^d / p(x))); frequent features starve rare ones of
  gradient (Tachet des Combes, Pezeshki et al., arXiv 1809.06848, which names "gradient starvation"); superposition models give time ~ 1/p_i
  (arXiv 2602.01045). This is the 'drift' exponent -1 of PREREG v1, shown for supervised regression/classification,
  not for a reward-trained decision token.
- Noise-limited steps grow with the gradient noise scale B_noise / B (McCandlish et al., arXiv 1812.06162); with a
  signal linear in D this gives the 'noise' exponent -2. No paper we found measures it for a sparse text cue under RL.
- RL against monitors: models learn to hide intent from a CoT monitor under RL pressure (Baker et al., arXiv
  2503.11926); models learn to evade latent-space probes under RL (Gupta et al., RL-Obfuscation, arXiv 2506.14261);
  fine-tuned models hide from unseen activation monitors (Neural Chameleons, arXiv 2512.11949); with CoT necessary,
  evasion is hard (Emmons et al., arXiv 2507.05246). None of these measures how fast reward learns a cue as a function
  of how often the cue marks the monitored rows, and none runs a matched-size neutral positive control next to a
  probe-induced text trace.
- Our own posts: matched-trace-powered (trace 0.0% [-3.2, +3.4], planted 6.2% but not above start: control failed);
  reader-bound-shield (dense 83-point cue learned at ~100-150 updates; 10-point cue flat); cue-size-law (exponent
  -0.90 on an activation-size axis, a different unit).

## Verdict

NEW, with two named limits. (1) The 'learning time ~ 1/frequency' law itself is not new (Basri et al. 1906.00425;
gradient starvation 1809.06848; 2602.01045). Tonight's two-point slope -0.83 [-0.96, -0.75] is the first measurement
we found of that exponent for a reward-trained decision token on a sparse text cue, with the onset forecast frozen
before the 0.10 run (drift 645-860, noise 5,546-7,395; observed 600). It is a two-point slope at n = 1 per
strength, so the post must call it a slope between two runs, not a law. (2) The positive control is new relative to
our own matched-trace-powered post (control failed there: 6.2%, not above start; here 86.5%, +86.0 above start),
but it holds only for the changed recipe (no std division, penalty 0.42) and fixed thoughts. The 'readability, not
size' claim of the goal's if_right is NOT shown (X sits at token 0, the trace is spread); the post must not make it.
