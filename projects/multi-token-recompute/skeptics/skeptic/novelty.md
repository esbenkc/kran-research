# Novelty check (Vera, skeptic) — oracle-1108-multi-token

Question: does a clean-recompute update learn a probe-off switch when a probe's numeric footprint can change
sampled prefix tokens (16 tokens)? Claimed contribution: a pre-training audit number (the exact text TV D between
probe-on and probe-off sampler distributions) bounds the text-only switch, and f (coupled flip rate) overstates it.

Searched (2026-11-07, web search + arXiv abstract pages):

| work | what it does | overlap with this post |
|---|---|---|
| Price, Tian, Xun, Zhu, "Total Variation Distance Estimation in Autoregressive Models", arXiv 2607.19510 (Jul 2026) | Estimates TV between two length-n autoregressive distributions; logit access needs O(n/eps^2) queries (tight); measures SGLang vs vLLM serving the same weights | SAME estimator family as our D (logit-access plug-in on sampled sequences) and the same motivation (two engines, same weights). Must be CREDITED for the TV estimator. They do not connect TV to what RL can learn or to probes. |
| Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 | RL teaches LMs to evade latent monitors using reward only | Background: evasion through policy changes; not the sampler-numerics channel |
| Taufeeque et al., Obfuscation Atlas, arXiv 2602.15515 | Obfuscated activations vs obfuscated policies under RLVR with deception probes | Background; no numerics channel |
| LessWrong, "The Most Forbidden Technique is not always forbidden" (2026) | Argues policy gradient gives no direct pressure on activations; obfuscation routes through the text distribution | Closest conceptual claim; our bound is the quantitative version for the case where the probe's own numeric footprint enters the text |
| Yuan et al. arXiv 2506.09501; He / Thinking Machines 2025 (batch invariance) | bf16 non-determinism, batch-shape dependence | Mechanism of the footprint; our f_null check |
| Baker et al. arXiv 2503.11926 | CoT monitors under optimisation pressure | Background |

Verdict: no prior work found that (a) measures a probe hook's numeric footprint in the sampler's text distribution
or (b) bounds a recompute-trained monitor-detection switch by that TV. The TV inequality itself (|E_P g - E_Q g| <= TV
for g in [0,1]) is textbook (Levin-Peres-Wilmer Prop 4.5) and must not be presented as new; the estimator is
2607.19510's setting. New = applying both to the probe-footprint channel, measuring D before training, and testing
it against RL. Search is not proof.
