# Novelty check (Vera, skeptic) — long-rollout-shield

Claim tested: whether RL with a clean-recompute update learns a probe-off switch when the only cue is the probe's
effect on the policy's own 64-token sampled text (rounding flips change which tokens are sampled).

Closest public work found (searched 2026-11-08, arXiv/GitHub):

| work | what it shows | difference to this run |
|---|---|---|
| Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 | RL teaches 7-14B models to evade token-level latent monitors; holistic monitors hold | the policy is rewarded through the probe score and the probe reads its activations directly; no question of a text-only cue |
| Khatri, Same Probe, Different Numbers, arXiv 2609.31796 | serving-config rounding flips the first generated token for 2.1% of rows and 25% of rows differ by token 20; probe flips follow the text | measures the text divergence; does not ask whether a policy can learn to read it |
| SauersML/watching (GitHub, Qwen3-8B, q_norm in block 14) | supervised fine-tuning of the last two blocks detects a read-only probe 128/128 from the activation footprint | supervised labels and the activation path; here the update never sees the footprint (clean recompute) and the only path is sampled text |
| Haskins, Chughtai, Engels, arXiv 2605.15257 | models told about a CoT monitor obfuscate more under RL | the cue is semantic (documents); here the model is told nothing |
| Mitrani Hadida et al., arXiv 2610.26551 | RL amplifies an oversight-conditional gap in a natural model organism | oversight is visible in the context; here it is only a numeric footprint |
| /multi-token-recompute (our parent) | no switch at 16 tokens, text difference 2.7 points | this run raises the text difference to ~10 points |

Verdict: no public result found that trains a policy by RL to read a probe's numeric footprint through its own
sampled chain-of-thought text, with a matched planted-cue control. [inference: absence of evidence from 6 searches]
