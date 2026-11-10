# Novelty check (Vera, skeptic) — cue-size-law

Claim checked: in a reward-trained policy (GRPO, LoRA, AdamW) the number of updates until a cue-conditional switch
appears scales as a power of the cue's size, measured against a slope fixed before the run (-2).

Searched 2026-11-10 (web search, arXiv abs pages): "policy gradient time to learn feature scales inverse square of
signal", "Adam vs SGD learning time feature magnitude sign descent exponent", "RL fine-tuning language model time to
learn conditional behaviour trigger strength", "training-inference mismatch RL exploit bitwise".

| Work | What it shows | Overlap with this post |
|---|---|---|
| Saxe, McClelland, Ganguli, arXiv 1312.6120 (ICLR 2014) | In deep linear nets each input-output mode is learned on a time scale inversely proportional to its singular value (plateau then sigmoid jump). | Theory of the two-factor (LoRA = B A) plateau-then-jump and a -1 exponent in signal strength; gradient flow, supervised, no RL, no optimizer normalisation. |
| Ren, Nichani, Wu, Lee, arXiv 2504.19983 | Sharp per-direction transition times in SGD learning of shallow nets; signal strength sets each direction's time. | Same idea (transition time set by a direction's strength), supervised, SGD only. |
| Atanasov, Meterez, Simon, Pehlevan, arXiv 2410.04642 (ICLR 2025) | Rich-regime nets show a long plateau then a drop; the optimal learning rate scales as a power of the feature-learning strength; per its appendix F (seen via a search snippet, not read in full), SignSGD/Adam leaves the init saddle in O(1) time. | Supports plateau-then-jump and optimizer-dependent exponents; supervised, no RL, no cue. |
| Flynn, Goldhirsh, Keating, Seroussi, arXiv 2605.22481 | In regularised linear models backdoor success peaks at finite trigger strength; the worst direction is the minimum-variance direction of the data. | Static (end-of-training) dependence on trigger size and direction; no learning-time law. |
| Rando and Tramèr, arXiv 2311.14455; BadGPT; GREAT arXiv 2510.09260 | RLHF backdoors from poisoned feedback; poisoning rate and epochs set success. | Vary data fraction, not the size of an internal cue, and report end success, not time to switch. |
| vLLM + TorchTitan bitwise-consistent RL (vllm.ai blog 2025-11-10); arXiv 2605.14220 | Train-inference numeric mismatch makes RL off-policy and unstable. | Motivation (numeric leaks exist); no measurement of how fast a policy learns to condition on a leak of a given size. |

Verdict: NEW as a measurement. I found no paper that times RL learning of a cue-conditional switch against the size of a
planted internal cue with a pre-registered exponent. The underlying theory pieces (1/s mode times, optimizer-dependent
exponents) are old and the post must credit Saxe et al. 2014 for the plateau-then-jump and the 1/strength scaling.
