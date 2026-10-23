# Novelty notes — skeptic2

Question checked: whether prior public work already reports that gradient training attacks a pre-norm last-layer probe by using the zero-cost RMSNorm scale symmetry.

Searches run:
- `"2502.03407" residual stream probe standardize activations` -> Goldowsky-Dill et al. 2025, strategic deception probes. Abstract reports linear probes with AUROC 0.96-0.999 and 95-99% catch at 1% FPR. It does not mention pre-norm vs post-norm probe placement, RMSNorm scale invariance, or a norm-rescale evasion mechanism. Source: https://arxiv.org/abs/2502.03407
- `"2412.09565" obfuscated activations probes` -> Bailey et al. 2025, Obfuscated Activations Bypass LLM Latent-Space Defenses. The paper reports obfuscation attacks against latent monitors, including probes, with recall sometimes falling from 100% to 0% while retaining behavior. It frames activation malleability broadly. I found no claim about RMSNorm radial scale symmetry or gradient training using the final-norm null direction. Source: https://huggingface.co/papers/2412.09565 / https://arxiv.org/abs/2412.09565
- `"2506.14261" RL obfuscation activations probes` -> Gupta and Jenner 2025, RL-Obfuscation. Abstract reports fine-tuned LMs can evade token-level latent monitors while maintaining black-box behavior, and may repurpose tokens to have different internal representations. It does not state a norm-rescale route or a pre/post RMSNorm design rule. Source: https://arxiv.org/abs/2506.14261
- `scale invariance normalized neural networks gradient orthogonal input RMSNorm arxiv` -> Zhang and Sennrich 2019, RMSNorm. Abstract states RMSNorm gives re-scaling invariance. This supports the mathematical mechanism, but is not a monitor-evasion result. Source: https://arxiv.org/abs/1910.07467
- Exact novelty searches for `"RMSNorm" "linear probe" "rescale" "evade" OR "evasion"` and `"pre-norm" "probe" "RMSNorm" "activations" evasion` failed due search-provider bot blocks. I did not use these failures as evidence.

Current novelty conclusion: public work covers (a) latent-space probe monitoring, (b) obfuscation/evasion of latent monitors, and (c) RMSNorm scale invariance. I did not find a public result tying these together into the specific claim tested here: gradient fine-tuning against a pre-norm final-layer probe using the zero-output-cost radial rescale, with a post-norm control.
