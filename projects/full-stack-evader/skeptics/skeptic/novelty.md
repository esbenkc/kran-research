# Vera — novelty check (theory-1021-full-stack-evader)

Claim under test (target (a) so far): a LoRA fine-tune on all 28 blocks, aimed at one clean-fit country-token probe at
layer 18, with an output-matching term, removes that probe's separation (AUROC 0.53-0.57, catch 2.5%) but the same
clean-fit probe at layer 19 still catches 78% and layers 20-28 catch 94-98%: a one-layer blind spot.

| prior work | what it shows | overlap with our claim |
|---|---|---|
| Jesse Li, "Toy Model of Activation Obfuscation", LessWrong 2026-08-14 (https://www.greaterwrong.com/posts/a5hkSyi9JDjkMdNBa/toy-model-of-activation-obfuscation) | toy residual MLP trained against a re-trained logreg probe at layer 2: logreg fails at layer 2 (AUROC 0.62), "accuracy quickly recovers in later layers"; training against probes on several layers is Pareto-worse for the model | SAME qualitative picture (blind at the trained layer, readable again right after), in a toy model with an adaptive probe. Not in research.md as of 16:30. The post should credit it. |
| McGuinness, Serrano, Bailey, Emmons, Neural Chameleons, arXiv 2512.11949 (Fig. 5) | activation shift peaks at the monitored layer 12, "localized mechanism rather than a whole-network effect" | agrees: localized change. They measure shift norm, not clean-fit catch per layer; trigger-conditioned, generated text. |
| Bailey et al., Obfuscated Activations Bypass LLM Latent-Space Defenses, arXiv 2412.09565 (sec. 5) | obfuscating against monitors over more layers is harder and costs task performance; attacks transfer across probe types (logreg -> MLP) | our (a) shows little transfer across LAYERS for a full-stack attacker; their transfer is across probe families at the same layers. |
| Patrawala, Feng, Jones, Steinhardt, LLM Layers Immediately Correct Each Other, arXiv 2609.07876 | adjacent layers systematically oppose each other's contributions | a possible mechanism for 'the next block puts the signal back'; context only, not tested here. |
| Westphal et al., arXiv 2606.09411; Libon et al., arXiv 2609.38645 | attack all tracked layers with a KL anchor; retrained probes recover | not the single-target transfer question. |
| Gupta & Jenner, RL-Obfuscation, arXiv 2506.14261 | under RL on generated text, max-pool and attention probes "remain robust" while token-level probes fall | our (b) arm runs the other way: with the probe's own gradient, the attention megaprobe falls (3.3%) while every plain probe stays (97-100%). Different attacker (white-box gradient vs RL). |

Verdict (final): the one-layer blind spot is NOT new as a phenomenon (toy model: Li 2026; localized shift: McGuinness
et al. 2025). New here: the measurement in a 1.5B LLM with every block editable and outputs held on the monitored
statements, clean-fit probes re-thresholded per layer; the megaprobe-target arm (attention probe falls first, plain
probes stay); and the lambda-0 control (without the output term every probe falls, at KL 0.46). Wording in the post
(commit 93c3022): 'agrees with a toy model by Jesse Li and with Neural Chameleons' and Gupta & Jenner as contrast,
no 'first' claim. Checked: arXiv 2512.11949 abstract (McGuinness, Serrano, Bailey, Emmons), 2609.07876 (Patrawala,
Feng, Jones, Steinhardt), the LessWrong post text (AUROC 0.62 quote).
