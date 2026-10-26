# Novelty check — Vera (skeptic), room oracle-1026-audit-penalty

Claim checked: a LoRA on the last two blocks of Qwen3-1.7B is trained AGAINST a causal head-swap audit (the
/probe-switch-heads audit: swap a head's o_proj-input slice from the unprobed run), to see whether a probe-triggered
switch keeps firing while that audit stops finding it.

Searched 2026-10-25 ~18:15 PDT (web search + arXiv abstract pages), independent of Iris's research.md.

| Work | What it does | Same as ours? |
|---|---|---|
| Heo, Joo & Moon 2019, arXiv 1902.02041 | Fine-tunes CNNs with the interpretation map in the loss; fools LRP / Grad-CAM | Method ancestor (interpretation in the loss). Saliency maps, not causal patching; vision, not LMs. |
| Bailey et al., arXiv 2412.09565 (Obfuscated Activations) | White-box attacker trains activations to fool latent probes / OOD monitors | Probes, not a head-level causal audit. |
| Neural Chameleons, arXiv 2512.11949 | Text-triggered evasion of unseen activation monitors | Probes, text trigger. |
| Florencio & Barton, arXiv 2506.18053 | Architectural obfuscation (permutations / linear maps) vs logit lens, path patching, head ablation in GPT-2 small | Fixed obfuscation map, not an adaptive attacker trained against the audit. |
| Dingeto, arXiv 2609.21996 (Probe of Internal Recognition) | Trains an "anti-probe organism" that drives a readout margin to zero | Readout/probe, not a head-swap audit. |
| Frank, arXiv 2604.04385 (How Alignment Routes) | Interchange (swap) screening finds gates that per-head ablation misses | Audit method comparison on untrained-to-evade models. |
| Chhabra, Zhu & Khalili, arXiv 2503.01896 | Toxic fine-tuning corrupts IOI mechanisms; re-training restores them ("neuroplasticity") | Mechanism change under fine-tuning, no audit in the loss. |

Verdict: I found no public work that puts a causal head-swap (activation-patching) audit score into the training loss
of a language model and measures whether the behaviour survives. "First adaptive evasion of an interpretability
audit" would be false (Heo et al. 2019). A safe scope: "an adaptive test of a head-swap audit on a numeric
(probe-sensing) trigger". This matches Iris's scope line.
