# Hugo (skeptic2) novelty check — refit-probe

Searched independently (web search, 2026-10-19). What the closest work already reports:

- **Libon et al., arXiv 2609.38645** ("Alignment via Training Against Probes Without Losing Monitorability"): linear probes fitted from scratch on the trained models reach held-out AUROC 0.85-0.90 (harmfulness) and 0.98-0.99 (dishonesty), close to the base models. Their main regime uses continuously updated probes and 8B/7B-scale models; no label-count curve; no held-text control. https://arxiv.org/abs/2609.38645
- **Wehner & Fritz, arXiv 2510.21531**: "simply retraining probes after optimization recovers high detection accuracy", for toxicity with probe-based SFT/DPO. Not deception, no label-count curve. https://arxiv.org/abs/2510.21531
- **Cundy & Gleave, arXiv 2505.13787** (DolusChat): a frozen lie detector can still catch an evading policy's lies with a lower decision boundary, at more false alarms. Frozen detector only.
- **Taufeeque et al. (Obfuscation Atlas), arXiv 2602.15515**, App. L: continual retraining during RL gave no significant help, attributed to off-domain refit data.
- Our own probe-ensemble run: an OFF-policy attention-probe refit gave 0.72-0.82 vs base 0.79.

**New here:** an ON-policy refit, with a label-count curve (25-400), after DPO against a FROZEN probe, on deception, plus the held-text control (the untrained model reading the trained model's answers), which shows the refit's gap to base comes from the answers, not the activations. The result also runs against the Libon headline in direction: on a 0.5B model with an LLM-judge label, the refit does not reach base on one of four runs (0.72 vs 0.82). Scale, label noise (judge-vs-haiku agreement 0.82-0.86) and the absolute ceiling (base 0.82, not 0.98) differ from Libon, so the post must not frame it as a contradiction of that paper.
