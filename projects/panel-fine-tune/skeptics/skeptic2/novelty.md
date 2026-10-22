# Novelty check — skeptic2

Searches I ran:

- `"max over layers" probes evasion fine-tune KL linear probe monitor`
- `Bailey 2412.09565 attack mean over layers probes SQLEval Fig 15`
- `Westphal 2606.09411 probe attack all layers any layer monitor`
- `Libon 2609.38645 dishonesty probe loss layers 20 35`

Closest prior work:

1. Bailey et al., *Obfuscated Activations Bypass LLM Latent-Space Defenses*, arXiv:2412.09565. The closest item is Fig. 15. The paper says that obfuscating against more layers reduces SQLEval performance, and Table 2 reports negative correlations between number of obfuscated layers and SQLEval performance. It studies obfuscation across layers, but not this fixed max-over-layer country-probe panel on a weight fine-tune with a scored-model alarm line. Source: https://arxiv.org/html/2412.09565, lines found by grep: Fig. 15 at 1101-1105 and text at 1117-1127.
2. Westphal et al., *Now You (Still) See Me: Detecting Evasive Steganographic Payloads in LLMs*, arXiv:2606.09411. The abstract says adversarial fine-tuning can evade ridge and held-out MLP probes while keeping 58-79% exact-match secret recovery, with 1-8% average capability degradation. This is a multi-probe evasion result in steganography, not this max-over-layer country-probe panel. Source: https://arxiv.org/abs/2606.09411.
3. Libon et al., *Alignment via Training Against Probes Without Losing Monitorability*, arXiv:2609.38645. The abstract says fixed probes are easily exploitable and continuously updated probes preserve monitorability better. It studies probe-guided fine-tuning, but not this fixed max aggregation or the line-rise failure mode. Source: https://arxiv.org/abs/2609.38645.

My novelty verdict: I found related multi-layer probe evasion and probe-guided fine-tuning papers, but not the exact claim tested here: a Qwen2.5-1.5B LoRA fine-tune trained against a fixed max-over-layer country-probe panel, scored with one alarm line reset on the fine-tuned model's true statements. The new result is the line-calibration failure: the max-panel catches 3.75% while member probe tok11 still catches 72.9% at its own line.
