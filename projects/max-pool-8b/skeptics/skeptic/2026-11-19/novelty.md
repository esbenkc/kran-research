# Novelty check (Vera, skeptic) — max-pool-8b

Question: at Qwen3-8B, after a LoRA fine-tune (weights only, fixed text) lowers an averaging (mean-pooled) truth probe's
score on false statements, does a max-pooled tokenwise probe that the attacker never saw still catch them at 1% false
alarms (alarm line refit on the attacked model)?

| work | what it tests | overlap with ours | does it already answer the question? |
|---|---|---|---|
| RL-Obfuscation, Gupta & Jenner, arXiv 2506.14261 | RL fine-tune of the policy to evade latent monitors while keeping behaviour; the model may change the TEXT it writes | max-pool and attention probes were robust; mean-aggregated probes were evaded ("apart from Attention probes, all probes are evaded by the model with mean aggregation", html v4) | No. Their attacker changes generated tokens and trains against each monitor type; ours keeps the text fixed and attacks only the mean probe, then tests an UNSEEN max probe. Their result is the field claim our run tests on a different threat model. |
| Neural Chameleons, arXiv 2512.11949 | fine-tune to evade UNSEEN monitors of other concepts on a trigger (Gemma-2-9b and others) | unseen-monitor evasion | No. Token scores are aggregated by the MEAN (their eq. 8, App. C.3); max is only over layers (eq. 9). No max-over-tokens monitor. |
| Bailey et al., Obfuscated activations, arXiv 2412.09565 | embedding and weight attacks against latent defenses | the softplus-on-probe-score attack loss our recipe uses | Not at 8B on fixed-text truth statements with an unseen max-over-tokens probe (as far as I found). |
| Locating and Steering Refusal Beyond Attention, arXiv 2609.04721 | refusal direction across transformer and SSM | states "the break is a property of mean pooling" for a harm probe at a 1% benign false-positive rate | No. Prompt-level evasion of a refusal/harm probe, not a weight attack on a truth probe. |
| Operating-Point Failure of Frozen Guardrail Probes, arXiv 2611.19216 | frozen probe + threshold across 16 public model updates | why we refit the line on every checked model (the frozen line's false-alarm rate moves 1% -> 15% with AUROC nearly unchanged) | No; supports the re-threshold design choice. |
| Our own: max-pool-survives (Qwen2.5-1.5B, theory-1019) | same recipe at 1.5B | parent: max14 0.0% [0.0, 0.9] after the attack, clean 91.4% | This run is its scale replication at 8B; new only in model size and the random-direction control. |
| Our own: panel-fine-tune-8b (Qwen3-8B, oracle-1118) | fine-tune against a 20-layer country-token panel with output-KL 1e4 | 8B, same city data | Different monitor (country-token readout, max over LAYERS, not over tokens) and a different attack (panel, KL on). Does not answer ours. |

Verdict: new as a scale replication. The claim it can make is narrow: one model (Qwen3-8B), one dataset (Geometry of
Truth cities, arXiv 2310.06824), one attack recipe (lambda 0, 150 or 300 steps). It does NOT test the RL-Obfuscation
setting (the attacker there may change text), so it cannot contradict that paper; it can only say that the
max-pool protection reported there does not carry over (or does carry over) to a fixed-text weight attack at 8B.
